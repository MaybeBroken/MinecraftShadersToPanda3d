"""Parsers for the ``*.properties`` metadata that drives a shaderpack.

Three kinds of file share the OptiFine/Iris ``.properties`` grammar (``key=value``,
``#`` comments, trailing ``\\`` line continuations):

* ``shaders.properties`` — profiles, the ``screen.*`` menu tree, which options are
  sliders, per-program enable conditions, and forced video settings.
* ``block.properties`` / ``item.properties`` / ``entity.properties`` — map Minecraft
  ids (``minecraft:grass_block`` ...) to the numeric render-category id a shader reads
  from ``mc_Entity.x`` (e.g. ``block.10500`` groups all leaves).

Everything here is plain text parsing; nothing touches OpenGL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "Properties",
    "BlockMapping",
    "parse_properties",
    "parse_block_mapping",
    "read_raw",
    "read_pairs",
    "eval_condition",
]


# -- boolean/numeric condition evaluation ---------------------------------
# Shared by `program.*.enabled` expressions (program.py's enable_expr) AND
# `.properties` files' own `#if`/`#else`/`#endif` preprocessor blocks below
# — both are OptiFine/Iris expressions over the pack's own option names, just
# used in two different places. Lives here (not pipeline/graph.py, the
# original home) so pack/ doesn't have to depend on pipeline/ to preprocess
# its own conditionals; pipeline/graph.py re-imports this same function.
_COND_TOKEN = re.compile(
    r"\s*(\(|\)|&&|\|\||==|!=|<=|>=|<|>|!|[A-Za-z_]\w*|-?\d+\.?\d*)"
)


def eval_condition(expr: str, values: dict[str, object]) -> bool:
    """Evaluate an OptiFine condition: ``SHADOW && !RETRO_FILTER``,
    ``SHADOW_QUALITY == -1``, ``(A || B) && C >= 2``.

    Bare identifiers resolve to their option value truthiness (toggles use
    their bool; numeric/string options are truthy when non-zero/non-empty;
    unknown identifiers are treated as ``False``/``0``) — this is the whole
    grammar `program.*.enabled` needs. Comparison operators
    (``==``/``!=``/``<``/``>``/``<=``/``>=``) additionally let a
    `shaders.properties` ``#if`` block gate on a numeric option's exact
    value (e.g. Complementary's ``#if SHADOW_QUALITY == -1``), which a
    pure-boolean grammar can't express.
    """
    tokens: list[str] = []
    pos = 0
    while pos < len(expr):
        m = _COND_TOKEN.match(expr, pos)
        if not m:
            break
        tokens.append(m.group(1))
        pos = m.end()

    def numeric(tok: str) -> float:
        try:
            return float(tok)
        except ValueError:
            pass
        if tok not in values:
            return 0.0
        val = values[tok]
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 1.0 if val else 0.0  # non-numeric (e.g. enum token): truthy/falsy

    # Recursive-descent: or -> and -> comparison -> unary -> atom. Every
    # level operates on floats (0.0/1.0 for booleans) so a bare identifier
    # and a `==`-comparison compose in the same grammar without a separate
    # boolean/numeric type split.
    i = 0

    def parse_or() -> float:
        nonlocal i
        left = parse_and()
        while i < len(tokens) and tokens[i] == "||":
            i += 1
            right = parse_and()
            left = 1.0 if (left != 0.0 or right != 0.0) else 0.0
        return left

    def parse_and() -> float:
        nonlocal i
        left = parse_comparison()
        while i < len(tokens) and tokens[i] == "&&":
            i += 1
            right = parse_comparison()
            left = 1.0 if (left != 0.0 and right != 0.0) else 0.0
        return left

    def parse_comparison() -> float:
        nonlocal i
        left = parse_unary()
        if i < len(tokens) and tokens[i] in ("==", "!=", "<", ">", "<=", ">="):
            op = tokens[i]
            i += 1
            right = parse_unary()
            result = {
                "==": left == right, "!=": left != right,
                "<": left < right, ">": left > right,
                "<=": left <= right, ">=": left >= right,
            }[op]
            return 1.0 if result else 0.0
        return left

    def parse_unary() -> float:
        nonlocal i
        if i < len(tokens) and tokens[i] == "!":
            i += 1
            return 0.0 if parse_unary() != 0.0 else 1.0
        return parse_atom()

    def parse_atom() -> float:
        nonlocal i
        if i >= len(tokens):
            return 0.0
        tok = tokens[i]
        if tok == "(":
            i += 1
            val = parse_or()
            if i < len(tokens) and tokens[i] == ")":
                i += 1
            return val
        i += 1
        return numeric(tok)

    return parse_or() != 0.0


_IF_RE = re.compile(r"^\s*#\s*if\s+(.*?)\s*$")
_IFDEF_RE = re.compile(r"^\s*#\s*ifdef\s+(\w+)\s*$")
_IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+(\w+)\s*$")
_ELSE_RE = re.compile(r"^\s*#\s*else\b")
_ENDIF_RE = re.compile(r"^\s*#\s*endif\b")


def _strip_conditionals(text: str, values: dict[str, object]) -> str:
    """Resolve ``#if``/``#ifdef``/``#ifndef``/``#else``/``#endif`` blocks in
    a ``.properties`` file, blanking out lines in a branch not taken.

    OptiFine/Iris' own `.properties` grammar supports this (Complementary
    Unbound gates several `program.*.enabled` lines behind ``#if
    SHADOW_QUALITY == -1``-style blocks) — every line was previously read
    unconditionally (``#`` was treated as a plain end-of-line comment
    marker, same as vanilla ``.properties``), which silently applied a
    pack's *disabled-by-a-specific-option-value* overrides as if they were
    unconditional, permanently disabling whatever they gated (confirmed:
    this is why Complementary Unbound's shadow map never rendered — its
    ``shadow.enabled=false`` line only applies when ``SHADOW_QUALITY==-1``,
    not always). ``values`` should be the pack's *current* option values —
    call sites re-resolve this on every rebuild (profile switch, settings
    change), matching OptiFine/Iris re-evaluating it on every shader reload.
    """
    out: list[str] = []
    # (branch_active, parent_active) per nesting level; a line is emitted
    # only when every level on the stack is active.
    stack: list[tuple[bool, bool]] = []

    def active() -> bool:
        return all(a for a, _ in stack)

    for line in text.splitlines():
        if m := _IF_RE.match(line):
            parent = active()
            stack.append((parent and eval_condition(m.group(1), values), parent))
            out.append("")
            continue
        if m := _IFDEF_RE.match(line):
            parent = active()
            stack.append((parent and eval_condition(m.group(1), values), parent))
            out.append("")
            continue
        if m := _IFNDEF_RE.match(line):
            parent = active()
            stack.append((parent and not eval_condition(m.group(1), values), parent))
            out.append("")
            continue
        if _ELSE_RE.match(line):
            if stack:
                taken, parent = stack[-1]
                stack[-1] = (parent and not taken, parent)
            out.append("")
            continue
        if _ENDIF_RE.match(line):
            if stack:
                stack.pop()
            out.append("")
            continue
        out.append(line if active() else "")
    return "\n".join(out)


def read_pairs(text: str, values: dict[str, object] | None = None) -> list[tuple[str, str]]:
    """Parse a ``.properties`` file into ``(key, value)`` pairs, in order.

    Honours ``#``/``//`` comments, blank lines, trailing ``\\`` line
    continuations, and (when ``values`` is given) ``#if``/``#else``/
    ``#endif`` conditional blocks (see :func:`_strip_conditionals`) — without
    ``values``, unknown option identifiers default to falsy/0 in any such
    block's condition, which is usually (not always) the pack's own default.
    Duplicate keys are preserved (block/item files legitimately reuse a
    category id across several lines).
    """
    if "#if" in text or "#ifdef" in text or "#ifndef" in text:
        text = _strip_conditionals(text, values or {})
    pairs: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        i += 1
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        # Join continuations, collapsing the split point to a single space.
        while raw.rstrip().endswith("\\") and i < len(lines):
            raw = raw.rstrip()[:-1].rstrip() + " " + lines[i].strip()
            i += 1
        if "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        pairs.append((key.strip(), value.strip()))
    return pairs


def read_raw(text: str, values: dict[str, object] | None = None) -> dict[str, str]:
    """Parse into ``{key: value}`` (last wins for duplicate keys)."""
    return dict(read_pairs(text, values))


@dataclass
class Properties:
    """Parsed ``shaders.properties``."""

    #: profile name -> raw token list (e.g. "SHADOW shadowMapResolution=1024 !TAA").
    profiles: dict[str, str] = field(default_factory=dict)
    #: screen name ("" is the root) -> ordered element tokens.
    screens: dict[str, list[str]] = field(default_factory=dict)
    #: option names the pack wants rendered as sliders.
    sliders: list[str] = field(default_factory=list)
    #: "<world>/<program>" -> boolean-expression string over option names.
    program_enabled: dict[str, str] = field(default_factory=dict)
    #: forced video settings and any other scalar keys (e.g. dynamicHandLight=true).
    settings: dict[str, str] = field(default_factory=dict)
    #: everything else, verbatim, for features not modelled yet.
    raw: dict[str, str] = field(default_factory=dict)

    def screen_tree(self, name: str = "") -> dict:
        """Recursively expand a screen into ``{name, elements:[...]}``.

        ``[SUB]`` tokens become nested screens; ``<empty>`` becomes a spacer;
        every other token is an option/link leaf. Guards against cyclic or
        missing sub-screens.
        """
        return self._tree(name, frozenset())

    def _tree(self, name: str, seen: frozenset[str]) -> dict:
        node: dict = {"name": name, "elements": []}
        if name in seen:
            return node
        for token in self.screens.get(name, []):
            if token == "<empty>":
                node["elements"].append({"type": "spacer"})
            elif token.startswith("[") and token.endswith("]"):
                sub = token[1:-1]
                node["elements"].append(
                    {"type": "screen", "name": sub,
                     "screen": self._tree(sub, seen | {name})}
                )
            else:
                node["elements"].append({"type": "option", "name": token})
        return node


_SCREEN_RE = re.compile(r"^screen\.(.+)$")
_PROFILE_RE = re.compile(r"^profile\.(.+)$")
_PROGRAM_EN_RE = re.compile(r"^program\.(.+)\.enabled$")


def parse_properties(text: str, values: dict[str, object] | None = None) -> Properties:
    """Parse ``shaders.properties`` text into a :class:`Properties`.

    ``values`` (the pack's current option values) resolves any ``#if``/
    ``#else``/``#endif`` blocks the file uses to conditionally set
    `program.*.enabled`/etc — see :func:`_strip_conditionals`. Omit it (or
    pass values from before the pack's own options were discovered) and
    unknown identifiers in such a block default falsy/0, which usually but
    not always matches the pack's real default.
    """
    raw = read_raw(text, values)
    props = Properties()
    for key, value in raw.items():
        if key == "screen":
            props.screens[""] = value.split()
        elif m := _SCREEN_RE.match(key):
            props.screens[m.group(1)] = value.split()
        elif m := _PROFILE_RE.match(key):
            props.profiles[m.group(1)] = value
        elif key == "sliders":
            props.sliders = value.split()
        elif m := _PROGRAM_EN_RE.match(key):
            props.program_enabled[m.group(1)] = value
        elif key in (
            "dynamicHandLight", "oldHandLight", "oldLighting", "separateAo",
            "underwaterOverlay", "vignette",
        ) or key.startswith("particles."):
            props.settings[key] = value
        else:
            props.raw[key] = value
    return props


@dataclass
class BlockMapping:
    """A Minecraft id -> render-category-id mapping (block/item/entity)."""

    kind: str  # "block" | "item" | "entity"
    #: category id (int) -> list of Minecraft ids assigned to it.
    ids_by_category: dict[int, list[str]] = field(default_factory=dict)
    #: Minecraft id -> category id (the reverse lookup an engine uses at tag time).
    category_by_id: dict[str, int] = field(default_factory=dict)

    def category_of(self, mc_id: str, default: int = 0) -> int:
        """Category id for a Minecraft id (``minecraft:`` prefix optional)."""
        if mc_id in self.category_by_id:
            return self.category_by_id[mc_id]
        if ":" not in mc_id:
            return self.category_by_id.get(f"minecraft:{mc_id}", default)
        return self.category_by_id.get(mc_id.split(":", 1)[1], default)


def parse_block_mapping(text: str, kind: str = "block") -> BlockMapping:
    """Parse a ``block``/``item``/``entity`` properties file.

    Lines look like ``block.10500=minecraft:oak_leaves minecraft:birch_leaves``.
    A category id may legitimately appear on several lines; ids accumulate.
    """
    mapping = BlockMapping(kind=kind)
    prefix = kind + "."
    for key, value in read_pairs(text):
        if not key.startswith(prefix):
            continue
        cat_token = key[len(prefix):]
        try:
            category = int(cat_token)
        except ValueError:
            continue
        ids = [tok for tok in value.split() if tok and not tok.isspace()]
        mapping.ids_by_category.setdefault(category, []).extend(ids)
        for mc_id in ids:
            # Blocks can carry state predicates (name:prop=val); key on the name.
            name = mc_id.split(":properties")[0] if ":properties" in mc_id else mc_id
            mapping.category_by_id[name] = category
    return mapping
