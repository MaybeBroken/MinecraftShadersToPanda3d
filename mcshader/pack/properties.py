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
]


def read_pairs(text: str) -> list[tuple[str, str]]:
    """Parse a ``.properties`` file into ``(key, value)`` pairs, in order.

    Honours ``#``/``//`` comments, blank lines, and trailing ``\\`` line
    continuations. Duplicate keys are preserved (block/item files legitimately
    reuse a category id across several lines).
    """
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


def read_raw(text: str) -> dict[str, str]:
    """Parse into ``{key: value}`` (last wins for duplicate keys)."""
    return dict(read_pairs(text))


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


def parse_properties(text: str) -> Properties:
    """Parse ``shaders.properties`` text into a :class:`Properties`."""
    raw = read_raw(text)
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
