"""The Iris-style shader options system.

A shaderpack exposes user options as GLSL declarations carrying an allowed-value
comment, plus bare toggles referenced from the menu:

    #define SHADOW                                    // a toggle (on when uncommented)
    #define AO_STRENGTH 1.00 //[0.25 0.50 ... 2.00]   // a slider / enum
    const int shadowMapResolution = 2048; //[512 1024 2048 4096]

:class:`ShaderOptions` discovers these, tracks the current value of each, applies
named *profiles* (``profile.MEDIUM=profile.LOW AO !TAA shadowDistance=192.0``),
and rewrites the pack's source so the chosen values take effect — the same
mechanism OptiFine/Iris use. It also exposes the full menu tree (labels, values,
ranges, screen placement) as plain data for a config file or a future GUI, and
round-trips a simple ``name=value`` option file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..pack.properties import Properties

__all__ = ["Option", "ShaderOptions"]


@dataclass
class Option:
    name: str
    kind: str                    # "toggle" | "enum" | "const"
    default: object              # bool for toggle; str token otherwise
    allowed: list[str] = field(default_factory=list)
    const_type: str | None = None  # e.g. "int", "float" (const options only)
    source_file: str = ""


# Declaration matchers (applied per line).
_CONST = re.compile(
    r"^(\s*)const\s+(\w+)\s+(\w+)\s*=\s*([^;]+);(.*)$"
)
_DEFINE = re.compile(
    r"^(\s*)(//\s*)?#define\s+(\w+)\b[ \t]*(\S+)?(.*)$"
)
_ALLOWED = re.compile(r"//\s*\[([^\]]*)\]")


class ShaderOptions:
    def __init__(self, options: dict[str, Option], values: dict[str, object]):
        self.options = options
        self._values = values

    # -- discovery -------------------------------------------------------
    @classmethod
    def from_pack(cls, sources: dict[str, str], props: Properties | None = None) -> "ShaderOptions":
        """Discover options from ``sources`` (``path -> text``).

        ``props`` qualifies bare toggles: a value-less ``#define X`` is only
        treated as an option if ``X`` is named in the menu (screens/sliders/
        profiles), which avoids sweeping up internal defines.
        """
        menu = _menu_names(props) if props else None
        options: dict[str, Option] = {}
        values: dict[str, object] = {}

        for path, text in sources.items():
            for line in text.splitlines():
                opt = _parse_option_line(line, path, menu)
                if opt is None or opt.name in options:
                    continue
                options[opt.name] = opt
                values[opt.name] = opt.default

        return cls(options, values)

    # -- values ----------------------------------------------------------
    def get(self, name: str) -> object:
        return self._values[name]

    def set(self, name: str, value: object) -> None:
        if name not in self.options:
            raise KeyError(f"unknown option {name!r}")
        opt = self.options[name]
        if opt.kind == "toggle":
            self._values[name] = _as_bool(value)
        else:
            token = _fmt_token(value)
            if opt.allowed and token not in opt.allowed:
                raise ValueError(
                    f"{name}={token!r} not allowed; choose from {opt.allowed}"
                )
            self._values[name] = token

    def values(self) -> dict[str, object]:
        return dict(self._values)

    def reset(self) -> None:
        self._values = {name: opt.default for name, opt in self.options.items()}

    # -- profiles --------------------------------------------------------
    def apply_profile(self, name: str, props: Properties) -> None:
        """Apply a named profile from ``shaders.properties`` (recursively)."""
        self.reset()
        self._apply_profile(name, props, frozenset())

    def _apply_profile(self, name: str, props: Properties, seen: frozenset[str]) -> None:
        if name in seen or name not in props.profiles:
            return
        for token in props.profiles[name].split():
            if token.startswith("profile."):
                self._apply_profile(token[len("profile."):], props, seen | {name})
            elif token.startswith("!"):
                opt = token[1:]
                if opt in self.options:
                    self._values[opt] = False
            elif "=" in token:
                key, _, val = token.partition("=")
                if key in self.options:
                    try:
                        self.set(key, val)
                    except ValueError:
                        self._values[key] = val  # profiles may exceed the slider list
            elif token in self.options:
                self._values[token] = True

    # -- application to source ------------------------------------------
    def rewrite_source(self, text: str) -> str:
        """Rewrite every option declaration in ``text`` to its current value."""
        return "\n".join(self._rewrite_line(line) for line in text.splitlines())

    def rewrite_files(self, files: dict[str, str]) -> dict[str, str]:
        """Return ``files`` with option-declaring files rewritten in place."""
        touched = {opt.source_file for opt in self.options.values()}
        return {
            path: (self.rewrite_source(text) if path in touched else text)
            for path, text in files.items()
        }

    def _rewrite_line(self, line: str) -> str:
        if m := _CONST.match(line):
            indent, ctype, name, _value, tail = m.groups()
            if name in self.options and self.options[name].kind == "const":
                return f"{indent}const {ctype} {name} = {self._values[name]};{tail}"
            return line
        if m := _DEFINE.match(line):
            indent, commented, name, value, tail = m.groups()
            if name not in self.options:
                return line
            opt = self.options[name]
            if opt.kind == "toggle":
                on = _as_bool(self._values[name])
                prefix = "" if on else "//"
                return f"{indent}{prefix}#define {name}{tail}"
            # enum/param define keeps its trailing //[..] comment.
            return f"{indent}#define {name} {self._values[name]}{tail}"
        return line

    # -- menu tree / persistence ----------------------------------------
    def menu_tree(self, props: Properties, name: str = "") -> dict:
        """The screen tree with each option annotated by value/allowed/kind."""
        base = props.screen_tree(name)
        return self._annotate(base, props)

    def _annotate(self, node: dict, props: Properties) -> dict:
        out = {"name": node["name"], "elements": []}
        for el in node["elements"]:
            if el["type"] == "option" and el["name"] in self.options:
                opt = self.options[el["name"]]
                out["elements"].append({
                    "type": "option", "name": el["name"], "kind": opt.kind,
                    "value": self._values[el["name"]], "allowed": opt.allowed,
                    "is_slider": el["name"] in props.sliders,
                })
            elif el["type"] == "screen":
                out["elements"].append({
                    "type": "screen", "name": el["name"],
                    "screen": self._annotate(el["screen"], props),
                })
            else:
                out["elements"].append(el)
        return out

    def dumps(self) -> str:
        """Serialise current values to an option-file string."""
        lines = []
        for name in sorted(self._values):
            val = self._values[name]
            lines.append(f"{name}={_fmt_token(val) if not isinstance(val, bool) else str(val).lower()}")
        return "\n".join(lines) + "\n"

    def loads(self, text: str) -> None:
        """Load values from an option-file string (unknown keys ignored)."""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key not in self.options:
                continue
            if self.options[key].kind == "toggle":
                self._values[key] = _as_bool(val)
            else:
                self._values[key] = val


# -- helpers -------------------------------------------------------------
def _menu_names(props: Properties) -> set[str]:
    names: set[str] = set(props.sliders)
    for tokens in props.screens.values():
        for tok in tokens:
            if tok == "<empty>" or (tok.startswith("[") and tok.endswith("]")):
                continue
            if tok.startswith("<") and tok.endswith(">"):
                continue
            names.add(tok)
    for expr in props.profiles.values():
        for tok in expr.split():
            tok = tok.lstrip("!")
            if "=" in tok:
                tok = tok.split("=", 1)[0]
            if tok and not tok.startswith("profile."):
                names.add(tok)
    return names


def _parse_option_line(line: str, path: str, menu: set[str] | None) -> Option | None:
    if m := _CONST.match(line):
        _indent, ctype, name, value, tail = m.groups()
        allowed_m = _ALLOWED.search(tail)
        if not allowed_m:
            return None  # only const declarations with an allowed-list are options
        return Option(name, "const", value.strip(),
                      allowed=allowed_m.group(1).split(), const_type=ctype,
                      source_file=path)

    if m := _DEFINE.match(line):
        _indent, commented, name, value, tail = m.groups()
        allowed_m = _ALLOWED.search(tail or "")
        if value is not None and not value.startswith("//"):
            if not allowed_m:
                return None  # a value #define with no allowed-list isn't configurable
            return Option(name, "enum", value,
                          allowed=allowed_m.group(1).split(), source_file=path)
        # Bare toggle: qualify against the menu, default on when uncommented.
        if menu is not None and name not in menu:
            return None
        return Option(name, "toggle", commented is None, source_file=path)

    return None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "on", "yes")


def _fmt_token(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
