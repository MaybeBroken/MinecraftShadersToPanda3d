"""GLSL source preprocessing for Minecraft (OptiFine/Iris) shaderpacks.

Two purely-textual jobs (no GL context required):

* :func:`resolve_includes` expands ``#include`` the way OptiFine does
  (pack-root-relative, quoted or angle-bracketed).
* :func:`select_stage` performs the stage separation Iris packs rely on. Those
  packs ship one combined ``.glsl`` guarded by ``#ifdef VSH`` / ``#ifdef FSH``,
  with tiny ``.vsh``/``.fsh`` stubs that ``#define`` one of them and include the
  combined file. ``select_stage`` evaluates *only* the VSH/FSH conditionals —
  keeping code shared between the two stages, dropping the other stage's blocks,
  and leaving every other preprocessor directive untouched for the GLSL
  compiler.
"""

from __future__ import annotations

import os
import posixpath
import re

__all__ = ["resolve_includes", "select_stage"]

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+["<]([^">]+)[">]')


def resolve_includes(
    source: str,
    base_dir: str = ".",
    files: dict[str, str] | None = None,
    current_dir: str = "",
    _seen: frozenset[str] = frozenset(),
) -> str:
    """Recursively expand ``#include`` directives, OptiFine-style.

    A path beginning with ``/`` is pack-root-relative; any other path is relative
    to the *including file's* directory (``current_dir``). This is why packs can
    ``#include "lightColor.glsl"`` from a sibling in ``lib/color/``. Lookups hit
    the in-memory ``files`` map first, then the filesystem under ``base_dir``.
    Unresolvable includes are left in place so the failure is visible; cycles are
    broken.
    """
    files = files or {}
    out: list[str] = []
    for line in source.splitlines():
        match = _INCLUDE_RE.match(line)
        if not match:
            out.append(line)
            continue

        path = match.group(1)
        if path.startswith("/"):
            key = path.lstrip("/")
        else:
            key = posixpath.normpath(posixpath.join(current_dir, path)).lstrip("./")

        if key in _seen:
            out.append(f"// [mcshader] skipped cyclic include: {path}")
            continue

        included = None
        for candidate in (key, path, path.lstrip("/")):
            if candidate in files:
                included = files[candidate]
                break
        if included is None:
            try:
                with open(os.path.join(base_dir, key), "r", encoding="utf-8") as handle:
                    included = handle.read()
            except OSError:
                out.append(line)  # keep original so the failure surfaces
                continue

        out.append(resolve_includes(
            included, base_dir, files, posixpath.dirname(key), _seen | {key}))
    return "\n".join(out)


_IFDEF = re.compile(r"^\s*#\s*ifdef\s+(\w+)\b")
_IFNDEF = re.compile(r"^\s*#\s*ifndef\s+(\w+)\b")
_IF = re.compile(r"^\s*#\s*if\s+(.*)$")
_ANY_OPEN = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
_ELSE = re.compile(r"^\s*#\s*else\b")
_ENDIF = re.compile(r"^\s*#\s*endif\b")


def _stage_macro(line: str) -> str | None:
    """Return 'VSH'/'FSH' if the directive opens on that macro, else None."""
    if m := _IFDEF.match(line):
        return m.group(1) if m.group(1) in ("VSH", "FSH") else None
    if m := _IF.match(line):
        expr = m.group(1)
        if re.search(r"\bVSH\b", expr):
            return "VSH"
        if re.search(r"\bFSH\b", expr):
            return "FSH"
    return None


def select_stage(source: str, stage: str) -> str:
    """Resolve ``#ifdef VSH``/``#ifdef FSH`` for one ``stage``.

    ``stage`` is ``"vertex"`` (keep VSH) or ``"fragment"`` (keep FSH). Content
    shared between stages is preserved; the other stage's guarded blocks are
    removed; all non-stage directives pass through unchanged.
    """
    keep = "VSH" if stage == "vertex" else "FSH"
    drop = "FSH" if stage == "vertex" else "VSH"

    out: list[str] = []
    stack: list[str] = []  # 'keep' -> strip its #endif; 'pass' -> emit its #endif
    skip = 0               # nesting depth while dropping a block

    for raw in source.splitlines():
        # Inside a dropped block: track nesting to find its close.
        if skip:
            if _ANY_OPEN.match(raw):
                skip += 1
            elif _ENDIF.match(raw):
                skip -= 1
            elif skip == 1 and _ELSE.match(raw):
                skip = 0          # the #else opens the branch we DO want
                stack.append("keep")
            continue

        # #ifndef VSH/FSH inverts keep/drop.
        if m := _IFNDEF.match(raw):
            if m.group(1) == drop:
                stack.append("keep")
                continue
            if m.group(1) == keep:
                skip = 1
                continue

        macro = _stage_macro(raw)
        if macro == drop:
            skip = 1
            continue
        if macro == keep:
            stack.append("keep")
            continue

        if _ANY_OPEN.match(raw):
            stack.append("pass")
            out.append(raw)
            continue
        if _ELSE.match(raw):
            if stack and stack[-1] == "keep":
                stack.pop()       # #else of a kept stage guard -> drop other branch
                skip = 1
            else:
                out.append(raw)
            continue
        if _ENDIF.match(raw):
            if stack and stack.pop() == "keep":
                continue          # strip the stage guard's #endif
            out.append(raw)
            continue

        out.append(raw)

    return "\n".join(out)
