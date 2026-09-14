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
    _seen: set[str] | None = None,
    stage: str | None = None,
) -> str:
    """Recursively expand ``#include`` directives, OptiFine/Iris-style.

    A path beginning with ``/`` is pack-root-relative; any other path is relative
    to the *including file's* directory (``current_dir``). This is why packs can
    ``#include "lightColor.glsl"`` from a sibling in ``lib/color/``. Lookups hit
    the in-memory ``files`` map first, then the filesystem under ``base_dir``.
    Unresolvable includes are left in place so the failure is visible.

    ``_seen`` tracks every pack-relative path already expanded *anywhere* in
    this call tree, not just on the current include chain — so it catches true
    cycles (A includes B includes A) *and* the far more common case of a
    "diamond" include graph, where two unrelated files both include the same
    shared library file. Iris's own `#include` genuinely behaves this way
    (each file expands at most once per compiled program, like `#pragma
    once`); textually re-expanding it per include site — the previous
    behavior, which only tracked ancestors on the current branch — is not
    just wasteful but was observed multiplying a real pack's combined source
    into 10,000+ line files (a `lib/` file included from several siblings,
    each of which is itself included from several gbuffers/composite
    programs), which is exactly the kind of size that risks tipping a GLSL
    compiler into truncating output or timing out entirely — a very plausible
    cause of intermittent "vertex shader lacks main" link failures seen on a
    pack with a much deeper include graph than the ones this was built
    against. A library file double-included under the old behavior was
    already inert in practice (real packs guard their own library files with
    `#ifndef X_GLSL`/`#define X_GLSL`/`#endif` precisely because `#include`
    doesn't dedupe on its own in C-derived preprocessors) — deduping here
    only removes the redundant text, it doesn't change what the shader
    compiles to.
    """
    if _seen is None:
        _seen = set()
    if stage is not None:
        # Drop the other stage's code *before* scanning this file for
        # includes, so `_seen` only ever records what this stage actually
        # compiles. Without it the expand-once rule is applied across the
        # combined VSH+FSH source, and whichever stage block comes first in
        # the file claims the single expansion of any library both blocks
        # include — then `select_stage` deletes that block for the other
        # stage, taking the library's only copy with it. Real symptom: with
        # `TAA` on, gbuffers_terrain.glsl includes lib/util/jitter.glsl from
        # both its FSH and (transitively) its VSH block; the vertex shader
        # ended up referencing `TAAJitter` with no definition and failed to
        # compile, so the whole scene silently fell back to Panda3D's own
        # default shading. Applying the selection per file (rather than once
        # at the end) is what makes "expand at most once per compiled
        # program" mean the right thing — a program here is one *stage*.
        source = select_stage(source, stage)
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
            out.append(f"// [mcshader] skipped already-included file: {path}")
            continue
        _seen.add(key)  # mark before recursing so a cycle back to `key` also hits the check above

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
            included, base_dir, files, posixpath.dirname(key), _seen, stage))
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
