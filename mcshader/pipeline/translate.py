"""Translate one pipeline pass to engine-ready GLSL, with options applied.

This is the bridge the runner leans on:

    options.rewrite_files -> resolve #include -> select stage
    -> (fragment) find the active DRAWBUFFERS set -> map gl_FragData[k] to colortex[k]
    -> translate to the target dialect

Keeping it here (engine-independent) means it can be unit-tested without a GL
context and reused by any backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config.options import ShaderOptions
from ..glsl import preprocess
from ..glsl.dialect import translate_stage
from ..pack.loader import ShaderPack
from .graph import active_outputs

__all__ = ["TranslatedPass", "translate_pass"]


@dataclass
class TranslatedPass:
    name: str
    world: str
    vertex: str | None
    fragment: str | None
    outputs: list[int] = field(default_factory=list)  # active colortex targets
    mc_uniforms: list[str] = field(default_factory=list)


def translate_pass(
    pack: ShaderPack,
    options: ShaderOptions,
    name: str,
    *,
    world: str = "world0",
    target: str = "panda3d",
    mode: str = "colortex",
    gbuffer_map: dict[int, int] | None = None,
    max_location: int | None = None,
) -> TranslatedPass:
    """Produce translated vertex/fragment sources for one program.

    Applies the current option values first (so ``#if`` branches and slider
    values resolve to the user's choices), then routes fragment outputs to
    framebuffer attachment locations according to ``mode``:

    * ``"colortex"`` (default): ``gl_FragData[k]`` -> location ``colortex[k]`` — the
      literal colortex index. Correct when a GPU has an attachment per colortex.
    * ``"gbuffer"``: location = ``gbuffer_map[colortex]`` — packs the shared
      gbuffer FBO's distinct colortex into a compact attachment range (fits GPUs
      with only 8 color attachments).
    * ``"compact"``: location = ``k`` — for a single fullscreen pass whose small
      FBO binds attachment ``k`` to ``colortex[outputs[k]]``.
    """
    files = options.rewrite_files(pack.files)
    program = pack.read_program(name, world=world)
    base_dir = pack.base_dir or "."

    def prep(raw: str, stage: str) -> str:
        # Stage selection happens *inside* include expansion (see
        # `resolve_includes`'s `stage` argument): an Iris program is one
        # combined source with an `#ifdef VSH` and an `#ifdef FSH` block, and
        # both blocks legitimately include the same library — each compiled
        # stage needs its own copy, so the expand-once bookkeeping has to be
        # per stage, not per file. The trailing `select_stage` is a no-op for
        # a stage-aware expansion and is kept only so a caller passing
        # already-expanded source still gets the selection applied.
        resolved = preprocess.resolve_includes(
            raw, base_dir=base_dir, files=files, stage=stage)
        return preprocess.select_stage(resolved, stage)

    vert_raw = program.vertex or program.combined
    frag_raw = program.fragment or program.combined

    result = TranslatedPass(name=name, world=world, vertex=None, fragment=None)
    used: set[str] = set()

    if vert_raw is not None:
        tr = translate_stage(prep(vert_raw, "vertex"), "vertex", target)
        result.vertex = tr.source
        used.update(tr.mc_uniforms)

    if frag_raw is not None:
        frag_src = prep(frag_raw, "fragment")
        outputs = active_outputs(frag_src)
        result.outputs = outputs
        tr = translate_stage(
            frag_src, "fragment", target,
            frag_output_map=_output_map(frag_src, outputs, mode, gbuffer_map, max_location),
        )
        result.fragment = tr.source
        used.update(tr.mc_uniforms)

    result.mc_uniforms = sorted(used)
    return result


_FRAGDATA = re.compile(r"gl_FragData\s*\[\s*(\d+)\s*\]")


def _output_map(
    frag_src: str,
    outputs: list[int],
    mode: str = "colortex",
    gbuffer_map: dict[int, int] | None = None,
    max_location: int | None = None,
) -> dict[int, int]:
    """Map each ``gl_FragData[k]`` to a unique attachment location.

    The preferred location depends on ``mode`` (see :func:`translate_pass`).
    Subscripts beyond the active DRAWBUFFERS length — leftovers from #if branches
    this pre-preprocessor stage can't yet resolve — collisions, and any location
    at/above ``max_location`` fall back to the next free in-range location, so two
    outputs never share a ``layout(location=)`` and no output exceeds the GPU's
    draw-buffer count. Both keep the shader linkable.
    """
    gbuffer_map = gbuffer_map or {}
    ceiling = max_location if max_location is not None else 64
    subs = sorted({int(m) for m in _FRAGDATA.findall(frag_src)})
    used: set[int] = set()
    omap: dict[int, int] = {}
    for k in subs:
        colortex = outputs[k] if k < len(outputs) else None
        if mode == "compact":
            loc = k
        elif mode == "gbuffer":
            loc = gbuffer_map.get(colortex) if colortex is not None else None
        else:  # "colortex"
            loc = colortex
        if loc is None or loc in used or loc >= ceiling:
            loc = next((i for i in range(ceiling) if i not in used), None)
        if loc is None:  # no free slot left; reuse the last (rare, dead branches)
            loc = ceiling - 1
        omap[k] = loc
        used.add(loc)
    return omap
