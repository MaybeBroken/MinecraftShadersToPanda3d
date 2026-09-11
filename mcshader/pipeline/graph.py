"""Build the render graph for one dimension of a shaderpack.

A shaderpack is a fixed sequence of passes writing into a shared pool of
``colortex`` buffers:

    shadow (+shadowcomp) -> opaque gbuffers -> deferred* -> translucent gbuffers
    -> composite* -> final

Each fullscreen pass declares which buffers it writes with a ``/* DRAWBUFFERS:0195 */``
comment (or Iris ``RENDERTARGETS: 0,1,9,5``); the pool's formats/clear flags come from
``const int colortexNFormat`` / ``const bool colortexNClear`` declarations. This module
extracts all of that into a :class:`PipelineGraph` — the engine-independent description
the Panda3D runner then instantiates. No OpenGL here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..pack.loader import ShaderPack
from ..pack.properties import eval_condition
from ..glsl import preprocess

__all__ = [
    "Pass", "BufferSet", "PipelineGraph", "build_graph", "eval_condition",
    "active_outputs",
]

# Canonical execution order (Iris/OptiFine). Absent programs are skipped;
# numeric siblings (deferred1.., composite1..) expand automatically.
_SHADOW = ["shadow"]
_SHADOWCOMP = ["shadowcomp"]
_PREPARE = ["prepare"]
_OPAQUE_GBUFFERS = [
    "gbuffers_skybasic", "gbuffers_skytextured", "gbuffers_clouds",
    "gbuffers_basic", "gbuffers_textured", "gbuffers_terrain",
    "gbuffers_block", "gbuffers_entities", "gbuffers_entities_glowing",
    "gbuffers_spidereyes", "gbuffers_armor_glint", "gbuffers_damagedblock",
    "gbuffers_beaconbeam", "gbuffers_hand",
]
_DEFERRED = ["deferred"]
_TRANSLUCENT_GBUFFERS = ["gbuffers_water", "gbuffers_hand_water", "gbuffers_weather"]
_COMPOSITE = ["composite"]
_FINAL = ["final"]

_DRAWBUFFERS = re.compile(r"/\*\s*DRAWBUFFERS\s*:\s*([0-9]+)\s*\*/")
_RENDERTARGETS = re.compile(r"/\*\s*RENDERTARGETS\s*:\s*([0-9,\s]+)\*/")
_FORMAT = re.compile(r"colortex(\d+)Format\s*=\s*(\w+)")
_CLEAR = re.compile(r"colortex(\d+)Clear\s*=\s*(true|false)")


@dataclass
class Pass:
    name: str
    kind: str            # shadow|shadowcomp|prepare|geometry|deferred|composite|final
    world: str
    outputs: list[int] = field(default_factory=list)  # colortex indices written
    enable_expr: str | None = None
    is_fullscreen: bool = True   # geometry passes are False (they draw scene geometry)

    def enabled(self, values: dict[str, object]) -> bool:
        return eval_condition(self.enable_expr, values) if self.enable_expr else True


@dataclass
class BufferSet:
    formats: dict[int, str] = field(default_factory=dict)
    clear: dict[int, bool] = field(default_factory=dict)

    def indices(self) -> list[int]:
        return sorted(self.formats)

    def format_of(self, index: int, default: str = "RGBA16") -> str:
        return self.formats.get(index, default)


@dataclass
class PipelineGraph:
    world: str
    passes: list[Pass]
    buffers: BufferSet

    def enabled_passes(self, values: dict[str, object]) -> list[Pass]:
        return [p for p in self.passes if p.enabled(values)]

    def geometry_passes(self) -> list[Pass]:
        return [p for p in self.passes if p.kind == "geometry"]

    def fullscreen_passes(self) -> list[Pass]:
        return [p for p in self.passes if p.is_fullscreen]


def _numbered(base: str, existing: set[str]) -> list[str]:
    """Expand ``base`` + numbered siblings that exist (composite, composite1..)."""
    out = [base] if base in existing else []
    n = 1
    while f"{base}{n}" in existing:
        out.append(f"{base}{n}")
        n += 1
    return out


def _fragment_text(pack: ShaderPack, name: str, world: str) -> str:
    """Fully-resolved fragment source for a program (includes expanded)."""
    try:
        prog = pack.read_program(name, world=world)
    except KeyError:
        return ""
    raw = prog.fragment or prog.combined
    if raw is None:
        return ""
    resolved = preprocess.resolve_includes(raw, base_dir=pack.base_dir or ".", files=pack.files)
    return preprocess.select_stage(resolved, "fragment")


def active_outputs(fragment_text: str) -> list[int]:
    """The active DRAWBUFFERS/RENDERTARGETS set for already-preprocessed source.

    The runner calls this after applying options + resolving ``#if`` branches,
    so exactly one directive remains; the last one wins if several survive.
    """
    matches = _DRAWBUFFERS.findall(fragment_text)
    if matches:
        return [int(c) for c in matches[-1]]
    rt = _RENDERTARGETS.findall(fragment_text)
    if rt:
        return [int(tok) for tok in re.split(r"[,\s]+", rt[-1].strip()) if tok]
    return []


def _outputs_of(pack: ShaderPack, name: str, world: str) -> list[int]:
    """Union of every buffer a program's fragment stage might write.

    Static description only — BSL guards several DRAWBUFFERS variants behind
    ``#if`` branches, so the runtime set is a subset resolved by :func:`active_outputs`.
    """
    text = _fragment_text(pack, name, world)
    found: set[int] = set()
    for group in _DRAWBUFFERS.findall(text):
        found.update(int(c) for c in group)
    for group in _RENDERTARGETS.findall(text):
        found.update(int(tok) for tok in re.split(r"[,\s]+", group.strip()) if tok)
    return sorted(found)


def build_graph(pack: ShaderPack, world: str = "world0",
                 values: dict[str, object] | None = None) -> PipelineGraph:
    """Construct the ordered pass list and buffer pool for ``world``.

    ``values`` (the pack's *current* option values) resolves any ``#if``
    blocks `shaders.properties` uses around `program.*.enabled` lines (see
    ``pack.properties.parse_properties``) — omit it only for a pack known not
    to use them; a caller rebuilding the graph after an option change should
    always pass the current values so conditionally-enabled programs track
    the live option state, the same way OptiFine/Iris re-resolves them on
    every shader reload.
    """
    existing = set(pack.programs())
    props = pack.properties(values)

    # Buffer pool: scan every program's sources for format/clear declarations.
    buffers = BufferSet()
    for name in existing:
        try:
            prog = pack.read_program(name, world=world)
        except KeyError:
            continue
        for text in (prog.vertex, prog.fragment, prog.combined):
            if not text:
                continue
            for idx, fmt in _FORMAT.findall(text):
                buffers.formats.setdefault(int(idx), fmt)
            for idx, val in _CLEAR.findall(text):
                buffers.clear[int(idx)] = (val == "true")
    # colortex0/1 always exist even if formats are left default.
    for idx in (0, 1):
        buffers.formats.setdefault(idx, "RGBA16" if idx else "R11F_G11F_B10F")

    passes: list[Pass] = []

    def add(names: list[str], kind: str, fullscreen: bool) -> None:
        for name in names:
            passes.append(Pass(
                name=name, kind=kind, world=world,
                outputs=_outputs_of(pack, name, world),
                enable_expr=props.program_enabled.get(f"{world}/{name}"),
                is_fullscreen=fullscreen,
            ))

    add(_numbered("shadow", existing), "shadow", fullscreen=False)
    add(_numbered("shadowcomp", existing), "shadowcomp", fullscreen=True)
    add(_numbered("prepare", existing), "prepare", fullscreen=True)
    add([n for n in _OPAQUE_GBUFFERS if n in existing], "geometry", fullscreen=False)
    add(_numbered("deferred", existing), "deferred", fullscreen=True)
    add([n for n in _TRANSLUCENT_GBUFFERS if n in existing], "geometry", fullscreen=False)
    add(_numbered("composite", existing), "composite", fullscreen=True)
    add([n for n in _FINAL if n in existing], "final", fullscreen=True)

    # Every buffer a pass writes must exist in the pool; packs only declare a
    # format for buffers that need a non-default one, so fill the rest in.
    for p in passes:
        for idx in p.outputs:
            buffers.formats.setdefault(idx, "RGBA16")

    return PipelineGraph(world=world, passes=passes, buffers=buffers)
