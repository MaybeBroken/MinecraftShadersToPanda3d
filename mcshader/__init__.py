"""mcshader — Minecraft Shader Decompiler.

Run a Minecraft (OptiFine/Iris) shaderpack's whole deferred pipeline inside an
OpenGL engine, and re-shade your game by swapping packs. Tag scene objects with
Minecraft *render types* / block ids; drive features with an Iris-style options
system. Primary engine: Panda3D.

    from mcshader.engine import PipelineRenderer

    pipe = PipelineRenderer(base, "Shaders/", world="world0", profile="HIGH")
    pipe.set_render_type(terrain, "terrain")
    pipe.set_render_type(water, "water")
    pipe.set_block_id(custom_lamp, pipe.resolver.block_id("minecraft:sea_lantern"))
    pipe.apply_profile("ULTRA")     # or pipe.set_option("SHADOW_FILTER", True); pipe.recompile()
    pipe.swap_pack("OtherPack.zip") # re-shade everything

Inspect any pack's resolved pipeline/options without a GPU:

    from mcshader import load_pack, build_graph, ShaderOptions
    pack = load_pack("Shaders/")
    graph = build_graph(pack, "world0")
    opts = ShaderOptions.from_pack(pack.option_sources(), pack.properties())

A lighter per-object "simple effects" mode (glow/reflection/waving/movement) also
exists — see ``mcshader.effects`` and ``mcshader.engine.Panda3DAdapter``.
"""

from __future__ import annotations

# -- pipeline (primary) --------------------------------------------------
from .pack import ShaderPack
from .config import ShaderOptions, Option
from .pipeline import (
    PipelineGraph, build_graph, RenderTypeResolver, RENDER_TYPES, translate_pass,
)
from .decompiler import decompile_program, DecompiledProgram, load_pack

# -- simple per-object effects (secondary) ------------------------------
from .registry import ShaderRegistry, default_registry
from .effects import Effect, EffectParam, builtin_effects, BUILTIN_IDS

__version__ = "0.2.0"

__all__ = [
    # pipeline
    "ShaderPack", "load_pack",
    "ShaderOptions", "Option",
    "PipelineGraph", "build_graph", "RenderTypeResolver", "RENDER_TYPES",
    "translate_pass",
    "decompile_program", "DecompiledProgram",
    # simple effects
    "ShaderRegistry", "default_registry",
    "Effect", "EffectParam", "builtin_effects", "BUILTIN_IDS",
    "__version__",
]
