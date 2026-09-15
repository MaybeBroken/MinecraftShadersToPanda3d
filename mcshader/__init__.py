"""mcshader — Minecraft Shader Decompiler.

Run a Minecraft (OptiFine/Iris) shaderpack's whole deferred pipeline inside an
OpenGL engine, and re-shade your game by swapping packs. Tag scene objects with
Minecraft *render types* / block ids; drive features with an Iris-style options
system. Primary engine: Panda3D.

The short way in — ``init()`` makes the Panda3D window, loads a pack, builds a
sky and starts the pipeline; pass your own ``ShowBase`` as ``init(base)`` to
shade a game you already have:

    import mcshader

    app = mcshader.init()
    app.load("models/environment", type="terrain", scale=0.25)
    app.load_actor("models/panda-model", {"walk": "models/panda-walk4"},
                   type="entity", block="minecraft:sea_lantern", loop="walk")
    app.profile = "ULTRA"           # or app.option("SHADOW_FILTER", True)
    app.swap_pack("OtherPack.zip")  # re-shade everything
    app.run()

The full runner underneath (``app.pipe``) is always reachable, and usable on
its own:

    from mcshader.engine import PipelineRenderer

    pipe = PipelineRenderer(base, "Shaders/", world="world0", profile="HIGH")
    pipe.set_render_type(terrain, "terrain")
    pipe.set_block_id(custom_lamp, pipe.resolver.block_id("minecraft:sea_lantern"))
    pipe.apply_profile("ULTRA")

Inspect any pack's resolved pipeline/options without a GPU:

    from mcshader import load_pack, build_graph, ShaderOptions
    pack = load_pack("Shaders/")
    graph = build_graph(pack, "world0")
    opts = ShaderOptions.from_pack(pack.option_sources(), pack.properties())

A lighter per-object "simple effects" mode (glow/reflection/waving/movement) also
exists — see ``mcshader.effects`` and ``mcshader.engine.Panda3DAdapter``.
"""

from __future__ import annotations

# -- the simple front door (Panda3D imported lazily, only when you call it) --
from .app import ShaderApp, init, find_pack

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
    # one-call app
    "ShaderApp", "init", "find_pack",
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
