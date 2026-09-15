"""The deferred-pipeline layer: render graph, render-type mapping, per-pass translation."""

from .graph import Pass, BufferSet, PipelineGraph, build_graph, active_outputs, eval_condition
from .rendertypes import RENDER_TYPES, program_for, RenderTypeResolver
from .translate import TranslatedPass, translate_pass

__all__ = [
    "init", "ShaderApp",
    "Pass", "BufferSet", "PipelineGraph", "build_graph", "active_outputs", "eval_condition",
    "RENDER_TYPES", "program_for", "RenderTypeResolver",
    "TranslatedPass", "translate_pass",
]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    """Also answer to ``pipeline.init(base)`` / ``pipeline.ShaderApp``.

    The one-call facade lives in :mod:`mcshader.app` (it's an engine front
    end, not part of this engine-independent layer), but "set up the
    pipeline" is the thing people come to this name looking for. Forwarded
    lazily so importing this package still never touches Panda3D.
    """
    if name in ("init", "ShaderApp", "find_pack"):
        from .. import app
        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
