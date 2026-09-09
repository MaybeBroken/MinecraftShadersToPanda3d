"""Engine adapters that bind the effect system to concrete OpenGL backends.

Adapters are imported lazily via :func:`get_adapter` so that importing this
package never pulls in Panda3D or PyOpenGL unless you actually use them.
"""

from .base import EngineAdapter

__all__ = [
    "EngineAdapter", "get_adapter",
    "PipelineRenderer",                       # the deferred-pipeline runner (main feature)
    "Panda3DAdapter", "GenericGLAdapter",     # simple per-object effect modes
]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "PipelineRenderer":
        from .panda3d_pipeline import PipelineRenderer
        return PipelineRenderer
    if name == "Panda3DAdapter":
        from .panda3d_adapter import Panda3DAdapter
        return Panda3DAdapter
    if name == "GenericGLAdapter":
        from .generic_gl import GenericGLAdapter
        return GenericGLAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_adapter(engine: str, *args, **kwargs):
    """Instantiate an adapter by name: ``"panda3d"`` or ``"generic"``."""
    engine = engine.lower()
    if engine in ("panda3d", "panda"):
        from .panda3d_adapter import Panda3DAdapter
        return Panda3DAdapter(*args, **kwargs)
    if engine in ("generic", "gl", "opengl"):
        from .generic_gl import GenericGLAdapter
        return GenericGLAdapter(*args, **kwargs)
    raise ValueError(f"unknown engine {engine!r}; expected 'panda3d' or 'generic'")
