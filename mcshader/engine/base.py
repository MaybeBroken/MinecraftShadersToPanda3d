"""Engine-agnostic adapter interface.

An adapter binds the id-addressable effect system to a concrete OpenGL engine.
The contract is deliberately small — resolve an id to an effect, compile it,
attach it to an object, and keep its frame-driven uniforms fed — so new backends
(Panda3D, Harfang, raw GL, ...) only implement the engine-specific glue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..registry import ShaderRegistry, default_registry

__all__ = ["EngineAdapter"]


class EngineAdapter(ABC):
    def __init__(self, registry: ShaderRegistry | None = None):
        self.registry: ShaderRegistry = registry or default_registry()

    @abstractmethod
    def apply(self, obj: Any, shader_id: str, **overrides: Any) -> Any:
        """Apply the effect registered under ``shader_id`` to ``obj``.

        ``overrides`` replace individual effect parameter defaults. Returns an
        engine-specific handle (e.g. the compiled shader) where useful.
        """

    @abstractmethod
    def clear(self, obj: Any) -> None:
        """Remove any mcshader effect previously applied to ``obj``."""

    def available_ids(self) -> list[str]:
        return self.registry.ids()
