"""Data model for an extractable shader *effect*.

An :class:`Effect` is a self-contained, per-object shader technique lifted from
the Minecraft shader vocabulary (glow, reflection, waving, movement). It bundles
the vertex + fragment GLSL, the tunable parameters an author may override, and
the frame-driven "auto" inputs an engine adapter must feed every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["EffectParam", "Effect"]


@dataclass(frozen=True)
class EffectParam:
    """A tunable shader input with a sensible default.

    ``default`` of ``None`` marks a resource the caller (or adapter fallback)
    must provide — currently only used for the reflection cubemap.
    """
    name: str
    glsl_type: str
    default: Any
    description: str = ""


@dataclass
class Effect:
    """A named, parameterised shader effect addressable by :attr:`id`."""
    id: str
    name: str
    description: str
    vertex: str
    fragment: str
    params: dict[str, EffectParam] = field(default_factory=dict)
    #: Frame-driven uniforms the adapter feeds (mapped to providers by name).
    auto_inputs: list[str] = field(default_factory=list)
    #: GLSL dialect the sources are written in.
    dialect: str = "panda3d"
    #: Provenance: which Minecraft technique/program this derives from.
    mc_origin: str = ""
    tags: list[str] = field(default_factory=list)

    def defaults(self) -> dict[str, Any]:
        """Parameter name -> default value, skipping resources with no default."""
        return {p.name: p.default for p in self.params.values() if p.default is not None}
