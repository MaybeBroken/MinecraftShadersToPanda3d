"""Extractable per-object shader effects and the built-in effect library."""

from .base import Effect, EffectParam
from .library import builtin_effects, BUILTIN_IDS

__all__ = ["Effect", "EffectParam", "builtin_effects", "BUILTIN_IDS"]
