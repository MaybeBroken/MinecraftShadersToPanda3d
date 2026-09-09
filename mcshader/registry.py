"""The shader-id registry: the lookup table behind ``apply(node, "glow")``.

A :class:`ShaderRegistry` maps string ids to :class:`~mcshader.effects.base.Effect`
objects. It starts pre-seeded with the built-in library so
``registry.get("glow")`` works out of the box, and custom or pack-extracted
effects can be registered under their own ids. Engine adapters resolve ids
through a registry, which is what makes "load this object with this shader id"
work.
"""

from __future__ import annotations

from .effects.base import Effect
from .effects.library import builtin_effects

__all__ = ["ShaderRegistry", "default_registry"]


class ShaderRegistry:
    def __init__(self, *, include_builtins: bool = True):
        self._effects: dict[str, Effect] = {}
        if include_builtins:
            self._effects.update(builtin_effects())

    def register(self, effect: Effect, *, replace: bool = False) -> Effect:
        """Register ``effect`` under its ``id``.

        Raises :class:`KeyError` if the id is already taken and ``replace`` is
        False, so a typo can't silently shadow a built-in.
        """
        if not effect.id:
            raise ValueError("effect.id must be a non-empty string")
        if effect.id in self._effects and not replace:
            raise KeyError(
                f"shader id {effect.id!r} already registered; "
                f"pass replace=True to overwrite"
            )
        self._effects[effect.id] = effect
        return effect

    def get(self, shader_id: str) -> Effect:
        try:
            return self._effects[shader_id]
        except KeyError:
            known = ", ".join(sorted(self._effects)) or "(none)"
            raise KeyError(
                f"unknown shader id {shader_id!r}; registered ids: {known}"
            ) from None

    def unregister(self, shader_id: str) -> None:
        self._effects.pop(shader_id, None)

    def ids(self) -> list[str]:
        return sorted(self._effects)

    def __contains__(self, shader_id: object) -> bool:
        return shader_id in self._effects

    def __len__(self) -> int:
        return len(self._effects)

    def __iter__(self):
        return iter(self._effects.values())


def default_registry() -> ShaderRegistry:
    """A fresh registry seeded with the built-in effects."""
    return ShaderRegistry(include_builtins=True)
