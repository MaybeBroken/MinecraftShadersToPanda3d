"""Map engine objects to Minecraft render types (and block ids).

The engine never names a shader program directly. Instead it tags an object with
a *render type* — ``"terrain"``, ``"water"``, ``"entity"``, ``"hand"``, … — and
the pipeline resolves that to the pack's matching ``gbuffers_*`` program, honouring
OptiFine's fallback chain (a pack that ships no ``gbuffers_terrain`` falls back to
``gbuffers_textured`` then ``gbuffers_basic``). Custom items just pick a render type
and, optionally, a *block id* — the small integer BSL reads from ``mc_Entity.x`` to
decide waving/emissive/material behaviour — so a custom glowing crystal can render
exactly as the pack renders emissive blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pack.loader import ShaderPack

__all__ = ["RENDER_TYPES", "program_for", "RenderTypeResolver"]

# render type -> preferred program, then fallbacks (OptiFine order).
RENDER_TYPES: dict[str, list[str]] = {
    "terrain":   ["gbuffers_terrain", "gbuffers_textured", "gbuffers_basic"],
    "water":     ["gbuffers_water", "gbuffers_terrain", "gbuffers_textured", "gbuffers_basic"],
    "translucent": ["gbuffers_water", "gbuffers_terrain", "gbuffers_textured", "gbuffers_basic"],
    "entity":    ["gbuffers_entities", "gbuffers_textured", "gbuffers_basic"],
    "glowing":   ["gbuffers_entities_glowing", "gbuffers_entities", "gbuffers_textured", "gbuffers_basic"],
    "block_entity": ["gbuffers_block", "gbuffers_terrain", "gbuffers_textured", "gbuffers_basic"],
    "hand":      ["gbuffers_hand", "gbuffers_textured", "gbuffers_basic"],
    "textured":  ["gbuffers_textured", "gbuffers_basic"],
    "sky":       ["gbuffers_skytextured", "gbuffers_textured", "gbuffers_basic"],
    "sky_basic": ["gbuffers_skybasic", "gbuffers_basic"],
    "clouds":    ["gbuffers_clouds", "gbuffers_textured", "gbuffers_basic"],
    "weather":   ["gbuffers_weather", "gbuffers_textured", "gbuffers_basic"],
    "beacon":    ["gbuffers_beaconbeam", "gbuffers_textured", "gbuffers_basic"],
    "basic":     ["gbuffers_basic"],
}


def program_for(render_type: str, available: set[str]) -> str | None:
    """Resolve a render type to the best available program, or None."""
    for candidate in RENDER_TYPES.get(render_type, []):
        if candidate in available:
            return candidate
    return None


@dataclass
class RenderTypeResolver:
    """Resolves render types + block ids for a specific pack."""

    pack: ShaderPack

    def __post_init__(self) -> None:
        self._available = set(self.pack.programs())
        self._blocks = self.pack.block_mapping()
        self._items = self.pack.item_mapping()
        self._entities = self.pack.entity_mapping()

    def program(self, render_type: str) -> str | None:
        """The gbuffers program this pack uses for ``render_type``."""
        return program_for(render_type, self._available)

    def types(self) -> list[str]:
        """Render types this pack can actually serve (have a program)."""
        return [rt for rt in RENDER_TYPES if self.program(rt) is not None]

    def block_id(self, mc_id: str, default: int = 0) -> int:
        """Block id (mc_Entity.x value) for a Minecraft block name."""
        return self._blocks.category_of(mc_id, default)

    def item_id(self, mc_id: str, default: int = 0) -> int:
        return self._items.category_of(mc_id, default)

    def entity_id(self, mc_id: str, default: int = 0) -> int:
        return self._entities.category_of(mc_id, default)
