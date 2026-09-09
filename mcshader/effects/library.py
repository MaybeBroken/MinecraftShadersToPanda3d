"""The built-in effect library.

Four ready-to-use effects covering the techniques called out as the goal of the
project: glow, reflection, waving, and movement (animated UV scroll). Each is
authored in the Panda3D GLSL dialect and loaded from ``sources/`` at import
time. :func:`builtin_effects` returns fresh copies so callers can tweak
parameters without mutating the shared definitions.
"""

from __future__ import annotations

import copy
from importlib import resources

from .base import Effect, EffectParam

__all__ = ["builtin_effects", "BUILTIN_IDS"]

_SOURCES = resources.files(__package__).joinpath("sources")


def _src(name: str) -> str:
    return _SOURCES.joinpath(name).read_text(encoding="utf-8")


def _build() -> dict[str, Effect]:
    effects: dict[str, Effect] = {}

    effects["waving"] = Effect(
        id="waving",
        name="Waving",
        description="Wind-style vertex sway that grows toward the top of the mesh.",
        vertex=_src("waving.vert"),
        fragment=_src("waving.frag"),
        params={
            "u_wave_amplitude": EffectParam("u_wave_amplitude", "float", 0.08,
                "Maximum horizontal displacement in world units."),
            "u_wave_speed": EffectParam("u_wave_speed", "float", 2.0,
                "Oscillation speed."),
            "u_wave_scale": EffectParam("u_wave_scale", "float", 0.6,
                "Spatial frequency across the world; higher = tighter ripples."),
        },
        auto_inputs=["u_time"],
        mc_origin="gbuffers_terrain (waving grass/leaves)",
        tags=["movement", "waving", "vegetation"],
    )

    effects["glow"] = Effect(
        id="glow",
        name="Glow",
        description="Per-object emissive boost with an optional pulse.",
        vertex=_src("glow.vert"),
        fragment=_src("glow.frag"),
        params={
            "u_glow_color": EffectParam("u_glow_color", "vec3", (1.0, 0.85, 0.4),
                "Emissive colour added on top of the texture."),
            "u_glow_strength": EffectParam("u_glow_strength", "float", 0.8,
                "Emissive intensity."),
            "u_glow_pulse": EffectParam("u_glow_pulse", "float", 0.0,
                "Pulses per second; 0 keeps the glow steady."),
        },
        auto_inputs=["u_time"],
        mc_origin="gbuffers_entities_glowing / emissive maps",
        tags=["glow", "emissive"],
    )

    effects["reflection"] = Effect(
        id="reflection",
        name="Reflection",
        description="Fresnel-weighted cubemap reflection for water/metal looks.",
        vertex=_src("reflection.vert"),
        fragment=_src("reflection.frag"),
        params={
            "u_env_map": EffectParam("u_env_map", "samplerCube", None,
                "Environment cubemap; adapter binds a sky fallback if unset."),
            "u_reflectivity": EffectParam("u_reflectivity", "float", 0.15,
                "Base reflectance at head-on incidence."),
            "u_fresnel_power": EffectParam("u_fresnel_power", "float", 5.0,
                "Grazing-angle falloff sharpness."),
            "u_reflect_tint": EffectParam("u_reflect_tint", "vec3", (1.0, 1.0, 1.0),
                "Tint applied to the reflected colour."),
        },
        auto_inputs=["u_camera_pos"],
        mc_origin="composite reflections (SSR) / water",
        tags=["reflection", "water", "metal"],
    )

    effects["movement"] = Effect(
        id="movement",
        name="Movement (UV scroll)",
        description="Animated UV scroll with sine wobble: flowing water/lava.",
        vertex=_src("scroll.vert"),
        fragment=_src("scroll.frag"),
        params={
            "u_scroll_speed": EffectParam("u_scroll_speed", "vec2", (0.05, 0.0),
                "UV units scrolled per second."),
            "u_distort_amp": EffectParam("u_distort_amp", "float", 0.01,
                "Sine distortion amplitude in UV space."),
            "u_distort_freq": EffectParam("u_distort_freq", "float", 8.0,
                "Sine distortion frequency."),
        },
        auto_inputs=["u_time"],
        mc_origin="animated block textures / flowing water",
        tags=["movement", "scroll", "water", "lava"],
    )

    return effects


_REGISTRY = _build()
BUILTIN_IDS = tuple(_REGISTRY.keys())


def builtin_effects() -> dict[str, Effect]:
    """Return deep copies of all built-in effects, keyed by id."""
    return {eid: copy.deepcopy(effect) for eid, effect in _REGISTRY.items()}
