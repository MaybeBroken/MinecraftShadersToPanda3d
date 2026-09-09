"""Catalog of the uniforms Minecraft (OptiFine/Iris) shaders expect.

A shaderpack assumes the game feeds it a large set of well-known uniforms
(time, camera, matrices, the sun/moon position, and so on). When we lift a
shader out of the pack and into another engine, *something* has to supply those
values. This module names them, records their GLSL type, and flags the ones an
engine adapter can compute on its own (``auto``) versus the ones a caller must
provide (e.g. a shadow map that only exists in the deferred pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["McUniform", "CATALOG", "auto_uniforms", "lookup"]


@dataclass(frozen=True)
class McUniform:
    name: str
    glsl_type: str
    description: str
    #: True when an engine adapter can compute/feed the value each frame.
    auto: bool = False
    #: Hint for adapters on where the value comes from.
    provider: str | None = None


CATALOG: dict[str, McUniform] = {
    u.name: u
    for u in [
        # --- time ---------------------------------------------------------
        McUniform("frameTimeCounter", "float",
                  "Seconds since shader load, wraps ~3600s.",
                  auto=True, provider="time"),
        McUniform("frameTime", "float", "Duration of the last frame (s).",
                  auto=True, provider="delta_time"),
        McUniform("frameCounter", "int", "Frames since load.",
                  auto=True, provider="frame_number"),
        # --- camera / view ------------------------------------------------
        McUniform("cameraPosition", "vec3", "Camera position in world space.",
                  auto=True, provider="camera_position"),
        McUniform("previousCameraPosition", "vec3",
                  "Camera position last frame.", auto=True,
                  provider="prev_camera_position"),
        McUniform("viewWidth", "float", "Viewport width in pixels.",
                  auto=True, provider="viewport_width"),
        McUniform("viewHeight", "float", "Viewport height in pixels.",
                  auto=True, provider="viewport_height"),
        McUniform("aspectRatio", "float", "viewWidth / viewHeight.",
                  auto=True, provider="aspect_ratio"),
        McUniform("near", "float", "Near clip plane.", auto=True,
                  provider="near"),
        McUniform("far", "float", "Far clip plane.", auto=True,
                  provider="far"),
        # --- matrices -----------------------------------------------------
        McUniform("gbufferModelView", "mat4", "World->view matrix.",
                  auto=True, provider="model_view"),
        McUniform("gbufferModelViewInverse", "mat4", "View->world matrix.",
                  auto=True, provider="model_view_inverse"),
        McUniform("gbufferProjection", "mat4", "View->clip matrix.",
                  auto=True, provider="projection"),
        McUniform("gbufferProjectionInverse", "mat4", "Clip->view matrix.",
                  auto=True, provider="projection_inverse"),
        McUniform("gbufferPreviousModelView", "mat4", "Last frame's world->view.",
                  auto=True, provider="prev_model_view"),
        McUniform("gbufferPreviousProjection", "mat4", "Last frame's view->clip.",
                  auto=True, provider="prev_projection"),
        # --- shadow matrices ---------------------------------------------
        McUniform("shadowModelView", "mat4", "World->shadow-view matrix.",
                  auto=True, provider="shadow_model_view"),
        McUniform("shadowModelViewInverse", "mat4", "Shadow-view->world matrix.",
                  auto=True, provider="shadow_model_view_inverse"),
        McUniform("shadowProjection", "mat4", "Shadow-view->clip matrix.",
                  auto=True, provider="shadow_projection"),
        McUniform("shadowProjectionInverse", "mat4", "Shadow-clip->view matrix.",
                  auto=True, provider="shadow_projection_inverse"),
        # --- sky / lighting ----------------------------------------------
        McUniform("sunPosition", "vec3", "Sun position in view space.",
                  auto=True, provider="sun_position"),
        McUniform("moonPosition", "vec3", "Moon position in view space.",
                  auto=True, provider="moon_position"),
        McUniform("shadowLightPosition", "vec3",
                  "Direction to the dominant light (view space).", auto=True,
                  provider="shadow_light_position"),
        McUniform("upPosition", "vec3", "World up in view space.", auto=True,
                  provider="up_position"),
        McUniform("sunAngle", "float", "0..1 fraction of the day.", auto=True,
                  provider="sun_angle"),
        McUniform("rainStrength", "float", "0=clear, 1=raining.", auto=True,
                  provider="rain_strength"),
        McUniform("wetness", "float", "Smoothed rain strength.", auto=True,
                  provider="wetness"),
        McUniform("skyColor", "vec3", "Current sky tint.", auto=True,
                  provider="sky_color"),
        McUniform("fogColor", "vec3", "Current fog tint.", auto=True,
                  provider="fog_color"),
        # --- samplers (pipeline-bound; caller must supply) ---------------
        McUniform("depthtex0", "sampler2D", "Scene depth.",
                  provider="texture"),
        McUniform("shadowtex0", "sampler2D", "Shadow map depth.",
                  provider="texture"),
        McUniform("noisetex", "sampler2D", "Tiling noise texture.",
                  provider="texture"),
        McUniform("depthtex1", "sampler2D", "Scene depth (no translucents).",
                  provider="texture"),
        McUniform("depthtex2", "sampler2D", "Scene depth (no hand).",
                  provider="texture"),
        McUniform("shadowtex1", "sampler2D", "Shadow depth (opaque only).",
                  provider="texture"),
        McUniform("shadowcolor0", "sampler2D", "Shadow color attachment 0.",
                  provider="texture"),
        McUniform("shadowcolor1", "sampler2D", "Shadow color attachment 1.",
                  provider="texture"),
    ] + [
        # The colortex pool and its OptiFine aliases (gcolor..gaux4).
        McUniform(name, "sampler2D", f"Color attachment {name}.", provider="texture")
        for name in (
            [f"colortex{i}" for i in range(16)]
            + ["gcolor", "gdepth", "gnormal", "composite", "gaux1", "gaux2",
               "gaux3", "gaux4"]
        )
    ]
}


def lookup(name: str) -> McUniform | None:
    """Return the catalog entry for ``name`` if it is a known MC uniform."""
    return CATALOG.get(name)


def auto_uniforms() -> dict[str, McUniform]:
    """Uniforms an engine adapter can feed without caller involvement."""
    return {name: u for name, u in CATALOG.items() if u.auto}
