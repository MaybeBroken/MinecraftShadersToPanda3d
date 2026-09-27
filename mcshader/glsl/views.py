"""Per-camera uniforms for geometry programs, derived from Panda's built-ins.

A pack's gbuffers programs read the *camera* through uniforms --
``gbufferModelView``, ``gbufferProjection``, ``cameraPosition``,
``sunPosition``, ``shadowModelView``... The engine used to push those onto
``render``, which works for exactly one camera: every object in the shared
scene graph sees one value. With two cameras drawing the same scene (a VR
headset's eyes, or any multi-view setup) the second camera would render with
the first one's matrices.

Panda already hands every shader the matrices of whichever camera is drawing
it (``p3d_ViewMatrix``, ``p3d_ProjectionMatrix``...). This rewrites a
geometry program so the camera-dependent names are macros over those, plus a
few *world-anchored* uniforms (sun direction, shadow view) that really are
the same for every camera. The result is per-camera correct for free, with no
per-frame uniform traffic at all.

Conventions (all column-vector GLSL):

* Minecraft world space is Y-up and camera-relative; Panda's is Z-up.
  ``MCV_TO_PANDA`` maps (x, y, z)_mc -> (x, -z, y)_panda.
* ``p3d_ViewMatrix`` maps Panda world -> OpenGL view space (Y-up, -Z forward),
  which is exactly the view space every pack assumes.
"""

from __future__ import annotations

import re

__all__ = ["CAMERA_UNIFORMS", "WORLD_UNIFORMS", "apply_camera_macros"]

_PRELUDE_MACROS = """\
#define MCV_TO_PANDA mat4(1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, -1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0)
#define MCV_TO_MC mat4(1.0, 0.0, 0.0, 0.0,  0.0, 0.0, -1.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0)
#define MCV_VIEW (mat4(mat3(p3d_ViewMatrix)) * MCV_TO_PANDA)
#define MCV_VIEW_INV (MCV_TO_MC * mat4(mat3(p3d_ViewMatrixInverse)))
#define MCV_CAMERA vec3(p3d_ViewMatrixInverse[3].x, p3d_ViewMatrixInverse[3].z, -p3d_ViewMatrixInverse[3].y)
#define MCV_TRANSLATE(t) mat4(1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  (t).x, (t).y, (t).z, 1.0)
"""

#: Pack uniform -> expression.  Only names a program actually declares are
#: replaced, so a macro can never capture an unrelated identifier.
CAMERA_UNIFORMS = {
    "gbufferModelView": "MCV_VIEW",
    "gbufferModelViewInverse": "MCV_VIEW_INV",
    "gbufferProjection": "p3d_ProjectionMatrix",
    "gbufferProjectionInverse": "p3d_ProjectionMatrixInverse",
    "cameraPosition": "MCV_CAMERA",
    "sunPosition": "(mat3(MCV_VIEW) * mcSunDirection)",
    "shadowLightPosition": "(mat3(MCV_VIEW) * mcSunDirection)",
    "moonPosition": "(-(mat3(MCV_VIEW) * mcSunDirection))",
    "upPosition": "(mat3(MCV_VIEW) * vec3(0.0, 100.0, 0.0))",
    "shadowModelView": "(mcShadowView * MCV_TRANSLATE(MCV_CAMERA))",
    "shadowModelViewInverse": "(MCV_TRANSLATE(-MCV_CAMERA) * mcShadowViewInverse)",
}

#: Helper uniforms the macros need, identical for every camera.
WORLD_UNIFORMS = {
    "mcSunDirection": "vec3",        # sun direction * 100, Minecraft world axes
    "mcShadowView": "mat4",          # Minecraft world (absolute) -> shadow view
    "mcShadowViewInverse": "mat4",
}

_BUILTINS = {
    "p3d_ViewMatrix": "mat4",
    "p3d_ViewMatrixInverse": "mat4",
    "p3d_ProjectionMatrix": "mat4",
    "p3d_ProjectionMatrixInverse": "mat4",
}

_DECL = re.compile(
    r"^([ \t]*)uniform\s+((?:(?:highp|mediump|lowp)\s+)?)(\w+)\s+([^;]+);[ \t]*$", re.M)


def _declared(source: str, name: str) -> bool:
    return re.search(r"\buniform\s+(?:\w+\s+)*?\w+\s+[^;]*\b%s\b" % re.escape(name), source) is not None


def apply_camera_macros(source: str) -> str:
    """Rewrite one translated stage (see module docstring). No-op if the
    stage reads none of the camera uniforms."""
    removed: set[str] = set()

    def strip(match: re.Match) -> str:
        indent, precision, gtype, names = match.groups()
        kept = []
        for raw in names.split(","):
            bare = raw.strip().split("[")[0].strip()
            if bare in CAMERA_UNIFORMS:
                removed.add(bare)
            else:
                kept.append(raw.strip())
        if len(kept) == len(names.split(",")):
            return match.group(0)
        if not kept:
            return ""
        return f"{indent}uniform {precision}{gtype} {', '.join(kept)};"

    body = _DECL.sub(strip, source)
    if not removed:
        return source

    decls = [f"uniform {t} {n};" for n, t in {**_BUILTINS, **WORLD_UNIFORMS}.items()
             if not _declared(body, n)]
    macros = [f"#define {n} {CAMERA_UNIFORMS[n]}" for n in sorted(removed)]
    prelude = "\n".join(decls) + "\n" + _PRELUDE_MACROS + "\n".join(macros) + "\n"

    # After #version and any #extension lines that follow it.
    lines = body.split("\n")
    at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#version") or stripped.startswith("#extension"):
            at = i + 1
        elif stripped and not stripped.startswith("//"):
            break
    return "\n".join(lines[:at] + [prelude] + lines[at:])
