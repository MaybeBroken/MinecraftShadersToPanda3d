"""Camera-uniform macros for multi-view geometry programs (no GL needed)."""
import re

from mcshader.glsl.views import apply_camera_macros

SRC = """#version 330
#extension GL_ARB_shader_texture_lod : enable
uniform float frameTimeCounter;
uniform mat4 gbufferModelView, gbufferModelViewInverse;
uniform highp vec3 cameraPosition;
uniform mat4 shadowProjection;
void main() {
    vec3 p = (gbufferModelViewInverse * vec4(0.0)).xyz + cameraPosition;
}
"""


def test_declarations_become_macros():
    out = apply_camera_macros(SRC)
    assert not re.search(r"uniform[^;]*\bgbufferModelView\b", out)
    assert not re.search(r"uniform[^;]*\bcameraPosition\b", out)
    assert "#define gbufferModelView MCV_VIEW" in out
    assert "#define cameraPosition MCV_CAMERA" in out
    # untouched uniforms survive, including the shared shadow projection
    assert "uniform float frameTimeCounter;" in out
    assert "uniform mat4 shadowProjection;" in out
    # the built-ins the macros need are declared exactly once
    assert out.count("uniform mat4 p3d_ViewMatrixInverse;") == 1


def test_prelude_goes_after_version_and_extensions():
    lines = apply_camera_macros(SRC).split("\n")
    assert lines[0] == "#version 330"
    assert lines[1].startswith("#extension")
    assert lines[2].startswith("uniform mat4 p3d_ViewMatrix")


def test_only_declared_names_are_macroed():
    out = apply_camera_macros(SRC)
    assert "#define sunPosition" not in out  # never declared -> never captured


def test_existing_builtin_declarations_are_reused():
    src = SRC.replace("uniform float frameTimeCounter;", "uniform mat4 p3d_ProjectionMatrix;")
    out = apply_camera_macros(src)
    assert out.count("uniform mat4 p3d_ProjectionMatrix;") == 1


def test_noop_without_camera_uniforms():
    src = "#version 330\nuniform float frameTimeCounter;\nvoid main() {}\n"
    assert apply_camera_macros(src) == src
