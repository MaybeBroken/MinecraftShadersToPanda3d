from mcshader.glsl.dialect import translate_stage


def test_vertex_qualifiers_and_builtins():
    src = (
        "#version 120\n"
        "attribute vec4 custom;\n"
        "varying vec2 uv;\n"
        "void main() { gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; }"
    )
    r = translate_stage(src, "vertex", "panda3d")
    assert r.source.startswith("#version 330")
    assert "in vec4 custom;" in r.source
    assert "out vec2 uv;" in r.source
    assert "p3d_ModelViewProjectionMatrix" in r.source
    assert "p3d_Vertex" in r.source
    assert "gl_ModelViewProjectionMatrix" not in r.source
    assert "uniform mat4 p3d_ModelViewProjectionMatrix;" in r.source
    assert "in vec4 p3d_Vertex;" in r.source


def test_fragment_varying_becomes_in_and_fragcolor_declared():
    src = "varying vec2 uv;\nvoid main() { gl_FragColor = texture2D(s, uv); }"
    r = translate_stage(src, "fragment", "panda3d")
    assert "in vec2 uv;" in r.source
    assert "texture(" in r.source and "texture2D" not in r.source
    assert "out vec4 mcFragColor;" in r.source
    assert "gl_FragColor" not in r.source


def test_ftransform_expanded():
    r = translate_stage("void main() { gl_Position = ftransform(); }", "vertex")
    assert "ftransform" not in r.source
    assert "p3d_ModelViewProjectionMatrix * p3d_Vertex" in r.source


def test_generic_target_uses_mc_names():
    r = translate_stage("void main(){ gl_Position = gl_Vertex; }", "vertex", "generic")
    assert "mc_Vertex" in r.source and "p3d_" not in r.source


def test_reports_mc_uniforms():
    src = "uniform float frameTimeCounter;\nuniform vec3 cameraPosition;\nvoid main(){}"
    r = translate_stage(src, "fragment")
    assert "frameTimeCounter" in r.mc_uniforms
    assert "cameraPosition" in r.mc_uniforms


def test_frag_data_indices_declared():
    src = "void main(){ gl_FragData[0]=vec4(1.0); gl_FragData[2]=vec4(0.0); }"
    r = translate_stage(src, "fragment")
    assert "layout(location = 0) out vec4 mcFragData0;" in r.source
    assert "layout(location = 2) out vec4 mcFragData2;" in r.source


def test_invalid_target_and_stage_raise():
    import pytest
    with pytest.raises(ValueError):
        translate_stage("", "vertex", "nope")
    with pytest.raises(ValueError):
        translate_stage("", "geometry")


# --- regressions from real-hardware compilation (BSL on Mesa core GLSL 330) ---

def test_texture_sampler_identifier_is_renamed():
    # `uniform sampler2D texture;` shadows the core builtin; must be renamed,
    # but texture(...) calls must not be.
    src = "uniform sampler2D texture;\nvoid main(){ gl_FragColor = texture2D(texture, vec2(0.0)); }"
    r = translate_stage(src, "fragment", "panda3d")
    assert "sampler2D p3d_Texture0;" in r.source
    assert "texture(p3d_Texture0, vec2(0.0))" in r.source
    assert "sampler2D texture;" not in r.source


def test_gl_multitexcoord0_promoted_to_vec4():
    r = translate_stage(
        "attribute vec4 x;\nvoid main(){ gl_Position = gl_TextureMatrix[0] * gl_MultiTexCoord0; }",
        "vertex", "panda3d")
    assert "vec4(p3d_MultiTexCoord0, 0.0, 1.0)" in r.source
    assert "mat4(1.0)" in r.source  # gl_TextureMatrix -> identity
    assert "in vec2 p3d_MultiTexCoord0;" in r.source


def test_shadow2d_compat_macro_emitted():
    r = translate_stage("void main(){ float s = shadow2D(shadowtex0, vec3(0.0)).x; }",
                        "fragment", "panda3d")
    assert "#define shadow2D(s, p) vec4(texture((s), (p)))" in r.source


def test_extension_directive_hoisted_to_top():
    src = "uniform sampler2D s;\n#extension GL_ARB_shader_texture_lod : enable\nvoid main(){}"
    r = translate_stage(src, "fragment", "panda3d")
    lines = [l for l in r.source.splitlines() if l.strip()]
    assert lines[0] == "#version 330"
    # the extension appears before any injected declaration / body
    assert lines[1].startswith("#extension")
