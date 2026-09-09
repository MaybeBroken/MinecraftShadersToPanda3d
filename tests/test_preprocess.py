from mcshader.glsl.preprocess import resolve_includes, select_stage


def test_resolve_includes_from_files_map():
    files = {"lib/util.glsl": "float two() { return 2.0; }"}
    src = '#include "lib/util.glsl"\nvoid main() {}'
    out = resolve_includes(src, files=files)
    assert "float two()" in out
    assert "#include" not in out


def test_resolve_includes_leading_slash_matches_map():
    files = {"program/x.glsl": "int X;"}
    out = resolve_includes('#include "/program/x.glsl"', files=files)
    assert "int X;" in out


def test_resolve_includes_keeps_unresolvable():
    out = resolve_includes('#include "nope.glsl"', files={})
    assert '#include "nope.glsl"' in out


def test_resolve_includes_breaks_cycles():
    files = {"a": '#include "b"', "b": '#include "a"'}
    out = resolve_includes('#include "a"', files=files)
    assert "cyclic" in out


def test_select_stage_keeps_shared_drops_other():
    src = (
        "uniform float shared_u;\n"
        "#ifdef VSH\n"
        "in vec4 vpos;\n"
        "#endif\n"
        "#ifdef FSH\n"
        "out vec4 col;\n"
        "#endif\n"
    )
    vtx = select_stage(src, "vertex")
    assert "shared_u" in vtx and "in vec4 vpos;" in vtx
    assert "out vec4 col;" not in vtx and "#ifdef" not in vtx

    frag = select_stage(src, "fragment")
    assert "shared_u" in frag and "out vec4 col;" in frag
    assert "in vec4 vpos;" not in frag


def test_select_stage_passes_through_other_conditionals():
    src = "#ifdef VSH\n#ifdef OVERWORLD\nint a;\n#endif\n#endif\n"
    out = select_stage(src, "vertex")
    assert "#ifdef OVERWORLD" in out and "int a;" in out
    # the VSH guard itself is stripped, the OVERWORLD one preserved
    assert "VSH" not in out


def test_select_stage_handles_else_on_stage_guard():
    src = "#ifdef VSH\nint v;\n#else\nint f;\n#endif\n"
    assert "int v;" in select_stage(src, "vertex")
    assert "int f;" not in select_stage(src, "vertex")
    assert "int f;" in select_stage(src, "fragment")
    assert "int v;" not in select_stage(src, "fragment")


def test_resolve_includes_directory_relative():
    # a sibling include with no leading slash resolves against the including
    # file's directory (BSL: lib/color/dimensionColor.glsl includes "lightColor.glsl")
    files = {
        "lib/color/dimensionColor.glsl": '#include "lightColor.glsl"\nint dim;',
        "lib/color/lightColor.glsl": "int light;",
    }
    out = resolve_includes('#include "/lib/color/dimensionColor.glsl"', files=files)
    assert "int light;" in out and "int dim;" in out
    assert "#include" not in out
