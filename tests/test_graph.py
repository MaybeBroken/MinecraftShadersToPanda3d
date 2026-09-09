from mcshader.pipeline.graph import (
    Pass, BufferSet, eval_condition, active_outputs,
)


def test_eval_condition_boolean_logic():
    v = {"SHADOW": True, "RETRO_FILTER": False, "TAA": True, "AO_STRENGTH": "0.0"}
    assert eval_condition("SHADOW", v) is True
    assert eval_condition("SHADOW && !RETRO_FILTER", v) is True
    assert eval_condition("RETRO_FILTER || TAA", v) is True
    assert eval_condition("SHADOW && RETRO_FILTER", v) is False
    assert eval_condition("(SHADOW || RETRO_FILTER) && TAA", v) is True
    # numeric option: "0.0" is falsey, unknown is False
    assert eval_condition("AO_STRENGTH", v) is False
    assert eval_condition("UNKNOWN_OPT", v) is False


def test_active_outputs_last_directive_wins():
    src = "void main(){\n/*DRAWBUFFERS:01*/\n}\n// later\n/* DRAWBUFFERS:0195 */\n"
    assert active_outputs(src) == [0, 1, 9, 5]


def test_active_outputs_rendertargets_form():
    assert active_outputs("/* RENDERTARGETS: 0,4,7 */") == [0, 4, 7]


def test_pass_enabled_uses_expr():
    p = Pass("composite6", "composite", "world0", enable_expr="FXAA && !RETRO_FILTER")
    assert p.enabled({"FXAA": True, "RETRO_FILTER": False}) is True
    assert p.enabled({"FXAA": False, "RETRO_FILTER": False}) is False
    # no expr -> always on
    assert Pass("final", "final", "world0").enabled({}) is True


def test_buffer_set_defaults():
    bs = BufferSet(formats={0: "R11F_G11F_B10F"})
    assert bs.format_of(0) == "R11F_G11F_B10F"
    assert bs.format_of(7, default="RGBA16") == "RGBA16"
