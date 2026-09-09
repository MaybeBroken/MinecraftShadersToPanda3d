"""Integration tests against the bundled BSL shaderpack in Shaders/."""
import os
import re

import pytest

from mcshader import load_pack, decompile_program

PACK_DIR = os.path.join(os.path.dirname(__file__), "..", "Shaders")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(PACK_DIR), reason="Shaders/ pack not present"
)

LEGACY = {
    "attribute": r"\battribute\b",
    "varying": r"\bvarying\b",
    "texture2D": r"\btexture2D\b",
    "gl_FragColor": r"\bgl_FragColor\b",
    "gl_FragData": r"\bgl_FragData\b",
    "ftransform": r"\bftransform\b",
}


def test_pack_loads_programs():
    pack = load_pack(PACK_DIR)
    progs = pack.programs()
    assert "gbuffers_terrain" in progs
    assert len(progs) > 20


@pytest.mark.parametrize(
    "name",
    ["gbuffers_terrain", "gbuffers_water", "gbuffers_entities", "gbuffers_textured"],
)
def test_decompiled_stage_is_modern(name):
    pack = load_pack(PACK_DIR)
    prog = decompile_program(pack, name)
    assert prog.vertex and prog.fragment
    for tr in (prog.vertex, prog.fragment):
        assert tr.source.startswith("#version 330")
        assert "void main" in tr.source
        for label, pat in LEGACY.items():
            assert not re.search(pat, tr.source), f"{name}: leftover {label}"


def test_decompiled_reports_mc_uniforms():
    pack = load_pack(PACK_DIR)
    prog = decompile_program(pack, "gbuffers_terrain")
    assert "frameTimeCounter" in prog.mc_uniforms


def test_as_effect_registers():
    from mcshader import default_registry

    pack = load_pack(PACK_DIR)
    prog = decompile_program(pack, "gbuffers_terrain")
    reg = default_registry()
    reg.register(prog.as_effect("mc_terrain"))
    assert "mc_terrain" in reg
