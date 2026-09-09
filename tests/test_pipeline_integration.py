"""Integration tests: build the full pipeline for the bundled BSL pack (no GL)."""
import os
import re

import pytest

from mcshader import load_pack, build_graph, ShaderOptions, RenderTypeResolver
from mcshader.pipeline import translate_pass

PACK = os.path.join(os.path.dirname(__file__), "..", "Shaders")
pytestmark = pytest.mark.skipif(not os.path.isdir(PACK), reason="Shaders/ not present")


def _opts(pack):
    props = pack.properties()
    o = ShaderOptions.from_pack(pack.option_sources(), props)
    return props, o


def test_graph_pass_order():
    pack = load_pack(PACK)
    g = build_graph(pack, "world0")
    kinds = [p.kind for p in g.passes]
    # shadow precedes geometry precedes deferred precedes composite precedes final
    assert kinds.index("shadow") < kinds.index("geometry")
    assert kinds.index("geometry") < kinds.index("deferred")
    assert kinds.index("deferred") < kinds.index("composite")
    assert kinds.index("composite") < kinds.index("final")
    assert g.passes[-1].name == "final"


def test_buffers_discovered():
    pack = load_pack(PACK)
    g = build_graph(pack, "world0")
    assert g.buffers.format_of(0) == "R11F_G11F_B10F"
    assert 9 in g.buffers.formats


def test_profiles_change_enabled_passes():
    pack = load_pack(PACK)
    props, o = _opts(pack)
    g = build_graph(pack, "world0")
    o.apply_profile("MINIMUM", props)
    minimal = {p.name for p in g.enabled_passes(o.values())}
    o.apply_profile("ULTRA", props)
    ultra = {p.name for p in g.enabled_passes(o.values())}
    assert "shadow" not in minimal and "shadow" in ultra
    assert "composite7" not in minimal and "composite7" in ultra


@pytest.mark.parametrize("name", [
    "gbuffers_terrain", "gbuffers_water", "gbuffers_entities",
    "deferred", "composite", "composite5", "final",
])
def test_translate_pass_outputs_are_collision_free(name):
    pack = load_pack(PACK)
    props, o = _opts(pack)
    o.apply_profile("HIGH", props)
    tp = translate_pass(pack, o, name)
    assert tp.vertex and tp.fragment
    assert tp.fragment.startswith("#version 330")
    locs = re.findall(r"layout\(location = (\d+)\)", tp.fragment)
    assert len(locs) == len(set(locs)), f"{name} has duplicate output locations"


def test_resolver_maps_render_types():
    pack = load_pack(PACK)
    r = RenderTypeResolver(pack)
    assert r.program("terrain") == "gbuffers_terrain"
    assert r.program("water") == "gbuffers_water"
    assert "terrain" in r.types()


def test_options_rewrite_changes_all_referencing_files():
    pack = load_pack(PACK)
    props, o = _opts(pack)
    o.set("SHADOW", False)
    files = o.rewrite_files(pack.files)
    # the settings file where SHADOW is declared should now have it commented
    decl_file = o.options["SHADOW"].source_file
    assert "//#define SHADOW" in files[decl_file]
