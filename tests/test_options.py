from mcshader.pack.properties import parse_properties
from mcshader.config import ShaderOptions

SETTINGS = """
#define SHADOW
//#define TAA
#define AO_STRENGTH 1.00 //[0.25 0.50 1.00 2.00]
const int shadowMapResolution = 2048; //[512 1024 2048 4096]
#define INTERNAL_THING 3
"""

PROPS = parse_properties(
    "profile.LOW=SHADOW shadowMapResolution=1024 !TAA\n"
    "profile.HIGH=profile.LOW TAA AO_STRENGTH=2.00 shadowMapResolution=4096\n"
    "screen=SHADOW AO_STRENGTH shadowMapResolution TAA\n"
    "sliders=AO_STRENGTH shadowMapResolution\n"
)


def _opts():
    return ShaderOptions.from_pack({"lib/settings.glsl": SETTINGS}, PROPS)


def test_discovery_kinds_and_defaults():
    o = _opts()
    assert o.options["SHADOW"].kind == "toggle" and o.get("SHADOW") is True
    assert o.options["TAA"].kind == "toggle" and o.get("TAA") is False
    assert o.options["AO_STRENGTH"].kind == "enum"
    assert o.options["shadowMapResolution"].kind == "const"
    # bare #define not referenced in the menu is not an option
    assert "INTERNAL_THING" not in o.options


def test_profile_inheritance():
    o = _opts()
    o.apply_profile("HIGH", PROPS)
    assert o.get("SHADOW") is True        # from LOW
    assert o.get("TAA") is True            # HIGH re-enables
    assert o.get("AO_STRENGTH") == "2.00"
    assert o.get("shadowMapResolution") == "4096"  # HIGH overrides LOW's 1024


def test_set_validation():
    o = _opts()
    o.set("AO_STRENGTH", "0.50")
    assert o.get("AO_STRENGTH") == "0.50"
    import pytest
    with pytest.raises(ValueError):
        o.set("AO_STRENGTH", "9.9")


def test_rewrite_source_reflects_values():
    o = _opts()
    o.set("SHADOW", False)
    o.set("AO_STRENGTH", "2.00")
    o.set("shadowMapResolution", "512")
    out = o.rewrite_source(SETTINGS)
    assert "//#define SHADOW" in out
    assert "#define AO_STRENGTH 2.00" in out
    assert "const int shadowMapResolution = 512;" in out


def test_option_file_roundtrip():
    o = _opts()
    o.set("SHADOW", False)
    o.set("AO_STRENGTH", "0.50")
    dumped = o.dumps()
    o2 = _opts()
    o2.loads(dumped)
    assert o2.get("SHADOW") is False
    assert o2.get("AO_STRENGTH") == "0.50"


def test_menu_tree_annotates_values():
    o = _opts()
    tree = o.menu_tree(PROPS)
    opts_in_tree = {e["name"]: e for e in tree["elements"] if e["type"] == "option"}
    assert opts_in_tree["AO_STRENGTH"]["is_slider"] is True
    assert opts_in_tree["AO_STRENGTH"]["allowed"] == ["0.25", "0.50", "1.00", "2.00"]
