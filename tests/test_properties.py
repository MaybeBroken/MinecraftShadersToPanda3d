from mcshader.pack.properties import (
    read_raw, parse_properties, parse_block_mapping,
)


def test_read_raw_handles_continuations_and_comments():
    text = "# comment\nblock.1= a \\\n b \\\n c\nkey=value\n"
    raw = read_raw(text)
    assert raw["block.1"].split() == ["a", "b", "c"]
    assert raw["key"] == "value"
    assert "# comment" not in raw


def test_parse_properties_profiles_and_screens():
    text = (
        "profile.LOW=SHADOW shadowDistance=128.0\n"
        "profile.HIGH=profile.LOW AO !TAA\n"
        "screen=ABOUT [LIGHTING] <empty>\n"
        "screen.LIGHTING=SHADOW AO\n"
        "sliders=shadowDistance AO_STRENGTH\n"
        "program.world0/shadow.enabled=SHADOW\n"
        "dynamicHandLight=true\n"
    )
    p = parse_properties(text)
    assert p.profiles["HIGH"] == "profile.LOW AO !TAA"
    assert p.screens[""] == ["ABOUT", "[LIGHTING]", "<empty>"]
    assert p.screens["LIGHTING"] == ["SHADOW", "AO"]
    assert "shadowDistance" in p.sliders
    assert p.program_enabled["world0/shadow"] == "SHADOW"
    assert p.settings["dynamicHandLight"] == "true"


def test_screen_tree_nests_and_marks_types():
    p = parse_properties("screen=ABOUT [SUB] <empty>\nscreen.SUB=X\n")
    tree = p.screen_tree()
    types = [e["type"] for e in tree["elements"]]
    assert types == ["option", "screen", "spacer"]
    assert tree["elements"][1]["screen"]["elements"][0]["name"] == "X"


def test_screen_tree_breaks_cycles():
    p = parse_properties("screen=[A]\nscreen.A=[A]\n")
    # must terminate
    p.screen_tree()


def test_block_mapping_reverse_lookup():
    text = "block.10500=minecraft:oak_leaves minecraft:birch_leaves\nblock.100=minecraft:grass\n"
    bm = parse_block_mapping(text, "block")
    assert bm.category_of("minecraft:oak_leaves") == 10500
    assert bm.category_of("oak_leaves") == 10500       # prefix optional
    assert bm.category_of("minecraft:grass") == 100
    assert bm.category_of("minecraft:unknown", default=-1) == -1


def test_block_mapping_accumulates_across_lines():
    text = "block.5=minecraft:a\nblock.5=minecraft:b\n"
    bm = parse_block_mapping(text, "block")
    assert set(bm.ids_by_category[5]) == {"minecraft:a", "minecraft:b"}
