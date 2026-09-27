"""Saved option files (see mcshader/config/prefs.py)."""
import os

import pytest

from mcshader.config.prefs import format_prefs, parse_prefs

PACK = os.path.join(os.path.dirname(__file__), "..", "Shaders", "BSL_v10.0.zip")


def test_round_trip():
    text = format_prefs("BLOOM=true\nSHADOW_QUALITY=2\n", pack="X.zip", profile="HIGH")
    prefs = parse_prefs(text)
    assert (prefs.pack, prefs.profile) == ("X.zip", "HIGH")
    assert "SHADOW_QUALITY=2" in prefs.text
    assert prefs.matches("X.zip") and not prefs.matches("Y.zip")


def test_no_header_matches_any_pack():
    prefs = parse_prefs("BLOOM=false\n")
    assert prefs.pack == "" and prefs.profile == ""
    assert prefs.matches("anything.zip")


@pytest.mark.skipif(not os.path.isfile(PACK), reason="Shaders/BSL_v10.0.zip not present")
def test_prefs_apply_before_the_one_build(tmp_path):
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "window-type offscreen\nwin-size 320 180\naudio-library-name null\n")
    from direct.showbase.ShowBase import ShowBase
    import builtins
    import mcshader

    owned = getattr(builtins, "base", None) is None  # an earlier test may have one
    base = ShowBase() if owned else builtins.base
    try:
        from mcshader.pack.loader import ShaderPack
        from mcshader.config import ShaderOptions

        pack = ShaderPack.from_path(PACK)
        opts = ShaderOptions.from_pack(pack.option_sources(), pack.properties())
        name = next(n for n, o in opts.options.items() if o.kind == "toggle")
        flipped = not opts.get(name)
        path = tmp_path / "saved.prefs"
        path.write_text(format_prefs(f"{name}={str(flipped).lower()}\n",
                                     pack=pack.name, profile="LOW"))

        builds = set()
        app = mcshader.init(base, pack=PACK, profile="ULTRA", fly=False, sky=False,
                            prefs=path, progress=lambda e: builds.add(e.build))
        assert builds == {1}
        assert app.pipe.profile_name == "LOW"
        assert app.pipe.options.get(name) == flipped
        assert app.pipe.prefs_status.startswith("loaded")

        # saving writes to the same file by default, and reads back the same
        app.pipe.options.set(name, not flipped)
        assert app.pipe.save_prefs() == path
        assert f"{name}={str(not flipped).lower()}" in path.read_text()

        other = tmp_path / "other.prefs"
        other.write_text(format_prefs(f"{name}=true\n", pack="Other.zip", profile="LOW"))
        app.pipe.load_pack(PACK, profile="ULTRA", prefs=other)
        assert app.pipe.prefs_status.startswith("ignoring")
        assert app.pipe.profile_name == "ULTRA"

        app.pipe.load_pack(PACK, prefs=tmp_path / "missing.prefs")
        assert app.pipe.prefs_status.startswith("no shader preferences")
    finally:
        if owned:
            base.destroy()
