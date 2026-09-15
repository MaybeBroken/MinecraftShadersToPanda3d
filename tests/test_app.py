"""The one-call facade (``mcshader.init`` / :class:`mcshader.app.ShaderApp`).

No GPU here: a stand-in PipelineRenderer records what the app asks of it, which
is exactly what these tests are about — that the facade drives the real runner
correctly, especially re-applying the per-object state (block ids, lightmaps) a
rebuild drops.
"""

from __future__ import annotations

import os

import pytest

import mcshader
from mcshader import app as app_mod


# -- pack discovery ------------------------------------------------------

def test_find_pack_prefers_explicit_path():
    assert app_mod.find_pack("some/pack.zip") == "some/pack.zip"


def test_find_pack_uses_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCSHADER_PACK", "/packs/BSL.zip")
    assert app_mod.find_pack() == "/packs/BSL.zip"


def test_find_pack_finds_a_pack_directory(monkeypatch, tmp_path):
    (tmp_path / "Shaders" / "shaders").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MCSHADER_PACK", raising=False)
    assert os.path.basename(app_mod.find_pack()) == "Shaders"


def test_find_pack_finds_a_zip_inside_a_pack_folder(monkeypatch, tmp_path):
    packs = tmp_path / "shaderpacks"
    packs.mkdir()
    (packs / "BSL.zip").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MCSHADER_PACK", raising=False)
    assert app_mod.find_pack().endswith("BSL.zip")


def test_find_pack_searches_parent_directories(monkeypatch, tmp_path):
    (tmp_path / "Shaders" / "shaders").mkdir(parents=True)
    sub = tmp_path / "examples"
    sub.mkdir()
    monkeypatch.chdir(sub)
    monkeypatch.delenv("MCSHADER_PACK", raising=False)
    assert app_mod.find_pack().endswith("Shaders")


def test_find_pack_error_names_the_ways_out(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MCSHADER_PACK", raising=False)
    with pytest.raises(FileNotFoundError, match="MCSHADER_PACK"):
        app_mod.find_pack()


def test_importing_mcshader_does_not_import_panda3d():
    # The facade lives in the top-level namespace, but the engine it drives
    # must still be imported lazily — the parsing/options/graph layers have
    # to stay usable (and testable) with no GL stack installed.
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import mcshader, sys; print(any(m.split('.')[0] in ('panda3d', 'direct')"
         " for m in sys.modules))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# -- stand-ins -----------------------------------------------------------

class FakeNode:
    def __init__(self, name="node"):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"


class FakeProps:
    profiles = {"ULTRA": "", "LOW": "", "MEDIUM": ""}


class FakeOptions:
    def __init__(self):
        self.options = {"SHADOW": object(), "AO": object()}
        self._values = {"SHADOW": True, "AO": False}

    def get(self, name):
        return self._values[name]

    def values(self):
        return dict(self._values)


class FakeResolver:
    def types(self):
        return ["terrain", "entity", "water"]

    def block_id(self, mc_id):
        return {"minecraft:sea_lantern": 15100}.get(mc_id, 1)


class FakePipe:
    """Records the calls the facade makes, and forgets per-object shader
    inputs on a rebuild — the way the real runner does."""

    def __init__(self, base, pack_path, *, world="world0", profile=None, **_):
        self.pack = type("P", (), {"name": "FakePack"})()
        self.props = FakeProps()
        self.options = FakeOptions()
        self.resolver = FakeResolver()
        self.profile_name = profile
        self._enabled = True
        self.types: dict[FakeNode, str] = {}
        self.blocks: dict[FakeNode, int] = {}
        self.lights: dict[FakeNode, tuple] = {}
        self.rebuilds = 0
        self.sky_built = False

    def build_sky(self):
        self.sky_built = True
        return FakeNode("sky")

    def set_render_type(self, np, render_type):
        self.types[np] = render_type

    def set_block_id(self, np, block_id):
        self.blocks[np] = block_id

    def set_lightmap(self, np, block_light, sky_light):
        self.lights[np] = (block_light, sky_light)

    def clear(self, np):
        self.types.pop(np, None)
        self.blocks.pop(np, None)
        self.lights.pop(np, None)

    def _rebuild(self):
        """What recompile/swap_pack really do: replay render types, drop the
        rest (block ids and lightmaps are plain shader inputs)."""
        self.rebuilds += 1
        self.blocks.clear()
        self.lights.clear()

    def recompile(self):
        self._rebuild()

    def swap_pack(self, path):
        self._rebuild()

    def apply_profile(self, name):
        self.profile_name = name
        self._rebuild()

    def set_option(self, name, value):
        self.options._values[name] = value

    def set_enabled(self, enabled):
        self._enabled = enabled

    def debug_textures(self):
        return {"colortex0": object(), "colortex1": object()}

    def describe(self):
        return "Pack: FakePack"


@pytest.fixture
def app(monkeypatch):
    import mcshader.engine as engine

    monkeypatch.setattr(engine, "PipelineRenderer", FakePipe, raising=False)
    return mcshader.init(base=object(), pack="fake/pack", profile="LOW",
                         sky=False, fly=False)


# -- the facade ----------------------------------------------------------

def test_init_takes_the_pack_as_the_first_argument(monkeypatch):
    import mcshader.engine as engine

    monkeypatch.setattr(engine, "PipelineRenderer", FakePipe, raising=False)
    # init("Shaders/") reads as the pack — an engine handle is never a string.
    made = mcshader.init("Shaders/", sky=False, fly=False)
    assert made.pack_path == "Shaders/"


def test_attach_tags_and_tracks(app):
    node = FakeNode("terrain")
    app.attach(node, type="terrain", block="minecraft:sea_lantern", light=(0.5, 1.0))
    assert app.pipe.types[node] == "terrain"
    assert app.pipe.blocks[node] == 15100        # resolved from the mc id
    assert app.pipe.lights[node] == (0.5, 1.0)


def test_block_id_may_be_numeric(app):
    node = FakeNode()
    app.attach(node, type="terrain", block=42)
    assert app.pipe.blocks[node] == 42


def test_profile_switch_restores_block_ids_and_lightmaps(app):
    node = FakeNode()
    app.attach(node, type="entity", block=7, light=(0.0, 0.25))
    app.profile = "ULTRA"
    assert app.pipe.rebuilds == 1
    assert app.profile == "ULTRA"
    # The real runner replays render types itself but drops these two; the
    # facade is what makes a profile switch not silently reset materials.
    assert app.pipe.blocks[node] == 7
    assert app.pipe.lights[node] == (0.0, 0.25)


def test_swap_pack_and_reload_restore_too(app):
    node = FakeNode()
    app.attach(node, type="entity", block=7)
    app.reload()
    assert app.pipe.blocks[node] == 7
    app.swap_pack()
    assert app.pipe.blocks[node] == 7


def test_remove_stops_tracking(app):
    node = FakeNode()
    app.attach(node, type="entity", block=7)
    app.remove(node)
    app.reload()
    assert node not in app.pipe.blocks
    assert node not in app.pipe.types


def test_option_get_set_and_unknown(app):
    assert app.option("SHADOW") is True
    app.option("SHADOW", False)
    assert app.option("SHADOW") is False
    assert app.pipe.rebuilds == 1              # setting applies by default
    app.option("AO", True, apply=False)
    assert app.pipe.rebuilds == 1              # ... unless you batch
    with pytest.raises(KeyError, match="NOPE"):
        app.option("NOPE", 1)


def test_options_and_render_types_are_readable(app):
    assert app.options == {"SHADOW": True, "AO": False}
    assert app.render_types == ["terrain", "entity", "water"]
    assert app.buffers == ["colortex0", "colortex1"]
    assert app.describe() == "Pack: FakePack"


def test_profiles_come_out_in_canonical_order(app):
    # FakeProps declares them out of order; the app sorts the known ones.
    assert app.profiles == ["LOW", "MEDIUM", "ULTRA"]


def test_enabled_round_trips(app):
    assert app.enabled is True
    app.enabled = False
    assert app.pipe._enabled is False


def test_sky_is_built_by_default(monkeypatch):
    import mcshader.engine as engine

    monkeypatch.setattr(engine, "PipelineRenderer", FakePipe, raising=False)
    made = mcshader.init(base=object(), pack="fake", fly=False)
    assert made.pipe.sky_built and made.sky is not None
