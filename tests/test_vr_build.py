"""VR startup compiles the pack once: the screen-only overrides go in before
the first build, and the headset connecting reuses every translated program."""
import os

import pytest

PACK = os.path.join(os.path.dirname(__file__), "..", "Shaders", "BSL_v10.0.zip")


class FakeVR:
    """Just the Panda3d-VR contract the bridge uses."""

    def __init__(self, base):
        from panda3d.core import Camera, NodePath, PerspectiveLens

        self.eyes_ready = False
        self.connected = False
        self.eye_size = (160, 90)
        self.eye_cameras = [base.render.attach_new_node(Camera(f"eye{i}", PerspectiveLens()))
                            for i in range(2)]
        self.sources = {}

    def set_eye_source(self, i, color, depth):
        self.sources[i] = (color, depth)

    def clear_eye_sources(self):
        self.sources.clear()


@pytest.mark.skipif(not os.path.isfile(PACK), reason="Shaders/BSL_v10.0.zip not present")
def test_headset_connecting_translates_nothing():
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "window-type offscreen\nwin-size 320 180\naudio-library-name null\n")
    from direct.showbase.ShowBase import ShowBase
    import builtins
    import mcshader

    owned = getattr(builtins, "base", None) is None  # an earlier test may have one
    base = ShowBase() if owned else builtins.base
    try:
        vr = FakeVR(base)
        events = []
        app = mcshader.init(base, pack=PACK, profile="ULTRA", fly=False, sky=False,
                            vr=vr, progress=events.append)
        options = app.pipe.options
        assert {e.build for e in events} == {1}
        overridden = app.pipe.overridden_options
        assert overridden, "BSL ULTRA has screen-only effects on"
        assert all(not options.get(n) or options.get(n) in ("0", 0) for n in overridden)

        events.clear()
        vr.eyes_ready = vr.connected = True
        base.messenger.send("vr-eyes-ready", [vr])
        assert app.vr.active and set(vr.sources) == {0, 1}
        programs = [e for e in events if e.stage == "program"]
        assert programs and all(e.cached for e in programs)
        assert not any(e.message.startswith("Translating") for e in programs)

        # a settings save still writes the user's own values
        with app.vr.user_options() as opts:
            assert {n: opts.get(n) for n in overridden} == overridden
    finally:
        if owned:
            base.destroy()
