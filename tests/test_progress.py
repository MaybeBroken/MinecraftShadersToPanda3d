"""Build-progress events (see mcshader/progress.py)."""
import os

import pytest

from mcshader import BuildProgress

PACK = os.path.join(os.path.dirname(__file__), "..", "Shaders", "BSL_v10.0.zip")


def test_fraction():
    assert BuildProgress("pack", "Reading").fraction == 0.0
    assert BuildProgress("program", "x", done=3, total=12).fraction == 0.25
    assert BuildProgress("program", "x", done=13, total=12).fraction == 1.0
    assert BuildProgress("done", "ready").fraction == 1.0


@pytest.mark.skipif(not os.path.isfile(PACK), reason="Shaders/BSL_v10.0.zip not present")
def test_build_reports_every_program_then_done():
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "window-type offscreen\nwin-size 320 180\naudio-library-name null\n")
    from direct.showbase.ShowBase import ShowBase
    import mcshader

    import builtins
    owned = getattr(builtins, "base", None) is None  # an earlier test may have one
    base = ShowBase() if owned else builtins.base
    try:
        events: list[BuildProgress] = []
        app = mcshader.init(base, pack=PACK, fly=False, sky=False, progress=events.append)

        assert events[0].stage == "pack" and events[0].total == 0
        assert events[-1].stage == "done" and events[-1].fraction == 1.0
        finished = [e for e in events if e.stage == "program" and e.message.startswith(("Translated", "Skipped"))]
        total = events[-1].total
        # every program counted, plus one step per view's render targets
        assert len(finished) + len(app.pipe.views) == total
        fractions = [e.fraction for e in events]
        assert fractions == sorted(fractions)
        assert {e.build for e in events} == {1}

        events.clear()
        app.pipe.swap_pack(PACK)
        assert events[0].stage == "pack" and events[0].fraction == 0.0

        events.clear()
        app.pipe.recompile()
        assert {e.build for e in events} == {3}
        assert events[-1].stage == "done"
    finally:
        if owned:
            base.destroy()
