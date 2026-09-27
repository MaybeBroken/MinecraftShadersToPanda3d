"""Build-progress reporting, for loading screens.

Loading a pack is dominated by translating its programs (BSL ULTRA: ~23
programs, ~0.2 s each), so a pipeline build reports one step per program:

    def on_progress(p: mcshader.BuildProgress):
        print(f"{p.fraction:4.0%} {p.message}")

    app = mcshader.init(base, pack="BSL.zip", progress=on_progress)
    # or later: app.pipe.progress = on_progress   (every recompile reports)

The callback runs on the thread doing the build (Panda's main thread). An
event is immutable, so handing it to another thread, e.g. a Tk loading window
reading the latest one, needs no locking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = ["BuildProgress", "ProgressCallback"]


@dataclass(frozen=True)
class BuildProgress:
    """One step of a pipeline (re)build.

    ``stage`` is ``"pack"`` (reading the pack and its options; ``total`` is 0
    because the program count is not known yet), ``"program"`` (one program
    translated; ``program`` names it, ``ok`` says whether it compiled and
    ``cached`` whether an earlier build's translation was reused),
    ``"targets"`` (allocating render targets) or ``"done"``. ``build`` counts
    the builds this renderer has started, so a listener can tell a recompile
    from the first load.
    """

    stage: str
    message: str
    done: int = 0
    total: int = 0
    program: str | None = None
    ok: bool = True
    build: int = 1
    cached: bool = False

    @property
    def fraction(self) -> float:
        """Share of this build finished, 0..1."""
        if self.stage == "done":
            return 1.0
        return min(1.0, self.done / self.total) if self.total else 0.0


ProgressCallback = Callable[[BuildProgress], None]
