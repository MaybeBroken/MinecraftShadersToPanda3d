"""The bundled demo scene — ``python -m mcshader demo [pack]``.

Doubles as the worked example for the one-call API in :mod:`mcshader.app`:
everything below is the whole program, there is no Panda3D boilerplate
hiding anywhere else. ``examples/pipeline_demo.py`` runs exactly this.

The scene is Panda3D's own bundled tutorial content — the "environment"
island (real ground/rock/tree/bamboo textures, ~90 pieces of actual
geometry) and the animated panda actor — so it needs no assets of its own.
Fly around and poke at it live; the on-screen help lists every control.

Needs a display and a GPU with enough colour attachments (see the README's
limitations); the parsing/graph/options layers are exercised headlessly by
the test suite and by ``python -m mcshader pipeline <pack>``.
"""

from __future__ import annotations

from .app import ShaderApp, init

__all__ = ["build", "run"]

# Bulk-tag rules for the "environment" island model, tried in order against
# each sub-part's name — the realistic way a scene with ~90 pieces of
# geometry gets tagged (by naming convention), not by hand-picking every
# object. Ground gets the real terrain program; rocks and tree trunks are
# opaque static geometry; bamboo/branches/reed "planes" all carry Panda's
# own TransparencyAttrib (they're alpha-cut leaf cards), so "entity" — the
# render type actual foliage-waving programs key off — fits them best.
_ENV_RULES = [
    (r"^Ground", "terrain"),
    (r"^Rock", "block_entity"),
    (r"^TreeTrunk", "block_entity"),
    (r"^(Branch|Bamboo|Plane|Cylinder)", "entity"),
]


def build(pack: str | None = None, *, profile: str = "LOW", **kwargs) -> ShaderApp:
    """Build the demo app (window, pack, scene, HUD) without running it."""
    app = init(pack=pack, profile=profile, speed=40.0, **kwargs)

    app.load("models/environment", scale=0.25, pos=(-8, 42, 0),
             tag=_ENV_RULES, default_type="terrain")

    # An animated character actor — proves the pipeline shades skinned,
    # moving geometry correctly, not just static props.
    panda = app.load_actor(
        "models/panda-model", {"walk": "models/panda-walk4"}, loop="walk",
        type="entity", block="minecraft:sea_lantern",
        scale=0.005, pos=(0, 20, 0), hpr=180)

    app.camera(pos=(0, -170, 55), look_at=(0, 20, 10), fov=100)
    app.debug_ui(focus=panda, extra="[t] pause/resume the panda's walk")

    # One control the generic HUD can't know about: pause the walk.
    state = {"walking": True}

    def toggle_walk():
        state["walking"] = not state["walking"]
        panda.loop("walk") if state["walking"] else panda.stop()

    app.key("t", toggle_walk)
    return app


def run(pack: str | None = None, *, profile: str = "LOW", **kwargs) -> None:
    """Build the demo and open the window (blocks until it's closed)."""
    app = build(pack, profile=profile, **kwargs)
    print(app.describe())
    app.run()
