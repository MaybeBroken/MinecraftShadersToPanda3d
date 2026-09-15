"""WASD + arrow-key fly camera, for poking at a shaded scene by hand.

Moves ``base.cam`` directly (not ``base.camera``, its parent) — that's the
node the pipeline renderer actually reads for its view matrices (see
panda3d_pipeline.py's ``_dynamic_uniforms``), and the node
``set_camera(base.cam)`` renders from, so flying it around is the most
direct way to confirm the shading really responds to the camera and isn't
just a fixed on-screen gradient.
"""

from __future__ import annotations

from panda3d.core import ClockObject, Vec3

__all__ = ["attach_fly_camera"]

_KEYS = ("w", "a", "s", "d", "q", "e",
         "arrow_left", "arrow_right", "arrow_up", "arrow_down", "shift")


def attach_fly_camera(base, *, speed: float = 10.0, turn_speed: float = 90.0) -> None:
    """WASD to move, Q/E for down/up, arrow keys to look, shift to move faster.

    Disables Panda3D's default mouse-orbit camera control so the two don't
    fight over ``base.cam``'s transform.
    """
    base.disable_mouse()

    down: set[str] = set()
    for key in _KEYS:
        base.accept(key, down.add, [key])
        base.accept(f"{key}-up", down.discard, [key])

    clock = ClockObject.get_global_clock()

    def update(task):
        dt = clock.get_dt()
        boost = 3.0 if "shift" in down else 1.0

        move = Vec3(0, 0, 0)
        if "d" in down: move.x += 1
        if "a" in down: move.x -= 1
        if "w" in down: move.y += 1
        if "s" in down: move.y -= 1
        if "e" in down: move.z += 1
        if "q" in down: move.z -= 1
        if move.length_squared() > 0:
            move.normalize()
            base.cam.set_pos(base.cam, move * speed * boost * dt)

        h, p = base.cam.get_h(), base.cam.get_p()
        if "arrow_left" in down: h += turn_speed * dt
        if "arrow_right" in down: h -= turn_speed * dt
        if "arrow_up" in down: p = min(89.0, p + turn_speed * dt)
        if "arrow_down" in down: p = max(-89.0, p - turn_speed * dt)
        base.cam.set_hpr(h, p, 0)
        return task.cont

    base.taskMgr.add(update, "mcshader-flycam")
