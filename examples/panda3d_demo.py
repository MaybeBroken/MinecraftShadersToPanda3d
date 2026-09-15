"""Apply Minecraft-derived effects to objects by shader id, in Panda3D.

    pip install panda3d
    python examples/panda3d_demo.py

Spins up four models and gives each a different effect purely by id:
``waving``, ``glow``, ``reflection``, and ``movement``. This is the core use
case: "load an object with this shader id" so custom items pick up the
effect. Fly around and re-tag cubes live with different ids — including
clearing an effect entirely and reapplying one — to actually verify
``apply()``/``clear()`` work rather than trusting a static screenshot; see
the on-screen help (also printed to the console) for every control.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  # noqa: E402 (run from a source checkout without installing)


from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import AmbientLight, DirectionalLight, TextNode, Vec4

from mcshader.engine import Panda3DAdapter
from mcshader.ui import attach_fly_camera

HELP = """mcshader per-object effects demo
WASD/QE fly, arrows look, shift boost
[1]-[4]        select a cube (highlighted)
[[ ]]          cycle the selected cube's shader id
[c]            cycle a glow-colour preset (only affects "glow")
[x]            clear the selected cube's shader entirely
[v]            reapply its current id (proves clear()+apply() round-trip)
[k] screenshot   [h] hide/show this help"""

_GLOW_COLORS = [(0.2, 0.9, 1.0), (1.0, 0.3, 0.15), (0.4, 1.0, 0.3), (0.9, 0.2, 0.9)]


class Demo(ShowBase):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(0.1, 0.12, 0.16)
        self.cam.setPos(0, -14, 3)
        self.cam.lookAt(0, 0, 0)
        attach_fly_camera(self)

        key = self.render.attachNewNode(DirectionalLight("key"))
        key.setHpr(-45, -60, 0)
        self.render.setLight(key)
        amb = self.render.attachNewNode(AmbientLight("amb"))
        amb.node().setColor(Vec4(0.3, 0.3, 0.35, 1))
        self.render.setLight(amb)

        # One adapter drives every effect and feeds time/camera uniforms.
        self.fx = Panda3DAdapter(base=self)
        self._ids = self.fx.available_ids()

        layout = [
            ("waving", (-6, 0, 0), {}),
            ("glow", (-2, 0, 0), {"u_glow_color": _GLOW_COLORS[0], "u_glow_pulse": 1.5}),
            ("reflection", (2, 0, 0), {"u_reflectivity": 0.4}),
            ("movement", (6, 0, 0), {"u_scroll_speed": (0.15, 0.05)}),
        ]
        self._cubes = []        # NodePath per slot
        self._current_id = []   # this slot's last-applied shader id (kept across a clear)
        self._active = []       # whether that id is currently attached (False after [x])
        self._glow_color_idx = []
        for shader_id, pos, overrides in layout:
            model = self.loader.loadModel("models/misc/rgbCube")
            model.reparentTo(self.render)
            model.setPos(*pos)
            self.fx.apply(model, shader_id, **overrides)
            self._cubes.append(model)
            self._current_id.append(shader_id)
            self._active.append(True)
            self._glow_color_idx.append(0)
            print(f"applied {shader_id!r} to a cube at {pos}")

        print("Available shader ids:", self._ids)

        self._selected = 1  # start on the glow cube — colour cycling is visible immediately
        self._select_marker = self._build_select_marker()
        self._update_marker()

        print("\n" + HELP)
        self._help_text = OnscreenText(
            text=HELP, pos=(-1.3, 0.95), scale=0.045, align=TextNode.ALeft,
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.6), mayChange=False)
        self._status_text = OnscreenText(
            text="", pos=(-1.3, -0.9), scale=0.05, align=TextNode.ALeft,
            fg=(1, 1, 0.5, 1), shadow=(0, 0, 0, 0.6), mayChange=True)
        self._update_status()

        for i in range(4):
            self.accept(str(i + 1), self._select, [i])
        self.accept("[", self._cycle_id, [-1])
        self.accept("]", self._cycle_id, [1])
        self.accept("c", self._cycle_glow_color)
        self.accept("x", self._clear_selected)
        self.accept("v", self._reapply_selected)
        self.accept("k", self._screenshot)
        self.accept("h", self._toggle_help)

    # -- selection ------------------------------------------------------------
    def _build_select_marker(self):
        # A flat, colored plate under the selected cube — cheap to build from
        # the same primitive, no extra assets needed.
        marker = self.loader.loadModel("models/misc/rgbCube")
        marker.reparentTo(self.render)
        marker.set_scale(1.3, 1.3, 0.05)
        marker.set_color(1, 1, 0, 1)
        return marker

    def _update_marker(self):
        pos = self._cubes[self._selected].get_pos()
        self._select_marker.set_pos(pos.x, pos.y, pos.z - 1.1)

    def _select(self, index):
        self._selected = index
        self._update_marker()
        self._update_status()

    # -- live pokes -------------------------------------------------------------
    def _apply(self, i, shader_id):
        overrides = {"u_glow_color": _GLOW_COLORS[self._glow_color_idx[i]]} if shader_id == "glow" else {}
        self.fx.apply(self._cubes[i], shader_id, **overrides)
        self._current_id[i] = shader_id
        self._active[i] = True

    def _cycle_id(self, direction):
        i = self._selected
        idx = (self._ids.index(self._current_id[i]) + direction) % len(self._ids)
        new_id = self._ids[idx]
        self._apply(i, new_id)
        self._update_status()
        print(f"cube {i + 1} shader id -> {new_id!r}")

    def _cycle_glow_color(self):
        i = self._selected
        if self._current_id[i] != "glow" or not self._active[i]:
            print(f"cube {i + 1} isn't using 'glow' — [c] only affects glow's colour")
            return
        self._glow_color_idx[i] = (self._glow_color_idx[i] + 1) % len(_GLOW_COLORS)
        self._apply(i, "glow")
        self._update_status()
        print(f"cube {i + 1} glow colour -> {_GLOW_COLORS[self._glow_color_idx[i]]}")

    def _clear_selected(self):
        i = self._selected
        self.fx.clear(self._cubes[i])
        self._active[i] = False
        self._update_status()
        print(f"cube {i + 1} shader cleared (was {self._current_id[i]!r} — [v] restores it)")

    def _reapply_selected(self):
        i = self._selected
        self._apply(i, self._current_id[i])
        self._update_status()
        print(f"cube {i + 1} shader reapplied -> {self._current_id[i]!r}")

    def _screenshot(self):
        path = self.screenshot(namePrefix="mcshader-fx")
        print("saved", path)

    def _toggle_help(self):
        if self._help_text.is_hidden():
            self._help_text.show()
        else:
            self._help_text.hide()

    def _update_status(self):
        labels = [f"{sid}{'' if active else ' (cleared)'}"
                  for sid, active in zip(self._current_id, self._active)]
        self._status_text.setText(f"selected=cube {self._selected + 1}  cubes=[{', '.join(labels)}]")


if __name__ == "__main__":
    Demo().run()
