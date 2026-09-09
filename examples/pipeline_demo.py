"""Run a Minecraft shaderpack's deferred pipeline over a Panda3D scene.

    pip install panda3d
    python examples/pipeline_demo.py [path-to-pack]

Tags each object with a Minecraft render type and lets the pack shade the
whole frame (shadow -> gbuffers -> deferred -> composite -> final). Fly
around and poke at it live — profiles, options, render types, block ids, a
raw-buffer viewer, a pack reload — instead of trusting a single screenshot;
see the on-screen help (also printed to the console) for every control.
Needs a display + a GPU with enough color attachments (see the README's
limitations); the parsing/graph/options layers are exercised headlessly by
the tests and by examples/pipeline_describe.py.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  # noqa: E402 (run from a source checkout without installing)

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, CardMaker

from mcshader.engine import PipelineRenderer
from _flycam import attach_fly_camera
from _checker import make_checker_texture

HELP = """mcshader pipeline demo
WASD/QE fly, arrows look, shift boost   [g] reset camera to the establishing shot
[1][2][3] profile MINIMUM/HIGH/ULTRA
[z] toggle SHADOW option
[r] cycle crystal's render type
[b] cycle crystal's block id
[[ ]] step raw colortex buffers, [0] back to final
[p] swap_pack() reload (tags survive)
[k] screenshot   [h] hide/show this help"""

# A few blocks chosen to visibly exercise different pack behaviour (waving
# foliage, an emissive light source, a plain block) via the same block-id API.
_BLOCKS = ["minecraft:sea_lantern", "minecraft:oak_leaves", "minecraft:stone"]

_HOME_POS = (18, -26, 12)
_HOME_LOOK = (0, 0, 1)

# name, pos, scale, render_type — four pillars of different heights spread
# around the ground so there's actually something to judge shading, shadows,
# and parallax against; a single flat plane and two small cubes (the old
# scene) don't give enough to look at to tell whether anything is working.
_PILLARS = [
    ("pillar_nw", (-7, 6, 0.5), (1.5, 1.5, 3), "terrain"),
    ("pillar_ne", (7, 6, 1.5), (1.5, 1.5, 5), "block_entity"),
    ("pillar_sw", (-7, -2, 0), (1.5, 1.5, 2), "entity"),
    ("pillar_se", (7, -2, 1), (1.5, 1.5, 4), "textured"),
]


class Demo(ShowBase):
    def __init__(self, pack_path):
        super().__init__()
        self._pack_path = pack_path

        self.cam.set_pos(*_HOME_POS)
        self.cam.look_at(*_HOME_LOOK)
        attach_fly_camera(self)

        checker = make_checker_texture()

        # A big checkered plate: gives an unambiguous visual reference for
        # orientation, perspective, and whether lighting actually varies
        # across a lit surface (vs. one flat, unlit colour).
        self.ground = self.loader.load_model("models/misc/rgbCube")
        self.ground.reparent_to(self.render)
        self.ground.set_scale(30, 30, 0.2)
        self.ground.set_pos(0, 0, -1)
        self.ground.set_texture(checker, 1)

        self._pillars = []
        for _name, pos, scale, render_type in _PILLARS:
            pillar = self.loader.load_model("models/misc/rgbCube")
            pillar.reparent_to(self.render)
            pillar.set_scale(*scale)
            pillar.set_pos(*pos)
            pillar.set_texture(checker, 1)
            self._pillars.append((pillar, render_type))

        self.crystal = self.loader.load_model("models/misc/rgbCube")
        self.crystal.reparent_to(self.render)
        self.crystal.set_pos(0, 4, 0.5)
        self.crystal.set_scale(1.2)

        self.water = self.loader.load_model("models/misc/rgbCube")
        self.water.reparent_to(self.render)
        self.water.set_scale(9, 9, 0.1)
        self.water.set_pos(0, -14, -0.85)

        # Load the pack and shade the whole scene through its pipeline.
        # MINIMUM avoids BSL's cross-frame-history-dependent effects (this
        # runner doesn't have history buffers yet — see README) and renders
        # noticeably cleaner; try [2]/[3] for HIGH/ULTRA once that lands.
        self._profile_name = "MINIMUM"
        self.pipe = PipelineRenderer(self, pack_path, world="world0", profile=self._profile_name)
        self.pipe.set_render_type(self.ground, "terrain")
        self.pipe.set_render_type(self.water, "water")
        for pillar, render_type in self._pillars:
            self.pipe.set_render_type(pillar, render_type)

        self._render_types = self.pipe.resolver.types()
        self._rt_index = self._render_types.index("glowing") if "glowing" in self._render_types else 0
        self._block_index = 0
        self.pipe.set_render_type(self.crystal, self._render_types[self._rt_index])
        self._apply_block()

        self._debug_names = list(self.pipe.debug_textures().keys())
        self._debug_index = -1  # -1 == normal composited final view
        self._debug_quad = self._build_debug_quad()

        print(self.pipe.describe())
        print("\n" + HELP)
        self._help_text = OnscreenText(
            text=HELP, pos=(-1.3, 0.95), scale=0.045, align=TextNode.ALeft,
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.6), mayChange=False)
        self._status_text = OnscreenText(
            text="", pos=(-1.3, -0.92), scale=0.05, align=TextNode.ALeft,
            fg=(1, 1, 0.5, 1), shadow=(0, 0, 0, 0.6), mayChange=True)
        self._update_status()

        self.accept("1", self._profile, ["MINIMUM"])
        self.accept("2", self._profile, ["HIGH"])
        self.accept("3", self._profile, ["ULTRA"])
        self.accept("z", self._toggle_shadow)
        self.accept("r", self._cycle_render_type)
        self.accept("b", self._cycle_block)
        self.accept("[", self._cycle_debug, [-1])
        self.accept("]", self._cycle_debug, [1])
        self.accept("0", self._reset_debug)
        self.accept("p", self._swap_pack)
        self.accept("k", self._screenshot)
        self.accept("h", self._toggle_help)
        self.accept("g", self._go_home)

    def _go_home(self):
        self.cam.set_pos(*_HOME_POS)
        self.cam.look_at(*_HOME_LOOK)
        print("camera reset to the establishing shot")

    # -- raw buffer viewer -------------------------------------------------
    def _build_debug_quad(self):
        cm = CardMaker("mcshader-debug-quad")
        cm.set_frame_fullscreen_quad()
        quad = self.render2d.attach_new_node(cm.generate())
        quad.set_bin("fixed", 1000)  # draw on top of the pipeline's own final quad
        quad.set_depth_test(False)
        quad.set_depth_write(False)
        quad.hide()
        return quad

    def _cycle_debug(self, direction):
        names = self._debug_names
        if not names:
            return
        if self._debug_index < 0:
            self._debug_index = 0 if direction > 0 else len(names) - 1
        else:
            self._debug_index = (self._debug_index + direction) % len(names)
        tex = self.pipe.debug_textures()[names[self._debug_index]]
        self._debug_quad.set_texture(tex, 1)
        self._debug_quad.show()
        self._update_status()

    def _reset_debug(self):
        self._debug_index = -1
        self._debug_quad.hide()
        self._update_status()

    # -- live pipeline pokes ------------------------------------------------
    def _profile(self, name):
        self._profile_name = name
        self.pipe.apply_profile(name)
        self._debug_names = list(self.pipe.debug_textures().keys())
        self._update_status()
        print(f"profile -> {name}")

    def _toggle_shadow(self):
        self.pipe.set_option("SHADOW", not self.pipe.options.get("SHADOW"))
        self.pipe.recompile()
        self._update_status()
        print("SHADOW ->", self.pipe.options.get("SHADOW"))

    def _cycle_render_type(self):
        self._rt_index = (self._rt_index + 1) % len(self._render_types)
        rt = self._render_types[self._rt_index]
        self.pipe.set_render_type(self.crystal, rt)
        self._apply_block()  # set_render_type alone doesn't reapply the block id
        self._update_status()
        print("crystal render_type ->", rt)

    def _cycle_block(self):
        self._block_index = (self._block_index + 1) % len(_BLOCKS)
        self._apply_block()
        self._update_status()
        print("crystal block ->", _BLOCKS[self._block_index])

    def _apply_block(self):
        mc_id = _BLOCKS[self._block_index]
        self.pipe.set_block_id(self.crystal, self.pipe.resolver.block_id(mc_id))

    def _swap_pack(self):
        self.pipe.swap_pack(self._pack_path)
        self._apply_block()  # swap_pack() replays render-type tags but not block ids
        self._debug_names = list(self.pipe.debug_textures().keys())
        self._reset_debug()
        print("swap_pack() reloaded — render-type tags preserved")

    def _screenshot(self):
        path = self.screenshot(namePrefix="mcshader")
        print("saved", path)

    def _toggle_help(self):
        if self._help_text.is_hidden():
            self._help_text.show()
        else:
            self._help_text.hide()

    def _update_status(self):
        rt = self._render_types[self._rt_index]
        block = _BLOCKS[self._block_index]
        view = self._debug_names[self._debug_index] if self._debug_index >= 0 else "final"
        self._status_text.setText(
            f"profile={self._profile_name}  shadow={self.pipe.options.get('SHADOW')}  "
            f"crystal={rt}/{block}  view={view}"
        )


if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "Shaders/"
    Demo(pack).run()
