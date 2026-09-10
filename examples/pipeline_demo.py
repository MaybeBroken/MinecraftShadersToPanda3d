"""Run a Minecraft shaderpack's deferred pipeline over a real Panda3D scene.

    pip install panda3d
    python examples/pipeline_demo.py [path-to-pack]

Uses Panda3D's own bundled tutorial content — the "environment" island (real
ground/rock/tree/bamboo textures, ~90 pieces of actual geometry) and the
animated panda actor — instead of hand-placed primitives, tagged in bulk by
name pattern (see `_tagging.py`) the way a real game's assets would be: by
naming convention, not object-by-object. Lets the pack shade the whole frame
(shadow -> gbuffers -> deferred -> composite -> final) over that scene. Fly
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
from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, CardMaker

from mcshader.engine import PipelineRenderer
from _flycam import attach_fly_camera
from _tagging import tag_by_pattern
from _settings_panel import SettingsPanel

HELP = """mcshader pipeline demo
WASD/QE fly, arrows look, shift boost   [g] reset camera to the establishing shot
[1][2][3] profile MINIMUM/HIGH/ULTRA
[o] live shader settings panel (every option the pack exposes, tweak + see it apply)
[z] toggle SHADOW option    [t] pause/resume the panda's walk
[r] cycle the panda's render type
[b] cycle the panda's block id
[[ ]] step raw colortex buffers, [0] back to final
[p] swap_pack() reload (tags survive)
[k] screenshot   [h] hide/show this help"""

_PROFILES = ["MINIMUM", "LOW", "MEDIUM", "HIGH", "ULTRA"]

# A few blocks chosen to visibly exercise different pack behaviour (waving
# foliage, an emissive light source, a plain block) via the same block-id API.
_BLOCKS = ["minecraft:sea_lantern", "minecraft:oak_leaves", "minecraft:stone"]

_HOME_POS = (0, -170, 55)
_HOME_LOOK = (0, 20, 10)

# Bulk-tag rules for Panda3D's bundled "environment" island model, tried in
# order against each GeomNode's name — the realistic way a scene with ~90
# pieces of geometry gets tagged (by naming convention), not by hand-picking
# every object. Ground gets the real terrain program; rocks and tree trunks
# are opaque static geometry; bamboo/branches/reed "planes" all carry Panda's
# own TransparencyAttrib (they're alpha-cut leaf cards) so "entity" — the
# render type actual foliage-waving programs key off — fits them best.
_ENV_RULES = [
    (r"^Ground", "terrain"),
    (r"^Rock", "block_entity"),
    (r"^TreeTrunk", "block_entity"),
    (r"^(Branch|Bamboo|Plane|Cylinder)", "entity"),
]


class Demo(ShowBase):
    def __init__(self, pack_path):
        super().__init__()
        self._pack_path = pack_path

        self.cam.set_pos(*_HOME_POS)
        self.cam.look_at(*_HOME_LOOK)
        attach_fly_camera(self, speed=40.0)

        # The classic Panda3D tutorial island: real textures (ground, rock,
        # tree bark, bamboo, reeds) on real geometry, at the scale/position
        # from Panda3D's own "Hello World" tutorial.
        self.environ = self.loader.load_model("models/environment")
        self.environ.reparent_to(self.render)
        self.environ.set_scale(0.25, 0.25, 0.25)
        self.environ.set_pos(-8, 42, 0)

        # An animated character actor — proves the pipeline shades skinned,
        # moving geometry correctly, not just static props.
        self.actor = Actor("models/panda-model", {"walk": "models/panda-walk4"})
        self.actor.reparent_to(self.render)
        self.actor.set_scale(0.005)
        self.actor.set_pos(0, 20, 0)
        self.actor.set_h(180)
        self.actor.loop("walk")
        self._animating = True


        # Load the pack and shade the whole scene through its pipeline.
        # MINIMUM is still the lightest/cleanest profile, but HIGH/ULTRA's
        # TAA/AO denoising (which leans on colortex2's cross-frame history)
        # now actually converges over a few dozen frames instead of reading
        # permanently-zero history — see README.
        self._profile_name = "MINIMUM"
        self.pipe = PipelineRenderer(self, pack_path, world="world0", profile=self._profile_name)
        self.sky = self.pipe.build_sky()
        # BSL ships the sun/moon disc off by default (an Iris video-settings
        # default, not a technical limitation) — flip it on so there's
        # something to actually see in the sky; toggle it back off (along
        # with everything else) via the [o] settings panel.
        self.pipe.set_option("SHADER_SUN_MOON", True)
        self.pipe.recompile()

        tag_counts = tag_by_pattern(self.pipe, self.environ, _ENV_RULES, default="terrain")

        self._render_types = self.pipe.resolver.types()
        self._rt_index = self._render_types.index("entity") if "entity" in self._render_types else 0
        self._block_index = 0
        self.pipe.set_render_type(self.actor, self._render_types[self._rt_index])
        tag_counts[self._render_types[self._rt_index]] = tag_counts.get(self._render_types[self._rt_index], 0) + 1
        self._apply_block()

        self._debug_names = list(self.pipe.debug_textures().keys())
        self._debug_index = -1  # -1 == normal composited final view
        self._debug_quad = self._build_debug_quad()

        # The pack's whole Iris-style options menu (every screen/toggle/
        # slider it declares), live: change a value, see it recompile and
        # apply immediately.
        self.settings = SettingsPanel(
            self, self.pipe, profiles=_PROFILES, profile_name=self._profile_name,
            on_change=self._on_settings_change)

        print(self.pipe.describe())
        print("tagged by pattern:", tag_counts)
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
        self.accept("t", self._toggle_animation)
        self.accept("r", self._cycle_render_type)
        self.accept("b", self._cycle_block)
        self.accept("[", self._cycle_debug, [-1])
        self.accept("]", self._cycle_debug, [1])
        self.accept("0", self._reset_debug)
        self.accept("p", self._swap_pack)
        self.accept("k", self._screenshot)
        self.accept("h", self._toggle_help)
        self.accept("g", self._go_home)
        self.accept("o", self.settings.toggle)

    def _go_home(self):
        self.cam.set_pos(*_HOME_POS)
        self.cam.look_at(*_HOME_LOOK)
        print("camera reset to the establishing shot")

    def _toggle_animation(self):
        self._animating = not self._animating
        if self._animating:
            self.actor.loop("walk")
        else:
            self.actor.stop()
        self._update_status()
        print("panda animation ->", "playing" if self._animating else "paused")

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
        # apply_profile() resets every option to the profile's own defaults
        # (BSL ships the sun/moon disc off) — reapply the demo's own default
        # override, same as at startup.
        self.pipe.set_option("SHADER_SUN_MOON", True)
        self.pipe.recompile()
        self._debug_names = list(self.pipe.debug_textures().keys())
        if hasattr(self, "settings"):
            self.settings.set_profile_name(name)
        self._update_status()
        print(f"profile -> {name}")

    def _on_settings_change(self):
        """The settings panel just applied an option change (recompile()d
        the pipeline) — colortex buffers are fresh objects, so the debug
        viewer's texture references need refreshing too."""
        self._debug_names = list(self.pipe.debug_textures().keys())
        if self._debug_index >= 0:
            self._cycle_debug(0)  # re-bind the currently-viewed buffer
        self._update_status()

    def _toggle_shadow(self):
        self.pipe.set_option("SHADOW", not self.pipe.options.get("SHADOW"))
        self.pipe.recompile()
        self._update_status()
        print("SHADOW ->", self.pipe.options.get("SHADOW"))

    def _cycle_render_type(self):
        self._rt_index = (self._rt_index + 1) % len(self._render_types)
        rt = self._render_types[self._rt_index]
        self.pipe.set_render_type(self.actor, rt)
        self._apply_block()  # set_render_type alone doesn't reapply the block id
        self._update_status()
        print("panda render_type ->", rt)

    def _cycle_block(self):
        self._block_index = (self._block_index + 1) % len(_BLOCKS)
        self._apply_block()
        self._update_status()
        print("panda block ->", _BLOCKS[self._block_index])

    def _apply_block(self):
        mc_id = _BLOCKS[self._block_index]
        self.pipe.set_block_id(self.actor, self.pipe.resolver.block_id(mc_id))

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
            f"panda={rt}/{block}  view={view}"
        )


if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "Shaders/"
    Demo(pack).run()
