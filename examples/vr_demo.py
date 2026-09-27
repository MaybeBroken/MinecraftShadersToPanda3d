"""A Minecraft shaderpack in a VR headset: mcshader + Panda3d-VR.

    pip install -e path/to/Panda3d-VR     # makes `import panda3d_vr` work
    python examples/vr_demo.py [pack.zip] [--prefs saved.prefs]

``--prefs`` applies a settings file saved by a game's settings panel (the
``@pack``/``@profile`` + ``OPTION=value`` format LSE writes), e.g.
``--prefs ../LSE/module/minecraft_shader_pack/BSL_v10.0.prefs``.

With a headset connected (Quest over Link, Index, ...), each eye renders the
pack's whole pipeline -- shadows, deferred lighting, composites, TAA -- from
its own camera. Without one, the window shows the same pack through
Panda3d-VR's desktop simulator (right-drag to look, WASD to move).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d_vr import BaseVrApp, Locomotion  # noqa: E402

import mcshader  # noqa: E402


def apply_prefs(app, path):
    """Apply a saved ``@pack``/``@profile`` + ``OPTION=value`` settings file."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    header = dict(line.strip()[1:].split("=", 1) for line in text.splitlines()
                  if line.strip().startswith("@") and "=" in line)
    pipe = app.pipe
    pack = header.get("pack", "")
    if pack and pack != (pipe.pack.name or ""):
        print("prefs: written for pack %r, running %r -- not applied" % (pack, pipe.pack.name))
        return
    profile = header.get("profile", "")
    if profile and profile != pipe.profile_name:
        # Straight onto the options (like LSE does): apply_profile would recompile
        # and then lose the saved overrides applied below.
        pipe.options.apply_profile(profile, pipe.props)
        pipe.profile_name = profile
    pipe.options.loads(text)
    app.reload()
    print("prefs: loaded %s (profile %s)" % (os.path.basename(path), profile or "unchanged"))


class ShadedVR(BaseVrApp):
    def __init__(self, pack=None, prefs=None):
        # msaa=0: the pack anti-aliases itself (TAA), and Panda3d-VR's own eye
        # buffers sit idle while mcshader supplies the eye images.
        super().__init__(msaa=0, show_stats=True, debug_keys=True, near=0.05, far=300)
        self.shaders = mcshader.init(self, pack=pack, profile="MEDIUM", sky=True, vr=self.vr)
        if prefs:
            apply_prefs(self.shaders, prefs)
        # Panda's sample environment is ~1230 units across; 0.05 makes it a
        # believable ~60 m glade in a headset (1 unit = 1 m).
        self.shaders.load("models/environment", type="terrain", scale=0.05, pos=(-1.6, 8.4, 0))
        self.shaders.load("models/misc/sphere", type="entity", pos=(0, 6, 1.5), scale=0.4,
                          block="minecraft:sea_lantern")
        self.locomotion = Locomotion(self.vr)
        self.accept("vr-right-primary", lambda c: self.shaders._toggle_pack())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("pack", nargs="?", help="shaderpack .zip or directory")
    args.add_argument("--prefs", help="saved settings file to apply")
    opts = args.parse_args()
    ShadedVR(opts.pack, opts.prefs).run()
