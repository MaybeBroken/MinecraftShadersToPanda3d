"""A tiny procedural checker texture, shared by the example demos.

No bundled art assets are needed to tell whether a surface is actually being
textured/lit: a checker pattern makes both immediately obvious — contrast
between the two squares confirms texturing, and a brightness difference
between two faces at different angles (e.g. a pillar's top vs. side) confirms
per-surface lighting rather than a flat, unlit fill.
"""

from __future__ import annotations

__all__ = ["make_checker_texture"]


def make_checker_texture(size: int = 64, squares: int = 8, light: float = 1.0,
                          dark: float = 0.15):
    from panda3d.core import PNMImage, Texture

    img = PNMImage(size, size)
    img.add_alpha()
    img.alpha_fill(1.0)
    cell = max(1, size // squares)
    for y in range(size):
        for x in range(size):
            c = light if (x // cell + y // cell) % 2 == 0 else dark
            img.set_xel(x, y, c, c, c)
    tex = Texture("mcshader-checker")
    tex.load(img)
    return tex
