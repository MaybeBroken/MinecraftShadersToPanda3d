"""Raw-OpenGL helper for engines without a scene graph.

Unlike :class:`~mcshader.engine.panda3d_adapter.Panda3DAdapter`, a raw GL loop
owns its own geometry, VAOs, and draw calls, so this class does not "attach an
effect to an object". Instead it compiles an effect into a GL program and hands
back the program id plus cached uniform locations, leaving the draw loop to the
caller. It's the seam for wiring the effect library into Harfang's low-level GL
mode, moderngl, pyglet, or hand-rolled PyOpenGL renderers.

Built-in effects are authored in the Panda3D dialect (``p3d_*`` inputs). For a
raw-GL target, decompile with ``target="generic"`` (producing ``mc_*`` inputs
you bind yourself), or bind the ``p3d_*`` names to your own buffers.

PyOpenGL is imported lazily so importing ``mcshader`` never requires it.
"""

from __future__ import annotations

from typing import Any

from ..effects.base import Effect
from ..registry import ShaderRegistry, default_registry

__all__ = ["GenericGLAdapter", "CompiledProgram"]


class CompiledProgram:
    """A linked GL program plus lazily-cached uniform locations."""

    def __init__(self, program: int, effect: Effect):
        self.program = program
        self.effect = effect
        self._locations: dict[str, int] = {}

    def uniform_location(self, name: str) -> int:
        from OpenGL.GL import glGetUniformLocation

        if name not in self._locations:
            self._locations[name] = glGetUniformLocation(self.program, name)
        return self._locations[name]


class GenericGLAdapter:
    def __init__(self, registry: ShaderRegistry | None = None):
        self.registry = registry or default_registry()

    def compile(self, shader_id: str) -> CompiledProgram:
        """Compile and link the effect registered under ``shader_id``."""
        from OpenGL.GL import (
            GL_FRAGMENT_SHADER,
            GL_VERTEX_SHADER,
        )
        from OpenGL.GL.shaders import compileProgram, compileShader

        effect = self.registry.get(shader_id)
        program = compileProgram(
            compileShader(effect.vertex, GL_VERTEX_SHADER),
            compileShader(effect.fragment, GL_FRAGMENT_SHADER),
        )
        return CompiledProgram(int(program), effect)

    def available_ids(self) -> list[str]:
        return self.registry.ids()
