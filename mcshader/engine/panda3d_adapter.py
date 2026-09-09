"""Panda3D backend for the effect system.

``apply(nodepath, "glow")`` compiles the effect's GLSL, attaches it to the
NodePath, and seeds every parameter with its default (overridable per call).
Frame-driven inputs the effect declares — ``u_time`` and ``u_camera_pos`` — are
kept up to date by a single shared task, so waving sways and glows pulse without
any per-frame work from the caller.

Panda3D is imported lazily inside methods so the rest of ``mcshader`` (the
translator, registry, tests) works in environments where Panda3D isn't
installed.
"""

from __future__ import annotations

from typing import Any

from .base import EngineAdapter
from ..registry import ShaderRegistry


class Panda3DAdapter(EngineAdapter):
    def __init__(self, base: Any = None, registry: ShaderRegistry | None = None):
        """``base`` is a ShowBase; required only for automatic uniform feeding."""
        super().__init__(registry)
        self.base = base
        self._managed: list[tuple[Any, list[str]]] = []  # (nodepath, auto_inputs)
        self._task_started = False
        self._fallback_cube = None

    # -- public API ------------------------------------------------------
    def apply(self, nodepath: Any, shader_id: str, **overrides: Any) -> Any:
        from panda3d.core import Shader

        effect = self.registry.get(shader_id)
        if effect.dialect != "panda3d":
            raise ValueError(
                f"effect {shader_id!r} is in the {effect.dialect!r} dialect; "
                "the Panda3D adapter needs panda3d-dialect GLSL"
            )
        shader = Shader.make(
            Shader.SL_GLSL, vertex=effect.vertex, fragment=effect.fragment
        )
        nodepath.set_shader(shader)

        values = effect.defaults()
        values.update(overrides)
        for name, value in values.items():
            self._set_input(nodepath, name, value)

        # Reflection needs a cubemap; bind a sky-coloured fallback if unset.
        if "u_env_map" in effect.params and "u_env_map" not in overrides:
            self._set_input(nodepath, "u_env_map", self._fallback_cubemap())

        if effect.auto_inputs:
            self._managed.append((nodepath, list(effect.auto_inputs)))
            self._ensure_task()
        return shader

    def clear(self, nodepath: Any) -> None:
        nodepath.clear_shader()
        self._managed = [(np, ai) for np, ai in self._managed if np != nodepath]

    # -- helpers ---------------------------------------------------------
    def _set_input(self, nodepath: Any, name: str, value: Any) -> None:
        if isinstance(value, (tuple, list)):
            nodepath.set_shader_input(name, *value)
        else:
            nodepath.set_shader_input(name, value)

    def _ensure_task(self) -> None:
        if self._task_started or self.base is None:
            return
        self.base.taskMgr.add(self._update_task, "mcshader-uniforms")
        self._task_started = True

    def _update_task(self, task: Any) -> Any:
        from direct.task import Task
        from panda3d.core import ClockObject

        t = ClockObject.get_global_clock().get_frame_time()
        cam_pos = None
        if getattr(self.base, "camera", None) is not None:
            p = self.base.camera.get_pos(self.base.render)
            cam_pos = (p.x, p.y, p.z)
        for nodepath, auto_inputs in self._managed:
            if "u_time" in auto_inputs:
                nodepath.set_shader_input("u_time", t)
            if "u_camera_pos" in auto_inputs and cam_pos is not None:
                nodepath.set_shader_input("u_camera_pos", *cam_pos)
        return Task.cont

    def _fallback_cubemap(self) -> Any:
        """A 1x1 sky-blue cubemap so reflection compiles without an env map."""
        if self._fallback_cube is not None:
            return self._fallback_cube
        from panda3d.core import Texture

        tex = Texture("mcshader-env-fallback")
        tex.setup_cube_map(1, Texture.T_unsigned_byte, Texture.F_rgb)
        r, g, b = 135, 180, 235  # sky blue
        tex.set_ram_image(bytes([b, g, r]) * 6)  # Panda stores BGR, 6 faces
        self._fallback_cube = tex
        return tex
