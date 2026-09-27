"""Render a shaderpack in a VR headset through Panda3d-VR.

    vr = VRManager(base)                        # or a BaseVrApp's self.vr
    app = mcshader.init(base, pack="BSL.zip", vr=vr)

When a headset connects (``vr-eyes-ready``) the pipeline is rebuilt as two
eye views -- each eye camera gets its own gbuffer, composite chain and TAA
history, all sharing one set of compiled programs, one shadow map and one sky
-- and each eye's final image (plus its depth, for the compositor's
reprojection) is handed to the headset with ``vr.set_eye_source``. When it
disconnects, the pipeline falls back to the desktop window, which in
Panda3d-VR's simulator follows the head.

Nothing here imports Panda3d-VR: it relies only on its public contract
(``eyes_ready``, ``eye_size``, ``eye_cameras``, ``set_eye_source``,
``clear_eye_sources``, ``get_hidden_area_camera``, ``set_clip_planes`` and the
``vr-eyes-ready`` / ``vr-disconnected`` events).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

__all__ = ["VRBridge", "VR_OPTION_OVERRIDES"]

#: Pack options switched off while rendering to a headset. They simulate a
#: *camera* looking at a flat screen -- a focus point at the screen centre,
#: flares and dirt mirrored through the screen centre, darkened screen edges,
#: colour fringing, smearing -- none of which survives two eyes looking
#: through lenses: each eye gets its own "screen centre", so the effect no
#: longer agrees between the eyes. Only names the pack actually has are
#: touched; the saved values come back when the headset is gone.
VR_OPTION_OVERRIDES = {
    "DOF": False,
    "LENS_FLARE": False,
    "DIRTY_LENS": False,
    "VIGNETTE": False,
    "CHROMATIC_ABERRATION": 0,
    "MOTION_BLUR": False,
}


def _same(a: Any, b: Any) -> bool:
    """Option values compare as the pack stores them ('0' == 0, 'true' == True)."""
    return str(a).lower() == str(b).lower()


class VRBridge:
    def __init__(self, pipe: Any, vr: Any, *, hidden_area: bool = False,
                 option_overrides: dict | None = None):
        """``hidden_area`` draws the headset's hidden-area mesh into each
        gbuffer first, so no pass shades pixels the lenses can't show. It is
        off by default: the mesh sits at the near plane, and screen-space
        effects that sample around it (SSAO, SSR, bloom) can pick up a faint
        edge where it meets the visible image.

        ``option_overrides`` replaces :data:`VR_OPTION_OVERRIDES` (pass ``{}``
        to keep every pack option as it is)."""
        from direct.showbase.DirectObject import DirectObject

        self.pipe = pipe
        self.vr = vr
        self.hidden_area = hidden_area
        self.option_overrides = dict(VR_OPTION_OVERRIDES if option_overrides is None
                                     else option_overrides)
        self._saved_options: dict[str, Any] = {}
        self.active = False
        self.textures: list[Any] = []
        self._events = DirectObject()
        self._events.accept("vr-eyes-ready", self._on_eyes_ready)
        self._events.accept("vr-disconnected", self._to_window)
        if getattr(vr, "eyes_ready", False) and getattr(vr, "connected", False):
            self._on_eyes_ready(vr)

    def _on_eyes_ready(self, vr: Any) -> None:
        from panda3d.core import Texture
        from .engine.panda3d_pipeline import PipelineView

        size = tuple(vr.eye_size)
        if self.active and size == getattr(self, "_size", None):
            return  # already rendering to these eyes
        self._size = size
        self.textures = [Texture(f"mcshader-vr-eye{i}") for i in range(2)]
        views = []
        for i, tex in enumerate(self.textures):
            underlays = []
            if self.hidden_area:
                cam = vr.get_hidden_area_camera(i)
                if cam is not None:
                    underlays.append(cam)
            views.append(PipelineView(f"vr-eye{i}", vr.eye_cameras[i], size=size,
                                      output=tex, underlays=underlays, float_depth=True))
        changed = self._apply_overrides()
        self.pipe.set_views(views, reparse=changed)
        for i, view in enumerate(views):
            vr.set_eye_source(i, self.textures[i], view.depth_tex)
        self.active = True

    def _to_window(self, *_: Any) -> None:
        from .engine.panda3d_pipeline import PipelineView

        if not self.active:
            return
        self.active = False
        self.vr.clear_eye_sources()
        changed = self._restore_options()
        self.pipe.set_views([PipelineView("window", self.pipe.base.cam)], reparse=changed)

    def _apply_overrides(self) -> bool:
        options = self.pipe.options
        changed = False
        for name, value in self.option_overrides.items():
            if name not in options.options:
                continue
            current = options.get(name)
            if _same(current, value):
                continue
            # Anything else is the user's own choice (first time, or changed in a
            # settings panel since): remember it, then override it again.
            self._saved_options[name] = current
            options.set(name, value)
            changed = True
        if changed:
            print("[mcshader] VR: switched off screen-only effects: "
                  + ", ".join(sorted(self._saved_options)))
        return changed

    def reapply_overrides(self) -> bool:
        """After options were changed (a settings panel, loaded prefs), switch
        the screen-only effects off again. Returns True if anything changed."""
        return self.active and self._apply_overrides()

    @contextmanager
    def user_options(self):
        """Temporarily put the user's own values back -- wrap a settings save
        in this so it never writes the VR overrides to disk."""
        options = self.pipe.options
        current = {name: options.get(name) for name in self._saved_options}
        for name, value in self._saved_options.items():
            options.set(name, value)
        try:
            yield options
        finally:
            for name, value in current.items():
                options.set(name, value)

    def _restore_options(self) -> bool:
        options = self.pipe.options
        for name, value in self._saved_options.items():
            options.set(name, value)
        changed = bool(self._saved_options)
        self._saved_options = {}
        return changed

    def set_clip_planes(self, near: float | None, far: float | None) -> None:
        """Clip planes in scene units (the eye lenses belong to the headset)."""
        scale = getattr(self.vr, "world_scale", 1.0)
        near = self.vr.near * scale if near is None else near
        far = self.vr.far * scale if far is None else far
        self.vr.set_clip_planes(near / scale, far / scale)

    def detach(self) -> None:
        self._to_window()
        self._events.ignore_all()
