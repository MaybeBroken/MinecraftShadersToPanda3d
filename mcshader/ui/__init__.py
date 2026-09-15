"""Optional on-screen tools built on top of the pipeline: fly camera, bulk
tagging, and the pack's own live settings panel.

Everything here is a convenience — the pipeline runs without any of it — so
each symbol is imported lazily: importing ``mcshader.ui`` never pulls in
Panda3D or DirectGUI unless you actually touch one of them.
"""

from __future__ import annotations

__all__ = ["attach_fly_camera", "tag_by_pattern", "SettingsPanel"]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "attach_fly_camera":
        from .flycam import attach_fly_camera
        return attach_fly_camera
    if name == "tag_by_pattern":
        from .tagging import tag_by_pattern
        return tag_by_pattern
    if name == "SettingsPanel":
        from .settings_panel import SettingsPanel
        return SettingsPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
