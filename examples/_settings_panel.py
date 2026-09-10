"""A live, in-game settings panel for a PipelineRenderer's shader options.

Renders the pack's own Iris-style options menu (``ShaderOptions.menu_tree`` —
the same screens/toggles/sliders OptiFine/Iris would show in their video
settings) as a navigable DirectGUI panel, and applies every change
immediately via ``pipe.set_option()`` + ``pipe.recompile()`` — so you can
tweak a shader `#define`/`const` value and see its effect in real time,
without editing GLSL or restarting.

    from _settings_panel import SettingsPanel
    panel = SettingsPanel(base, pipe, profiles=["MINIMUM", "LOW", "MEDIUM", "HIGH", "ULTRA"])
    base.accept("o", panel.toggle)

Purely a demo/dev-tool convenience built on the engine-side options API
(``mcshader.config.options.ShaderOptions``) — nothing here is pipeline
machinery, and nothing here touches shader source directly.
"""

from __future__ import annotations

from direct.gui.DirectGui import (
    DGG, DirectButton, DirectCheckButton, DirectFrame, DirectLabel,
    DirectScrolledFrame,
)
from panda3d.core import TextNode

__all__ = ["SettingsPanel"]

_ROW_H = 0.062
_PANEL_L, _PANEL_R = -0.68, 0.72
_ROW_INSET = 0.06
_TEXT_COLOR = (0.92, 0.92, 0.92, 1)
_DIM_COLOR = (0.55, 0.55, 0.55, 1)
_BTN_COLOR = (0.22, 0.22, 0.22, 1)
_SCREEN_COLOR = (0.30, 0.42, 0.55, 1)


class SettingsPanel:
    """A collapsible panel over the pack's whole options tree, docked right."""

    def __init__(self, base, pipe, *, profiles: list[str] | None = None,
                 profile_name: str = "", on_change=None):
        self.base = base
        self.pipe = pipe
        self.profiles = profiles or []
        self._profile_name = profile_name
        self.on_change = on_change  # optional callback(), fired after any apply
        self._path: list[str] = []  # breadcrumb of screen names from the root
        self._rows: list[object] = []

        self.frame = DirectFrame(
            frameColor=(0.05, 0.05, 0.06, 0.88),
            frameSize=(_PANEL_L - 0.03, _PANEL_R + 0.03, -0.95, 0.95),
            pos=(1.02, 0, 0),
            state=DGG.NORMAL,
        )
        self.frame.hide()

        self.title = DirectLabel(
            parent=self.frame, text="", text_scale=0.05,
            text_fg=_TEXT_COLOR, text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0), pos=(_PANEL_L, 0, 0.87),
        )
        self.hint = DirectLabel(
            parent=self.frame,
            text="[o] close   click a category to open it   < > steps a value",
            text_scale=0.032, text_fg=_DIM_COLOR, text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0), pos=(_PANEL_L, 0, 0.80),
        )

        self.scroll = DirectScrolledFrame(
            parent=self.frame,
            frameColor=(0.09, 0.09, 0.10, 1),
            frameSize=(_PANEL_L, _PANEL_R, -0.90, 0.74),
            canvasSize=(_PANEL_L, _PANEL_R, -1.0, 0.0),
            scrollBarWidth=0.035,
            state=DGG.NORMAL,
        )
        self.canvas = self.scroll.getCanvas()

        self.refresh()

    # -- visibility -----------------------------------------------------
    def toggle(self) -> None:
        if self.frame.is_hidden():
            self.refresh()
            self.frame.show()
        else:
            self.frame.hide()

    def show(self) -> None:
        self.refresh()
        self.frame.show()

    def hide(self) -> None:
        self.frame.hide()

    # -- tree navigation --------------------------------------------------
    def _current_node(self) -> dict:
        node = self.pipe.options.menu_tree(self.pipe.props)
        for name in self._path:
            for el in node["elements"]:
                if el.get("type") == "screen" and el["name"] == name:
                    node = el["screen"]
                    break
        return node

    def _enter(self, name: str) -> None:
        self._path.append(name)
        self.refresh()

    def _back(self) -> None:
        if self._path:
            self._path.pop()
            self.refresh()

    # -- applying a change --------------------------------------------------
    def _apply(self, name: str, value: object) -> None:
        self.pipe.set_option(name, value)
        self.pipe.recompile()
        if self.on_change:
            self.on_change()
        self.refresh()

    def _step_enum(self, name: str, allowed: list[str], value: object, delta: int) -> None:
        cur = str(value)
        idx = allowed.index(cur) if cur in allowed else 0
        idx = (idx + delta) % len(allowed)
        self._apply(name, allowed[idx])

    def set_profile_name(self, name: str) -> None:
        """Keep the panel's profile row in sync with an external switch
        (e.g. the demo's own [1][2][3] profile hotkeys)."""
        self._profile_name = name
        if not self.frame.is_hidden():
            self.refresh()

    def _step_profile(self, delta: int) -> None:
        if not self.profiles:
            return
        idx = self.profiles.index(self._profile_name) if self._profile_name in self.profiles else 0
        idx = (idx + delta) % len(self.profiles)
        self._profile_name = self.profiles[idx]
        self.pipe.apply_profile(self._profile_name)
        if self.on_change:
            self.on_change()
        self.refresh()

    # -- rendering --------------------------------------------------------
    def refresh(self) -> None:
        for row in self._rows:
            row.destroy()
        self._rows.clear()

        node = self._current_node()
        self.title["text"] = "Settings: " + (" > ".join(self._path) if self._path else "root")

        y = -0.02
        if self._path:
            self._rows.append(DirectButton(
                parent=self.canvas, text="< back", text_scale=0.045,
                text_fg=_TEXT_COLOR, frameColor=_BTN_COLOR,
                frameSize=(-0.5, 0.5, -0.022, 0.032),
                pos=(_PANEL_L + 0.5, 0, y), command=self._back,
            ))
            y -= _ROW_H

        if not self._path and self.profiles:
            y = self._row_profile(y)

        for el in node["elements"]:
            t = el.get("type")
            if t == "spacer":
                continue
            elif t == "screen":
                y = self._row_screen(el, y)
            elif t == "option" and "kind" in el:
                y = self._row_option(el, y)
            # An "option" element with no "kind" (e.g. the menu's own
            # "<profile>" placeholder) isn't a real registered option —
            # the dedicated profile row above already covers it.

        self.scroll["canvasSize"] = (_PANEL_L, _PANEL_R, y - 0.02, 0.02)

    def _row_profile(self, y: float) -> float:
        self._rows.append(_stepper_row(
            self.canvas, y, f"profile: {self._profile_name}",
            on_prev=lambda: self._step_profile(-1),
            on_next=lambda: self._step_profile(1),
        ))
        return y - _ROW_H

    def _row_screen(self, el: dict, y: float) -> float:
        name = el["name"]
        self._rows.append(DirectButton(
            parent=self.canvas, text=f"{name}  >",
            text_scale=0.045, text_fg=(1, 1, 1, 1), text_align=TextNode.ALeft,
            frameColor=_SCREEN_COLOR, relief=DGG.FLAT,
            frameSize=(0, _PANEL_R - _PANEL_L - 0.02, -0.022, 0.032),
            pos=(_PANEL_L + 0.01, 0, y),
            command=self._enter, extraArgs=[name],
        ))
        return y - _ROW_H

    def _row_option(self, el: dict, y: float) -> float:
        name, kind, value, allowed = el["name"], el["kind"], el["value"], el["allowed"]
        if kind == "toggle":
            cb = DirectCheckButton(
                parent=self.canvas, text=name, text_scale=0.042,
                text_fg=_TEXT_COLOR, text_align=TextNode.ALeft,
                text_pos=(0.05, -0.014), boxPlacement="left",
                frameColor=(0, 0, 0, 0), scale=1.0,
                # DirectCheckButton's indicator is a DirectLabel whose own
                # frame auto-sizes to its (' '/'*') text — without an
                # explicit small text_scale here it defaults to a ~1-unit
                # font size, rendering as a huge pale rectangle over
                # everything below it instead of a small checkbox.
                indicator_text_scale=0.05,
                indicator_frameColor=(0.85, 0.85, 0.85, 1),
                indicator_text_fg=(0.1, 0.1, 0.1, 1),
                indicatorValue=bool(value),
                pos=(_PANEL_L + _ROW_INSET, 0, y),
                command=lambda status, name=name: self._apply(name, bool(status)),
            )
            self._rows.append(cb)
        else:
            label = name if len(allowed) <= 1 else f"{name}: {value}"
            self._rows.append(_stepper_row(
                self.canvas, y, label,
                on_prev=lambda name=name, allowed=allowed, value=value:
                    self._step_enum(name, allowed, value, -1),
                on_next=lambda name=name, allowed=allowed, value=value:
                    self._step_enum(name, allowed, value, 1),
                disabled=len(allowed) <= 1,
            ))
        return y - _ROW_H


def _stepper_row(parent, y, label, *, on_prev, on_next, disabled=False):
    row = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0),
                       frameSize=(_PANEL_L, _PANEL_R, -0.022, 0.032), pos=(0, 0, y))
    DirectLabel(parent=row, text=label, text_scale=0.040, text_fg=_TEXT_COLOR,
                text_align=TextNode.ALeft, frameColor=(0, 0, 0, 0),
                pos=(_PANEL_L + _ROW_INSET, 0, -0.012))
    if not disabled:
        DirectButton(parent=row, text="<", text_scale=0.04, frameColor=_BTN_COLOR,
                     frameSize=(-0.03, 0.03, -0.018, 0.028),
                     pos=(_PANEL_R - 0.10, 0, -0.008), command=on_prev)
        DirectButton(parent=row, text=">", text_scale=0.04, frameColor=_BTN_COLOR,
                     frameSize=(-0.03, 0.03, -0.018, 0.028),
                     pos=(_PANEL_R - 0.03, 0, -0.008), command=on_next)
    return row
