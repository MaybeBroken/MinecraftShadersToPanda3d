"""A live, in-game settings panel for a PipelineRenderer's shader options.

Renders the pack's own Iris-style options menu (``ShaderOptions.menu_tree`` —
the same screens/toggles/sliders OptiFine/Iris would show in their video
settings) as a navigable DirectGUI panel. Edits are *staged* locally as you
click/drag — nothing reaches the pipeline until you press "Apply Changes",
which is when ``pipe.set_option()`` (or ``pipe.apply_profile()``) and
``pipe.recompile()`` actually run. This mirrors Iris/OptiFine's own video
settings screen (which batches edits behind a "Done" button) and, since
``recompile()`` rebuilds the whole shader graph, avoids paying that cost on
every single click.

    from _settings_panel import SettingsPanel
    panel = SettingsPanel(base, pipe, profiles=["MINIMUM", "LOW", "MEDIUM", "HIGH", "ULTRA"])
    base.accept("o", panel.toggle)

Purely a demo/dev-tool convenience built on the engine-side options API
(``mcshader.config.options.ShaderOptions``) — nothing here is pipeline
machinery, and nothing here touches shader source directly.
"""

from __future__ import annotations

from direct.gui.DirectGui import (
    DGG, DirectButton, DirectEntry, DirectFrame, DirectLabel,
    DirectScrolledFrame, DirectSlider,
)
from panda3d.core import PGTop, TextNode

__all__ = ["SettingsPanel"]

# All the frameSize/pos constants below are authored in a fixed local unit
# system, not window-relative aspect2d units — see _reposition() for how
# that's turned into a constant on-screen pixel size. _SCALE=300 was chosen
# to match this panel's original look on a plain 800x600 window.
_SCALE = 300
_MARGIN_NDC = 0.02  # small gap from the true screen edge, as a fraction of
                    # render2d's -1..1 span (~8px on an 800px-wide window)

_ROW_H = 0.062
_PANEL_L, _PANEL_R = -0.68, 0.72
_ROW_INSET = 0.06
_ROW_W = _PANEL_R - _PANEL_L - 0.02
_SCROLL_TOP = 0.74
_SCROLL_BOTTOM = -0.74

_TEXT_COLOR = (0.92, 0.92, 0.92, 1)
_DIM_COLOR = (0.55, 0.55, 0.55, 1)
_MODIFIED_COLOR = (0.95, 0.78, 0.25, 1)
_BTN_COLOR = (0.22, 0.22, 0.22, 1)
_SCREEN_COLOR = (0.30, 0.42, 0.55, 1)
_TOGGLE_ON_COLOR = (0.20, 0.45, 0.28, 1)
_TOGGLE_OFF_COLOR = _BTN_COLOR
_ACCENT_COLOR = (0.24, 0.56, 0.30, 1)


class SettingsPanel:
    """A collapsible panel over the pack's whole options tree, docked right.

    Every toggle/slider/stepper edit only updates local "pending" state —
    the pipeline is untouched until :meth:`_commit` runs (the "Apply
    Changes" button). :meth:`_revert` discards pending edits instead, which
    is free (nothing was ever sent to the pipeline).
    """

    def __init__(self, base, pipe, *, profiles: list[str] | None = None,
                 profile_name: str = "", on_change=None):
        self.base = base
        self.pipe = pipe
        self.profiles = profiles or []
        self._profile_name = profile_name
        self.on_change = on_change  # optional callback(), fired after an apply
        self._path: list[str] = []  # breadcrumb of screen names from the root
        self._rows: list[object] = []

        # Staged-but-not-yet-applied edits.
        self._pending: dict[str, object] = {}
        self._pending_profile: str | None = None

        # DirectGui (the "PG" system) needs to be rooted under a PGTop node
        # to render/pick correctly — aspect2d and pixel2d are each one, and
        # parenting straight under a plain render2d node (no PGTop) instead
        # produces broken rendering (observed: DirectScrolledFrame's clip
        # region collapsing, and the whole window going solid magenta with
        # this panel's full widget tree) even though a lone DirectFrame can
        # look fine there. So: make our own PGTop, attached directly to
        # render2d — whose -1..1 span is *always* exactly the full window
        # on any platform/DPI, with zero recomputation ever needed, unlike
        # aspect2d, which rescales its whole coordinate space to compensate
        # for the window's aspect ratio (correct for the 3D scene, but
        # meant this panel would only ever look "right" at the aspect
        # ratio it happened to be authored/launched at) — and hook it up to
        # the same MouseWatcher aspect2d/pixel2d use, or clicks/drags on
        # this panel's buttons/sliders/entries would never register.
        self._pg_root = self.base.render2d.attachNewNode(PGTop("settings-panel"))
        self._pg_root.node().setMouseWatcher(self.base.mouseWatcherNode)

        # pos/scale are both set by _reposition() below (not fixed here) —
        # see that method for why.
        self.frame = DirectFrame(
            parent=self._pg_root,
            frameColor=(0.05, 0.05, 0.06, 0.88),
            frameSize=(_PANEL_L - 0.03, _PANEL_R + 0.03, -0.95, 0.95),
            state=DGG.NORMAL,
        )
        self.frame.hide()
        self._reposition()
        # Re-dock/re-scale on every resize instead of once at startup.
        self.base.accept("window-event", self._on_window_event)

        self.title = DirectLabel(
            parent=self.frame, text="", text_scale=0.05,
            text_fg=_TEXT_COLOR, text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0), pos=(_PANEL_L, 0, 0.87),
        )
        self.hint = DirectLabel(
            parent=self.frame,
            text="[o] close   click/drag to edit   Apply Changes to commit",
            text_scale=0.032, text_fg=_DIM_COLOR, text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0), pos=(_PANEL_L, 0, 0.80),
        )

        self.scroll = DirectScrolledFrame(
            parent=self.frame,
            frameColor=(0.09, 0.09, 0.10, 1),
            frameSize=(_PANEL_L, _PANEL_R, _SCROLL_BOTTOM, _SCROLL_TOP),
            canvasSize=(_PANEL_L, _PANEL_R, -1.0, 0.0),
            scrollBarWidth=0.035,
            state=DGG.NORMAL,
        )
        self.canvas = self.scroll.getCanvas()

        # Fixed apply bar — lives directly on the frame (not the scroll
        # canvas) so it never scrolls out of view and never gets torn down
        # by refresh()'s row rebuild.
        self.pending_label = DirectLabel(
            parent=self.frame, text="", text_scale=0.032,
            text_fg=_DIM_COLOR, text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0), pos=(_PANEL_L, 0, -0.80),
        )
        self.apply_btn = DirectButton(
            parent=self.frame, text="Apply Changes", text_scale=0.042,
            text_fg=(1, 1, 1, 1), relief=DGG.FLAT, frameColor=_BTN_COLOR,
            frameSize=(-0.34, 0.34, -0.028, 0.038),
            pos=(-0.24, 0, -0.90), command=self._commit,
        )
        self.revert_btn = DirectButton(
            parent=self.frame, text="Revert", text_scale=0.042,
            text_fg=(1, 1, 1, 1), relief=DGG.FLAT, frameColor=_BTN_COLOR,
            frameSize=(-0.16, 0.16, -0.028, 0.038),
            pos=(0.52, 0, -0.90), command=self._revert,
        )

        self.refresh()

    # -- visibility -----------------------------------------------------
    def _on_window_event(self, win=None) -> None:
        self._reposition()

    def _reposition(self) -> None:
        """Pin the panel's top-right corner to the screen's actual
        top-right corner, and give it a fixed real-pixel size — using
        render2d's native NDC space (always exactly -1..1 across the full
        window, on any platform/DPI, with zero recomputation needed) as the
        position reference, rather than an independently-read pixel count.

        The key property: position is written as a function of the *same*
        scale factor (sx/sz) used for sizing, so the docked corner stays
        exactly glued to the screen edge even if that factor is ever wrong
        for some reason (a platform DPI/window-scaling quirk, a resize
        event carrying a stale size, ...). A previous version derived
        scale (fixed once at construction) and position (re-read
        independently on every resize) from two separately-sourced pixel
        counts; any disagreement between them was directly visible as the
        panel drifting away from its corner during a live resize — moving
        at some multiple of the correct rate — until it slid off-screen.
        Here, expanding the algebra shows the corner is invariant no matter
        what sx/sz numerically are:

            right_edge = pos.x + right_local * sx
                       = ((1 - margin) - right_local * sx) + right_local * sx
                       = 1 - margin                                  (constant)

        so a wrong sx/sz can at most make the panel too big/small — it can
        no longer make it wander from the corner.
        """
        win = self.base.win
        if win is None or not win.hasSize():
            return
        w, h = win.getXSize(), win.getYSize()
        # Right after the window is created (before the window manager has
        # actually mapped/sized it), this can briefly be zero/invalid —
        # computing a dock position from that sends the panel off-screen,
        # and since nothing re-triggers _reposition() until the *next* real
        # resize, it can stay stranded there indefinitely if the user never
        # resizes the window. Refusing to move on a nonsensical size,
        # combined with refresh() re-calling this every time the panel is
        # actually shown (by which point the window has always finished
        # initializing), is what actually provides "wait until it's ready".
        if w <= 0 or h <= 0:
            return
        sx, sz = _SCALE * 2.0 / w, _SCALE * 2.0 / h
        left_local, right_local, top_local = _PANEL_L - 0.03, _PANEL_R + 0.03, 0.95
        x = (1 - _MARGIN_NDC) - right_local * sx
        # Narrow-window fallback: hug the left edge instead of leaving the
        # panel clipped off the right when it doesn't fit (e.g. a square or
        # portrait window) — same idea as before, adapted to this formula.
        x = max(x, (-1 + _MARGIN_NDC) - left_local * sx)
        z = (1 - _MARGIN_NDC) - top_local * sz
        self.frame.setPos(x, 0, z)
        self.frame.setScale(sx, 1, sz)

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

    # -- staging edits ------------------------------------------------------
    # Nothing below touches ``self.pipe`` — it only edits ``self._pending`` /
    # ``self._pending_profile`` and re-renders. The pipeline is only ever
    # touched by ``_commit``.
    def _stage(self, name: str, value: object) -> None:
        self._pending[name] = value
        self.refresh()

    def _stage_enum(self, name: str, allowed: list[str], value: object, delta: int) -> None:
        cur = str(value)
        idx = allowed.index(cur) if cur in allowed else 0
        idx = (idx + delta) % len(allowed)
        self._stage(name, allowed[idx])

    @property
    def profile_name(self) -> str:
        """The profile currently *applied* through this panel (e.g. after
        its own < > stepper + Apply) — lets an external caller (the demo's
        status HUD) stay in sync without duplicating the panel's own
        tracking. Does not reflect an unapplied pending profile pick."""
        return self._profile_name

    def set_profile_name(self, name: str) -> None:
        """Keep the panel's profile row in sync with an external switch
        (e.g. the demo's own [1][2][3] profile hotkeys), which applies the
        profile immediately and resets every option to its defaults — so
        any not-yet-applied edits made through this panel no longer apply
        to anything and are discarded rather than silently applied later
        on top of a baseline the user never chose."""
        self._profile_name = name
        self._pending_profile = None
        self._pending.clear()
        if not self.frame.is_hidden():
            self.refresh()
        else:
            self._sync_apply_bar()

    def _stage_profile(self, delta: int) -> None:
        if not self.profiles:
            return
        current = self._pending_profile if self._pending_profile is not None else self._profile_name
        idx = self.profiles.index(current) if current in self.profiles else 0
        idx = (idx + delta) % len(self.profiles)
        self._pending_profile = self.profiles[idx]
        self.refresh()

    # -- committing / discarding -------------------------------------------
    def _has_pending(self) -> bool:
        profile_changed = (
            self._pending_profile is not None and self._pending_profile != self._profile_name
        )
        return bool(self._pending) or profile_changed

    def _commit(self) -> None:
        if not self._has_pending():
            return
        profile_changed = self._pending_profile is not None and self._pending_profile != self._profile_name
        if profile_changed:
            self._profile_name = self._pending_profile
            self.pipe.apply_profile(self._profile_name)  # applies + recompiles
        for name, value in self._pending.items():
            self.pipe.set_option(name, value)
        if self._pending:
            self.pipe.recompile()
        self._pending.clear()
        self._pending_profile = None
        if self.on_change:
            self.on_change()
        self.refresh()

    def _revert(self) -> None:
        if not self._has_pending():
            return
        self._pending.clear()
        self._pending_profile = None
        self.refresh()

    def _sync_apply_bar(self) -> None:
        profile_changed = self._pending_profile is not None and self._pending_profile != self._profile_name
        total = len(self._pending) + (1 if profile_changed else 0)
        enabled = total > 0
        state = DGG.NORMAL if enabled else DGG.DISABLED
        self.apply_btn["state"] = state
        self.revert_btn["state"] = state
        self.apply_btn["frameColor"] = _ACCENT_COLOR if enabled else _BTN_COLOR
        if total == 0:
            self.pending_label["text"] = "no unapplied changes"
        else:
            self.pending_label["text"] = f"{total} unapplied change{'s' if total != 1 else ''}"

    # -- rendering --------------------------------------------------------
    def refresh(self) -> None:
        # Re-dock on every refresh (i.e. every show()/toggle()-open), not
        # just once at construction — by the time the panel is actually
        # shown the window is guaranteed to be fully initialized, so this
        # is what makes the panel self-correct even if the very first
        # _reposition() (in __init__) ran before the window was ready.
        self._reposition()

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
        self._sync_apply_bar()

    def _row_profile(self, y: float) -> float:
        current = self._pending_profile if self._pending_profile is not None else self._profile_name
        dirty = self._pending_profile is not None and self._pending_profile != self._profile_name
        label = f"profile: {current}" + (" *" if dirty else "")
        self._rows.append(_stepper_row(
            self.canvas, y, label,
            on_prev=lambda: self._stage_profile(-1),
            on_next=lambda: self._stage_profile(1),
            text_fg=_MODIFIED_COLOR if dirty else _TEXT_COLOR,
        ))
        return y - _ROW_H

    def _row_screen(self, el: dict, y: float) -> float:
        name = el["name"]
        self._rows.append(DirectButton(
            parent=self.canvas, text=f"{name}  >",
            text_scale=0.045, text_fg=(1, 1, 1, 1), text_align=TextNode.ALeft,
            frameColor=_SCREEN_COLOR, relief=DGG.FLAT,
            frameSize=(0, _ROW_W, -0.022, 0.032),
            pos=(_PANEL_L + 0.01, 0, y),
            command=self._enter, extraArgs=[name],
        ))
        return y - _ROW_H

    def _row_option(self, el: dict, y: float) -> float:
        name, kind, allowed = el["name"], el["kind"], el["allowed"]
        applied = el["value"]
        value = self._pending.get(name, applied)
        dirty = name in self._pending
        if kind == "toggle":
            return self._row_toggle(name, bool(value), dirty, y)
        numeric = _is_numeric(applied)
        if el.get("is_slider") and len(allowed) > 1:
            return self._row_slider(name, allowed, value, applied, y)
        return self._row_stepper_option(name, allowed, value, applied, dirty, numeric, y)

    def _row_toggle(self, name: str, value: bool, dirty: bool, y: float) -> float:
        mark = "[x]" if value else "[ ]"
        text = f"{mark} {name}" + ("  *" if dirty else "")
        # The whole row is the hit target (frameSize spans the full row
        # width, matching what's actually drawn) rather than a separate
        # tiny indicator box under oversized text, so the visible button
        # and the clickable area are the same rectangle.
        self._rows.append(DirectButton(
            parent=self.canvas, text=text, text_scale=0.045,
            text_fg=(1, 1, 1, 1), text_align=TextNode.ALeft,
            frameColor=_TOGGLE_ON_COLOR if value else _TOGGLE_OFF_COLOR,
            relief=DGG.FLAT, frameSize=(0, _ROW_W, -0.022, 0.032),
            pos=(_PANEL_L + 0.01, 0, y),
            command=self._stage, extraArgs=[name, not value],
        ))
        return y - _ROW_H

    def _row_stepper_option(self, name: str, allowed: list[str], value: object, applied: object,
                             dirty: bool, numeric: bool, y: float) -> float:
        if numeric:
            self._rows.append(_numeric_row(
                self.canvas, y, name, value, allowed, dirty=dirty,
                on_type=lambda text, name=name, applied=applied: self._on_number_entry(text, name, applied),
                on_prev=lambda: self._stage_enum(name, allowed, value, -1),
                on_next=lambda: self._stage_enum(name, allowed, value, 1),
            ))
            return y - _ROW_H
        label = name if len(allowed) <= 1 else f"{name}: {value}"
        if dirty:
            label += "  *"
        self._rows.append(_stepper_row(
            self.canvas, y, label,
            on_prev=lambda: self._stage_enum(name, allowed, value, -1),
            on_next=lambda: self._stage_enum(name, allowed, value, 1),
            disabled=len(allowed) <= 1,
            text_fg=_MODIFIED_COLOR if dirty else _TEXT_COLOR,
        ))
        return y - _ROW_H

    def _on_number_entry(self, text: str, name: str, applied: object) -> None:
        """Commit a typed number (Enter or focus-out) — free-form, not
        limited to the pack's own ``allowed`` step list, since typing an
        exact value is the whole point. Invalid text is silently discarded:
        refresh() redraws the entry with the last good value."""
        token = _reformat_number(text, applied)
        if token is not None and token != str(applied):
            self._pending[name] = token
        elif token is not None:
            self._pending.pop(name, None)
        self.refresh()

    def _row_slider(self, name: str, allowed: list[str], value: object,
                     applied: object, y: float) -> float:
        dirty = str(value) != str(applied)
        idx = allowed.index(str(value)) if str(value) in allowed else 0
        row_h = _ROW_H * 1.5
        color = _MODIFIED_COLOR if dirty else _TEXT_COLOR

        row = DirectFrame(parent=self.canvas, frameColor=(0, 0, 0, 0),
                           frameSize=(_PANEL_L, _PANEL_R, y - row_h + 0.03, y + 0.032),
                           pos=(0, 0, 0))
        DirectLabel(
            parent=row, text=name, text_scale=0.040, text_fg=color,
            text_align=TextNode.ALeft, frameColor=(0, 0, 0, 0),
            pos=(_PANEL_L + _ROW_INSET, 0, y - 0.012),
        )
        # The value itself is a plain typable number, not just a slider
        # readout — allows an exact value the allowed step list may skip.
        entry = DirectEntry(
            parent=row, initialText=str(value), width=6, numLines=1,
            scale=0.040, text_align=TextNode.ARight,
            frameColor=(0.14, 0.14, 0.16, 1), text_fg=color,
            pos=(_PANEL_R - 0.02, 0, y - 0.012),
        )
        track_w = (_PANEL_R - _ROW_INSET) - (_PANEL_L + _ROW_INSET)
        slider = DirectSlider(
            parent=row, range=(0, max(len(allowed) - 1, 1)), value=idx, pageSize=1,
            frameSize=(0, track_w, -0.010, 0.010), frameColor=(0.16, 0.16, 0.18, 1),
            thumb_frameSize=(-0.014, 0.014, -0.022, 0.022), thumb_frameColor=_BTN_COLOR,
            thumb_relief=DGG.FLAT,
            pos=(_PANEL_L + _ROW_INSET, 0, y - row_h + 0.052),
        )
        # Dragging fires the command many times per second — rebuilding the
        # whole row list (refresh()) on every tick would destroy this very
        # slider mid-drag, so only this row's own entry/pending state is
        # touched here; the apply bar is cheap to keep in sync directly.
        slider["command"] = self._on_slider_move
        slider["extraArgs"] = [slider, entry, name, allowed, applied]

        def commit_typed(text, slider=slider, name=name, allowed=allowed, applied=applied):
            self._on_slider_entry(text, slider, name, allowed, applied)

        entry["command"] = commit_typed
        entry["focusOutCommand"] = lambda entry=entry, commit=commit_typed: commit(entry.get())

        self._rows.append(row)
        return y - row_h

    def _on_slider_move(self, slider, entry, name: str, allowed: list[str], applied: object) -> None:
        idx = int(round(slider["value"]))
        idx = max(0, min(len(allowed) - 1, idx))
        if slider["value"] != idx:
            slider["value"] = idx  # snap to the nearest allowed step
        value = allowed[idx]
        if value == str(applied):
            self._pending.pop(name, None)
        else:
            self._pending[name] = value
        dirty = name in self._pending
        entry.set(str(value))
        entry["text_fg"] = _MODIFIED_COLOR if dirty else _TEXT_COLOR
        self._sync_apply_bar()

    def _on_slider_entry(self, text: str, slider, name: str, allowed: list[str], applied: object) -> None:
        """A number typed directly into a slider row's entry — free-form,
        like _on_number_entry, but also snaps the slider thumb to whichever
        allowed step is closest, for visual feedback."""
        token = _reformat_number(text, applied)
        if token is None:
            self.refresh()  # bad input — snap back to the last good value
            return
        if token != str(applied):
            self._pending[name] = token
        else:
            self._pending.pop(name, None)
        nearest = min(range(len(allowed)), key=lambda i: abs(_safe_float(allowed[i]) - _safe_float(token)))
        slider["value"] = nearest
        self.refresh()


def _stepper_row(parent, y, label, *, on_prev, on_next, disabled=False, text_fg=_TEXT_COLOR):
    row = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0),
                       frameSize=(_PANEL_L, _PANEL_R, -0.022, 0.032), pos=(0, 0, y))
    DirectLabel(parent=row, text=label, text_scale=0.040, text_fg=text_fg,
                text_align=TextNode.ALeft, frameColor=(0, 0, 0, 0),
                pos=(_PANEL_L + _ROW_INSET, 0, -0.012))
    if not disabled:
        DirectButton(parent=row, text="<", text_scale=0.04, frameColor=_BTN_COLOR,
                     frameSize=(-0.045, 0.045, -0.028, 0.038),
                     pos=(_PANEL_R - 0.10, 0, -0.008), command=on_prev)
        DirectButton(parent=row, text=">", text_scale=0.04, frameColor=_BTN_COLOR,
                     frameSize=(-0.045, 0.045, -0.028, 0.038),
                     pos=(_PANEL_R - 0.03, 0, -0.008), command=on_next)
    return row


def _numeric_row(parent, y, name, value, allowed, *, dirty, on_type, on_prev, on_next):
    """Like _stepper_row, but the value is a typable DirectEntry instead of
    a plain label — for options whose value is just a number, typing an
    exact figure is more useful than clicking through the pack's own
    (often coarse) allowed-value list."""
    color = _MODIFIED_COLOR if dirty else _TEXT_COLOR
    row = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0),
                       frameSize=(_PANEL_L, _PANEL_R, -0.022, 0.032), pos=(0, 0, y))
    DirectLabel(parent=row, text=name, text_scale=0.040, text_fg=color,
                text_align=TextNode.ALeft, frameColor=(0, 0, 0, 0),
                pos=(_PANEL_L + _ROW_INSET, 0, -0.012))
    stepper = len(allowed) > 1
    entry = DirectEntry(
        parent=row, initialText=str(value), width=6, numLines=1,
        scale=0.040, text_align=TextNode.ARight,
        frameColor=(0.14, 0.14, 0.16, 1), text_fg=color,
        pos=(_PANEL_R - (0.20 if stepper else 0.02), 0, -0.012),
        command=on_type,
    )
    entry["focusOutCommand"] = lambda entry=entry, cmd=on_type: cmd(entry.get())
    if stepper:
        DirectButton(parent=row, text="<", text_scale=0.04, frameColor=_BTN_COLOR,
                     frameSize=(-0.045, 0.045, -0.028, 0.038),
                     pos=(_PANEL_R - 0.10, 0, -0.008), command=on_prev)
        DirectButton(parent=row, text=">", text_scale=0.04, frameColor=_BTN_COLOR,
                     frameSize=(-0.045, 0.045, -0.028, 0.038),
                     pos=(_PANEL_R - 0.03, 0, -0.008), command=on_next)
    return row


def _is_numeric(value: object) -> bool:
    try:
        float(str(value))
        return True
    except ValueError:
        return False


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _reformat_number(text: str, applied: object) -> str | None:
    """Parse a user-typed number, formatted back with the same decimal
    precision as the option's currently-applied token (so typing "1.5" for
    an option authored as "1.00" stages "1.50", matching the pack's own
    formatting) — or None if the text isn't a number at all."""
    text = text.strip()
    try:
        parsed = float(text)
    except ValueError:
        return None
    applied_s = str(applied)
    if "." in applied_s:
        decimals = len(applied_s.split(".", 1)[1])
        return f"{parsed:.{decimals}f}"
    return str(int(round(parsed)))
