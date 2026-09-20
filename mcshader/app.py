"""The one-call front door: a shaded Panda3D app in four lines.

``PipelineRenderer`` (see :mod:`mcshader.engine.panda3d_pipeline`) is the real
machinery, but using it means writing the Panda3D boilerplate around it every
time — make a ``ShowBase``, pose the camera, load and reparent and scale each
model, tag it, build a sky, remember to re-apply block ids after every
recompile. This module does all of that for you:

    import mcshader

    app = mcshader.init()                       # window + pack + sky + fly cam
    app.load("models/environment", type="terrain", scale=0.25, pos=(-8, 42, 0))
    app.load_actor("models/panda-model", {"walk": "models/panda-walk4"},
                   type="entity", scale=0.005, pos=(0, 20, 0), loop="walk")
    app.run()

Already have a Panda3D app? Hand it over and keep your own scene graph —
nothing else changes:

    app = mcshader.init(base)
    app.attach(my_terrain, type="terrain")

Everything the runner can do is reachable from here (``app.profile = "ULTRA"``,
``app.option("SHADOW", False)``, ``app.swap_pack(...)``, ``app.view("colortex1")``),
and :meth:`ShaderApp.debug_ui` installs the whole hands-on dev HUD — profile
hotkeys, the pack's live settings panel, the raw-buffer viewer — in one call.
The underlying objects stay public as ``app.pipe`` and ``app.base``, so
dropping down to the full API (or to raw Panda3D) is never a dead end.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Sequence

__all__ = ["ShaderApp", "init", "find_pack"]

#: Where :func:`find_pack` looks when you don't name a pack, in order.
_PACK_SEARCH = ("shaderpacks", "Shaders", "shaders")
#: How many parent directories above the cwd that search also covers.
_PACK_SEARCH_UP = 3

#: Canonical profile order, used to order a pack's own profile names
#: (and as the fallback for a pack that declares none).
_PROFILES = ["MINIMUM", "LOW", "MEDIUM", "HIGH", "ULTRA"]

_UNSET = object()  # "no value passed", distinct from a real None/False


def find_pack(path: str | None = None) -> str:
    """Resolve a shaderpack path, searching the usual places if given none.

    Order: the argument, ``$MCSHADER_PACK``, then a ``shaderpacks/``,
    ``Shaders/`` or ``shaders/`` directory at (or above) the working
    directory — taking a directory pack as-is, else its first ``.zip``.
    Walking up a few levels means a script run from a subdirectory
    (``python examples/foo.py`` from ``examples/``) still finds the project's
    packs, which is where they live in practice.
    """
    if path:
        return path
    env = os.environ.get("MCSHADER_PACK")
    if env:
        return env
    roots = [os.path.join(*([".."] * i)) if i else "." for i in range(_PACK_SEARCH_UP + 1)]
    for root in roots:
        for name in (os.path.join(root, n) for n in _PACK_SEARCH):
            if not os.path.isdir(name):
                continue
            # A pack directory proper (it has the shader programs in it), or
            # a folder of packs (zips / unpacked pack dirs) — take the first.
            entries = sorted(os.listdir(name))
            if any(e in ("shaders", "shaders.properties") for e in entries):
                return name
            for entry in entries:
                full = os.path.join(name, entry)
                if entry.lower().endswith(".zip") or os.path.isdir(full):
                    return full
    raise FileNotFoundError(
        "no shaderpack found — pass one explicitly, e.g. "
        "mcshader.init(pack='Shaders/BSL_v10.0.zip'), or set $MCSHADER_PACK, "
        f"or put a pack in one of: {', '.join(_PACK_SEARCH)}/"
    )


class _Tagged:
    """What we know about one tagged node, so a rebuild can restore it.

    ``PipelineRenderer.recompile`` / ``swap_pack`` replay render-type tags
    themselves, but block ids and lightmaps are plain shader inputs that a
    rebuild drops on the floor (the demo had to re-apply them by hand after
    every profile switch). Tracking them here is what makes
    ``app.profile = "ULTRA"`` a one-liner that doesn't quietly reset the
    scene's materials.
    """

    __slots__ = ("np", "type", "block", "light", "normal", "specular")

    def __init__(self, np: Any, render_type: str | None,
                 block: int | str | None, light: tuple[float, float] | None,
                 normal: Any = None, specular: Any = None):
        self.np = np
        self.type = render_type
        self.block = block
        self.light = light
        self.normal = normal
        self.specular = specular


class ShaderApp:
    """A Panda3D app with a Minecraft shaderpack already running over it.

    Construct via :func:`mcshader.init`. ``app.base`` is the ``ShowBase`` and
    ``app.pipe`` the :class:`~mcshader.engine.panda3d_pipeline.PipelineRenderer`
    — both public on purpose, for anything this facade doesn't wrap.
    """

    def __init__(self, pack: str | None = None, *, base: Any = None,
                 profile: str | None = "LOW", world: str = "world0",
                 sky: bool = True, fly: bool = True, speed: float = 40.0,
                 title: str = "mcshader", size: tuple[int, int] | None = None):
        self.pack_path = find_pack(pack)
        self.base = base if base is not None else _make_showbase(title, size)
        self._owns_base = base is None
        self._tagged: list[_Tagged] = []
        self._settings = None
        self._debug: dict[str, Any] = {}
        self._home: tuple[Any, Any] | None = None

        from .engine import PipelineRenderer

        self.pipe = PipelineRenderer(
            self.base, self.pack_path, world=world, profile=profile)
        #: The procedural sky dome (gradient, sun, moon, stars), or ``None``.
        self.sky = self.pipe.build_sky() if sky else None
        if fly:
            self.fly_camera(speed=speed)

    # -- scene ----------------------------------------------------------
    def load(self, model: str, *, type: str | None = None,
             tag: Sequence[tuple[str, str]] | None = None,
             default_type: str | None = None,
             pos: Sequence[float] | None = None,
             hpr: Sequence[float] | None = None,
             scale: float | Sequence[float] | None = None,
             block: int | str | None = None,
             light: tuple[float, float] | None = None,
             color: Sequence[float] | None = None,
             parent: Any = None) -> Any:
        """Load a model, place it, and shade it as a Minecraft render type.

        ``type`` tags the whole model (e.g. ``"terrain"``, ``"water"``,
        ``"entity"``); ``tag`` instead takes ordered ``(name_regex,
        render_type)`` rules applied per sub-part, the way a real scene gets
        tagged — by naming convention — with ``default_type`` catching
        whatever matches nothing. ``block`` may be a Minecraft id
        (``"minecraft:sea_lantern"``) or a raw numeric id; ``light`` is
        ``(block_light, sky_light)`` in 0..1 (see
        :meth:`~mcshader.engine.panda3d_pipeline.PipelineRenderer.set_lightmap`
        — the default assumes outdoors).

        Returns the ``NodePath``.
        """
        np = self.base.loader.load_model(model)
        np.reparent_to(parent if parent is not None else self.base.render)
        self._place(np, pos, hpr, scale, color)
        if tag:
            self.tag_by_pattern(np, tag, default=default_type)
            if block is not None or light is not None:
                # Shader inputs inherit down the graph, so setting these on
                # the root covers every sub-part the rules just tagged.
                self.attach(np, block=block, light=light)
        elif type or default_type:
            self.attach(np, type=type or default_type, block=block, light=light)
        elif block is not None or light is not None:
            self.attach(np, type=None, block=block, light=light)
        return np

    def load_actor(self, model: str, anims: dict[str, str] | None = None, *,
                   loop: str | None = None, type: str | None = None,
                   pos: Sequence[float] | None = None,
                   hpr: Sequence[float] | None = None,
                   scale: float | Sequence[float] | None = None,
                   block: int | str | None = None,
                   light: tuple[float, float] | None = None,
                   color: Sequence[float] | None = None,
                   parent: Any = None) -> Any:
        """Load an animated ``Actor`` and shade it. ``loop`` starts an animation.

        Same tagging/placement arguments as :meth:`load`; proves the pipeline
        shades skinned, moving geometry and not just static props.
        """
        from direct.actor.Actor import Actor

        actor = Actor(model, anims or {})
        actor.reparent_to(parent if parent is not None else self.base.render)
        self._place(actor, pos, hpr, scale, color)
        if loop:
            actor.loop(loop)
        if type or block is not None or light is not None:
            self.attach(actor, type=type, block=block, light=light)
        return actor

    def attach(self, nodepath: Any, *, type: str | None = None,
               block: int | str | None = None,
               light: tuple[float, float] | None = None,
               normal: Any = None, specular: Any = None) -> Any:
        """Shade a node you made yourself (or retag one you already gave us).

        Re-applied for you after every recompile, profile switch and pack
        swap — including ``block``, ``light`` and the PBR maps, all of which
        the raw runner drops on a rebuild.

        ``normal`` and ``specular`` are labPBR maps for this node; see
        :meth:`~mcshader.engine.panda3d_pipeline.PipelineRenderer.set_material_maps`
        for the channel layout and what else has to be true for them to be read.
        """
        entry = next((t for t in self._tagged if t.np == nodepath), None)
        if entry is None:
            entry = _Tagged(nodepath, None, None, None)
            self._tagged.append(entry)
        if type is not None:
            entry.type = type
        if block is not None:
            entry.block = block
        if light is not None:
            entry.light = light
        if normal is not None:
            entry.normal = normal
        if specular is not None:
            entry.specular = specular
        self._apply(entry)
        return nodepath

    #: ``attach`` reads better for a new node, ``set_type``/``set_block``/
    #: ``set_light`` for changing one later; they're the same call.
    def set_type(self, nodepath: Any, render_type: str) -> Any:
        return self.attach(nodepath, type=render_type)

    def set_block(self, nodepath: Any, block: int | str) -> Any:
        return self.attach(nodepath, block=block)

    def set_light(self, nodepath: Any, block_light: float, sky_light: float) -> Any:
        return self.attach(nodepath, light=(block_light, sky_light))

    def set_material_maps(self, nodepath: Any, normal: Any = None,
                          specular: Any = None) -> Any:
        """Give one node its own labPBR normal/specular maps."""
        return self.attach(nodepath, normal=normal, specular=specular)

    def set_eye_in_water(self, state: int) -> None:
        """Tell the pack the camera is under water (1), in lava (2) or in air (0).

        Drives every submerged effect the pack has -- water fog, underwater distortion,
        light shafts through water, and which side of the surface its fresnel is computed
        for. An engine has to call this itself; unfed, the pack renders as though the
        camera were never in water.
        """
        self.pipe.set_eye_in_water(state)

    def set_eye_brightness(self, block_light: float, sky_light: float,
                           *, immediate: bool = False) -> None:
        """How lit the place the camera is standing in is, as (block, sky) in 0..1.

        The pack's ``eyeBrightnessSmooth``. Distinct from :meth:`set_light`, which
        describes a *surface*: this describes the viewer, and the pack reads it to
        decide how much daylight reaches the air around them -- the fog's density
        and colour, and the tint of water fog seen from under the surface. Left
        unfed the camera is treated as standing outdoors under open sky, because
        the alternative default (zero) reads as "sealed in a cave" and quietly
        turns those effects black. Pass ``immediate=True`` after a teleport.
        """
        self.pipe.set_eye_brightness(block_light, sky_light, immediate=immediate)

    def tag_by_pattern(self, root: Any, rules: Sequence[tuple[str, str]], *,
                       default: str | None = None) -> dict[str, int]:
        """Tag every sub-part of ``root`` by the first matching name regex.

        Returns a ``{render_type: count}`` summary.
        """
        from .ui.tagging import tag_by_pattern

        counts = tag_by_pattern(_TaggerProxy(self), root, rules, default=default)
        return counts

    def remove(self, nodepath: Any) -> None:
        """Stop shading a node (it falls back to Panda3D's own rendering)."""
        self.pipe.clear(nodepath)
        self._tagged = [t for t in self._tagged if t.np != nodepath]

    def block_id(self, mc_id: str) -> int:
        """Numeric block id for a Minecraft id, per the pack's block.properties."""
        return self.pipe.resolver.block_id(mc_id)

    @property
    def render_types(self) -> list[str]:
        """Render types this pack actually has a program for."""
        return self.pipe.resolver.types()

    def _place(self, np: Any, pos, hpr, scale, color) -> None:
        if pos is not None:
            np.set_pos(*pos)
        if hpr is not None:
            np.set_hpr(*hpr) if isinstance(hpr, (list, tuple)) else np.set_h(hpr)
        if scale is not None:
            np.set_scale(*scale) if isinstance(scale, (list, tuple)) else np.set_scale(scale)
        if color is not None:
            np.set_color(*color)

    def _apply(self, entry: _Tagged) -> None:
        if entry.type:
            self.pipe.set_render_type(entry.np, entry.type)
        if entry.block is not None:
            block = entry.block
            self.pipe.set_block_id(
                entry.np, self.block_id(block) if isinstance(block, str) else int(block))
        if entry.light is not None:
            self.pipe.set_lightmap(entry.np, *entry.light)
        if entry.normal is not None or entry.specular is not None:
            self.pipe.set_material_maps(entry.np, entry.normal, entry.specular)

    # -- camera ---------------------------------------------------------
    def camera(self, *, pos: Sequence[float] | None = None,
               look_at: Sequence[float] | None = None,
               hpr: Sequence[float] | None = None,
               fov: float | None = None, far: float | None = None, near: float | None = None) -> Any:
        """Pose the camera (and remember the pose as ``debug_ui``'s [g] home).

        Moves ``base.cam``, not ``base.camera`` — that's the node the
        pipeline derives its view matrices from.
        """
        cam = self.base.cam
        if pos is not None:
            cam.set_pos(*pos)
        if look_at is not None:
            cam.look_at(*look_at)
        if hpr is not None:
            cam.set_hpr(*hpr)
        if fov is not None:
            self.base.camLens.set_fov(fov)
        if far is not None:
            self.base.camLens.set_far(far)
        if near is not None:
            self.base.camLens.set_near(near)
        self._home = (cam.get_pos(), cam.get_hpr())
        return cam

    def fly_camera(self, *, speed: float = 40.0, turn_speed: float = 90.0) -> None:
        """WASD/QE to fly, arrows to look, shift to boost."""
        from .ui.flycam import attach_fly_camera

        attach_fly_camera(self.base, speed=speed, turn_speed=turn_speed)

    def go_home(self) -> None:
        """Return the camera to the last pose set through :meth:`camera`."""
        if self._home is None:
            return
        pos, hpr = self._home
        self.base.cam.set_pos(pos)
        self.base.cam.set_hpr(hpr)

    # -- pack / options -------------------------------------------------
    @property
    def profile(self) -> str | None:
        """The active quality profile; assign to switch (``app.profile = "ULTRA"``)."""
        return self.pipe.profile_name

    @profile.setter
    def profile(self, name: str) -> None:
        self.pipe.apply_profile(name)
        self._after_rebuild()

    @property
    def profiles(self) -> list[str]:
        """Profiles this pack declares (falling back to the standard five)."""
        declared = list(getattr(self.pipe.props, "profiles", {}) or {})
        ordered = ([p for p in _PROFILES if p in declared]
                   + [p for p in declared if p not in _PROFILES])
        return ordered or list(_PROFILES)

    def option(self, name: str, value: object = _UNSET, *,
               apply: bool = True) -> object:
        """Read an option (``app.option("SHADOW")``) or set one (pass a value).

        Setting recompiles by default — pass ``apply=False`` to batch several
        edits and call :meth:`reload` once.
        """
        if value is _UNSET:
            return self.pipe.options.get(name)
        if name not in self.pipe.options.options:
            raise KeyError(
                f"pack {self.pipe.pack.name!r} has no option {name!r}; "
                f"see app.options or app.describe()")
        self.pipe.set_option(name, value)
        if apply:
            self.reload()
        return value

    @property
    def options(self) -> dict[str, object]:
        """Every option this pack exposes, with its current value."""
        return dict(self.pipe.options.values())

    def reload(self) -> None:
        """Recompile with the current option values (keeps every tag)."""
        self.pipe.recompile()
        self._after_rebuild()

    def swap_pack(self, pack: str | None = None) -> None:
        """Re-shade the whole scene with another pack — tags and profile survive."""
        self.pack_path = find_pack(pack) if pack else self.pack_path
        self.pipe.swap_pack(self.pack_path)
        self._after_rebuild()

    @property
    def enabled(self) -> bool:
        """``False`` renders the same scene with Panda3D's plain shading, to A/B against."""
        return self.pipe._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.pipe.set_enabled(bool(value))
        self._refresh_status()

    def describe(self) -> str:
        """A text summary of the resolved pipeline — buffers, pass order, types."""
        return self.pipe.describe()

    def _after_rebuild(self) -> None:
        """Restore what a rebuild dropped, then resync anything showing it."""
        for entry in self._tagged:
            if entry.block is not None or entry.light is not None:
                self._apply(entry)
        if self._debug:
            self._debug["names"] = list(self.pipe.debug_textures())
            if self._debug.get("index", -1) >= 0:
                self.view(self._debug["index"])  # re-bind: buffers are new objects
        if self._settings is not None and self.profile:
            self._settings.set_profile_name(self.profile)
        self._refresh_status()

    # -- inspection / UI ------------------------------------------------
    @property
    def buffers(self) -> list[str]:
        """Names of the raw intermediate buffers :meth:`view` can display."""
        return list(self.pipe.debug_textures())

    def view(self, buffer: str | int | None = None) -> None:
        """Show one raw ``colortex`` buffer fullscreen; ``None`` restores the
        normal composited frame. The fastest way to see what a pass wrote."""
        quad = self._debug.get("quad")
        if quad is None:
            quad = self._debug["quad"] = self._build_debug_quad()
            self._debug["names"] = list(self.pipe.debug_textures())
        names = self._debug["names"]
        if buffer is None:
            self._debug["index"] = -1
            quad.hide()
        else:
            index = buffer if isinstance(buffer, int) else names.index(buffer)
            index %= len(names)
            self._debug["index"] = index
            quad.set_texture(self.pipe.debug_textures()[names[index]], 1)
            quad.show()
        self._refresh_status()

    def step_view(self, direction: int = 1) -> None:
        """Walk the buffer list with :meth:`view` (``debug_ui``'s ``[`` / ``]``)."""
        names = self._debug.get("names") or self.buffers
        if not names:
            return
        index = self._debug.get("index", -1)
        self.view(0 if index < 0 and direction > 0 else
                  len(names) - 1 if index < 0 else index + direction)

    def _build_debug_quad(self) -> Any:
        from panda3d.core import CardMaker

        cm = CardMaker("mcshader-debug-quad")
        cm.set_frame_fullscreen_quad()
        quad = self.base.render2d.attach_new_node(cm.generate())
        quad.set_bin("fixed", 1000)  # over the pipeline's own final quad
        quad.set_depth_test(False)
        quad.set_depth_write(False)
        quad.hide()
        return quad

    @property
    def settings(self) -> Any:
        """The pack's own Iris-style settings menu, live (built on first use)."""
        if self._settings is None:
            from .ui.settings_panel import SettingsPanel

            if getattr(self.base, "mouseWatcherNode", None) is None:
                raise RuntimeError(
                    "the settings panel needs a real window (this app has no "
                    "mouse watcher — offscreen/headless ShowBase)")
            self._settings = SettingsPanel(
                self.base, self.pipe, profiles=self.profiles,
                profile_name=self.profile or "", on_change=self._after_rebuild)
        return self._settings

    def text(self, message: str, *, pos: tuple[float, float] = (-1.3, 0.95),
             scale: float = 0.045, color: tuple = (1, 1, 1, 1),
             changeable: bool = False) -> Any:
        """A line (or block) of screen text — a thin ``OnscreenText`` wrapper."""
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        return OnscreenText(
            text=message, pos=pos, scale=scale, align=TextNode.ALeft,
            fg=color, shadow=(0, 0, 0, 0.6), mayChange=changeable)

    def key(self, key: str, callback: Callable, *args) -> None:
        """Bind a key (``app.key("t", toggle_thing)``)."""
        self.base.accept(key, callback, list(args))

    def every_frame(self, callback: Callable[[float], Any], *,
                    name: str | None = None) -> None:
        """Run ``callback(dt)`` every frame — a task without the Task boilerplate."""
        from direct.task import Task
        from panda3d.core import ClockObject

        clock = ClockObject.get_global_clock()

        def task(_task):
            callback(clock.get_dt())
            return Task.cont

        self.base.taskMgr.add(task, name or f"mcshader-user-{id(callback)}")

    def screenshot(self, prefix: str = "mcshader") -> str:
        """Save a PNG of the window; returns the path."""
        return self.base.screenshot(namePrefix=prefix)

    def run(self) -> None:
        """Open the window and run (blocks, like Panda3D's own ``base.run()``)."""
        self.base.run()

    # -- the hands-on dev HUD -------------------------------------------
    _HELP = """mcshader
WASD/QE fly, arrows look, shift boost   [g] home camera
[1]-[5] profile {profiles}
[o] live settings panel (every option the pack exposes)
[y] pack on/off (A/B against plain Panda3D)   [z] toggle SHADOW
[[ ]] step raw buffers, [0] back to final
[r] cycle render type   [b] cycle block id     (focus object)
[p] reload pack   [k] screenshot   [h] hide this help"""

    def debug_ui(self, *, focus: Any = None,
                 blocks: Iterable[str] = ("minecraft:sea_lantern",
                                          "minecraft:oak_leaves",
                                          "minecraft:stone"),
                 help: bool = True, extra: str = "") -> None:
        """Install the whole hands-on dev HUD in one call.

        Profile hotkeys, the live settings panel, the pack on/off A/B, the
        raw-buffer viewer, pack reload, screenshots, and a status line. Pass
        ``focus`` to also get [r]/[b] cycling that object's render type and
        block id — the quickest way to see what a pack does differently to,
        say, a sea lantern versus stone. ``extra`` appends your own lines to
        the on-screen help, for keys you bind yourself.
        """
        self._debug.setdefault("index", -1)
        self._debug["names"] = list(self.pipe.debug_textures())
        self._debug["focus"] = focus
        self._debug["blocks"] = list(blocks)
        self._debug["block_index"] = 0
        types = self.render_types
        self._debug["types"] = types
        if focus is not None:
            entry = next((t for t in self._tagged if t.np == focus), None)
            current = entry.type if entry and entry.type else None
            self._debug["type_index"] = types.index(current) if current in types else 0

        if help:
            text = self._HELP.format(profiles="/".join(self.profiles[:5]))
            self._debug["help"] = self.text(
                text + ("\n" + extra if extra else ""))
        self._debug["status"] = self.text(
            "", pos=(-1.3, -0.92), scale=0.05, color=(1, 1, 0.5, 1), changeable=True)

        for i, name in enumerate(self.profiles[:5], start=1):
            self.key(str(i), self._set_profile, name)
        # Bound through a thunk, not `self.settings.toggle`: touching the
        # property here would build the DirectGUI panel up front, which
        # an offscreen/headless app can't do at all.
        self.key("o", self._toggle_settings)
        self.key("y", self._toggle_pack)
        self.key("z", self._toggle_shadow)
        self.key("[", self.step_view, -1)
        self.key("]", self.step_view, 1)
        self.key("0", self.view, None)
        self.key("p", self.swap_pack)
        self.key("k", self._shot)
        self.key("h", self._toggle_help)
        self.key("g", self.go_home)
        if focus is not None:
            self.key("r", self._cycle_type)
            self.key("b", self._cycle_block)
        self._refresh_status()

    def _toggle_settings(self) -> None:
        self.settings.toggle()

    def _set_profile(self, name: str) -> None:
        self.profile = name
        print("profile ->", name)

    def _toggle_pack(self) -> None:
        self.enabled = not self.enabled
        print("shader pack ->", "on" if self.enabled else "off (plain Panda3D)")

    def _toggle_shadow(self) -> None:
        # "SHADOW" is BSL's own toggle name, not a universal one — Complementary
        # Unbound gates shadows on the numeric SHADOW_QUALITY instead.
        if "SHADOW" not in self.pipe.options.options:
            print("this pack has no 'SHADOW' option (press [o] for its real options)")
            return
        print("SHADOW ->", self.option("SHADOW", not self.option("SHADOW")))

    def _cycle_type(self) -> None:
        types = self._debug["types"]
        self._debug["type_index"] = (self._debug["type_index"] + 1) % len(types)
        self.set_type(self._debug["focus"], types[self._debug["type_index"]])
        self._refresh_status()
        print("focus render_type ->", types[self._debug["type_index"]])

    def _cycle_block(self) -> None:
        blocks = self._debug["blocks"]
        self._debug["block_index"] = (self._debug["block_index"] + 1) % len(blocks)
        self.set_block(self._debug["focus"], blocks[self._debug["block_index"]])
        self._refresh_status()
        print("focus block ->", blocks[self._debug["block_index"]])

    def _shot(self) -> None:
        print("saved", self.screenshot())

    def _toggle_help(self) -> None:
        text = self._debug.get("help")
        if text is not None:
            (text.show if text.is_hidden() else text.hide)()

    def _refresh_status(self) -> None:
        status = self._debug.get("status")
        if status is None:
            return
        index = self._debug.get("index", -1)
        names = self._debug.get("names") or []
        view = names[index] if 0 <= index < len(names) else "final"
        shadow = (self.option("SHADOW")
                  if "SHADOW" in self.pipe.options.options else "n/a")
        focus = ""
        if self._debug.get("focus") is not None:
            focus = (f"  focus={self._debug['types'][self._debug['type_index']]}"
                     f"/{self._debug['blocks'][self._debug['block_index']]}")
        # `setText`, not `set_text`: OnscreenText is a Python class in
        # direct.gui, so it has none of Panda's C++ snake_case aliases.
        status.setText(
            f"pack={'on' if self.enabled else 'OFF'}  profile={self.profile}  "
            f"shadow={shadow}{focus}  view={view}")


class _TaggerProxy:
    """Adapts :func:`mcshader.ui.tagging.tag_by_pattern` (which speaks the raw
    runner's ``set_render_type``) onto the app, so bulk-tagged sub-parts get
    tracked for rebuild-restore like every other tagged node."""

    def __init__(self, app: ShaderApp):
        self._app = app

    def set_render_type(self, nodepath: Any, render_type: str) -> None:
        self._app.attach(nodepath, type=render_type)


def _make_showbase(title: str, size: tuple[int, int] | None) -> Any:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", f"window-title {title}")
    if size:
        loadPrcFileData("", f"win-size {size[0]} {size[1]}")
    return ShowBase()


def init(base: Any = None, pack: str | None = None, **kwargs: Any) -> ShaderApp:
    """Start a shaded app (making the Panda3D window if you don't pass one).

        app = mcshader.init()                   # new window, pack auto-found
        app = mcshader.init(base)               # over your existing ShowBase
        app = mcshader.init(pack="X.zip", profile="ULTRA", sky=False, fly=False)

    ``base`` is a ``ShowBase``; ``pack`` a pack directory or ``.zip`` (see
    :func:`find_pack` for where it looks by default). Other keywords go to
    :class:`ShaderApp`: ``profile``, ``world``, ``sky``, ``fly``, ``speed``,
    ``title``, ``size``.
    """
    # Tolerate init("Shaders/") — the first argument reads as the pack when
    # it's a string, since an engine handle never is one.
    if isinstance(base, str) and pack is None:
        base, pack = None, base
    return ShaderApp(pack, base=base, **kwargs)
