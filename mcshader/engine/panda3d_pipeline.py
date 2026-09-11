"""Run a shaderpack's whole deferred pipeline inside Panda3D.

This is the headline feature: load a pack, tag your scene objects with Minecraft
*render types*, and the pack shades the whole frame — shadow map, gbuffers,
deferred lighting, composite chain, final — exactly as it declares. Swap the pack
and everything re-shades; drive features through the Iris-style options system.

    pipe = PipelineRenderer(base, "Shaders/", world="world0", profile="HIGH")
    pipe.set_render_type(terrain_np, "terrain")
    pipe.set_render_type(water_np, "water")
    pipe.set_block_id(custom_crystal, pipe.resolver.block_id("minecraft:sea_lantern"))
    pipe.set_option("SHADOW_FILTER", True); pipe.recompile()
    pipe.apply_profile("ULTRA")
    pipe.swap_pack("OtherPack.zip")

Convention (kept uniform across every pass): framebuffer color attachment ``i``
is ``colortex i``, so a translated shader writing ``layout(location=i)`` lands in
``colortex i``. See the module notes for the honest limitations.

.. note::
   Panda3D is imported lazily, and the GPU wiring in this module needs on-hardware
   validation (a machine with a display) — the rest of ``mcshader`` (parsing,
   options, graph, translation) is fully unit-tested without a GL context. Use
   :meth:`describe` to inspect the resolved pipeline headlessly.
"""

from __future__ import annotations

import re
from typing import Any

from ..pack.loader import ShaderPack
from ..config.options import ShaderOptions
from ..pipeline.graph import PipelineGraph, build_graph
from ..pipeline.rendertypes import RenderTypeResolver
from ..pipeline.translate import translate_pass

__all__ = ["PipelineRenderer"]

# Minecraft internal format name -> Panda3D Texture format enum name.
_MC_FORMATS = {
    "RGBA8": "F_rgba8", "RGB8": "F_rgb8", "RG8": "F_rg", "R8": "F_red",
    "RGBA16": "F_rgba16", "RGB16": "F_rgb16",
    "RGBA16F": "F_rgba16", "RGB16F": "F_rgb16", "RG16F": "F_rg16",
    "RGBA32F": "F_rgba32", "RGB32F": "F_rgb32", "R32F": "F_r32",
    "R11F_G11F_B10F": "F_r11_g11_b10", "RGB10_A2": "F_rgb10_a2",
}


class PipelineRenderer:
    def __init__(self, base: Any, pack_path: str, *, world: str = "world0",
                 profile: str | None = None, target: str = "panda3d"):
        # Panda3D pads every render-to-texture target up to the next power of
        # two by default (e.g. a 960x540 window -> a 1024x1024 texture, with
        # only the [0,0]-[960,540] corner actually rendered into). Our
        # translated shaders sample colortex/shadowtex with raw [0,1] UV —
        # there's no Panda auto-texture-matrix correcting for the padding, so
        # every pass that reads another pass's output would sample a
        # squashed/offset copy of it, compounding across the composite chain
        # into exactly the "nested rectangle" corruption this was causing.
        # Modern GL (this targets 3.3+ core) has no power-of-two restriction,
        # so just turn the padding off; must happen before any buffer/texture
        # in this pipeline is created, and is a global, idempotent setting.
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "textures-power-2 none")

        self.base = base
        self.world = world
        self.target = target
        self._geometry: list[tuple[Any, str]] = []  # (nodepath, render_type)
        self._quads: list[Any] = []
        self._final_quads: list[Any] = []  # subset of _quads that cover the window
        self._colortex: dict[int, Any] = {}
        self._colortex_back: dict[int, Any] = {}
        self._task_started = False
        self._enabled = True  # see set_enabled() -- pack-shaded vs. plain rendering
        self.profile_name: str | None = None
        self.load_pack(pack_path, world=world, profile=profile)

    # -- pack / options lifecycle ---------------------------------------
    def load_pack(self, pack_path: str, *, world: str | None = None,
                  profile: str | None = None) -> None:
        """Load (or reload) a shaderpack and rebuild everything."""
        self.pack = ShaderPack.from_path(pack_path)
        self.world = world or self.world
        # Bootstrap parse: no option values exist yet, so any `#if` block in
        # shaders.properties resolves with unknown identifiers defaulting
        # falsy/0 (see pack.properties.parse_properties) — fine here, this
        # first Properties is only used to discover option/profile/menu
        # names, never `program.*.enabled` (build_graph re-parses with real
        # values below, before it's ever consulted).
        self.props = self.pack.properties()
        self.options = ShaderOptions.from_pack(self.pack.option_sources(), self.props)
        if profile:
            self.options.apply_profile(profile, self.props)
        self.profile_name = profile
        # Re-parse now that real option values exist, so `#if` blocks around
        # `program.*.enabled` (confirmed real: Complementary Unbound gates
        # its shadow program behind `#if SHADOW_QUALITY == -1`) resolve
        # against this pack's actual defaults/profile, not the bootstrap
        # all-falsy guess.
        self.props = self.pack.properties(self.options.values())
        self.graph: PipelineGraph = build_graph(self.pack, self.world, self.options.values())
        self.resolver = RenderTypeResolver(self.pack)
        self._build()

    def swap_pack(self, pack_path: str) -> None:
        """Re-shade the whole scene with a different pack, keeping tags and
        the currently active profile.

        Previously called ``load_pack(pack_path)`` with no ``profile``,
        which silently reset every option to the *new* pack's raw defaults
        regardless of what profile (e.g. ULTRA) was active before the
        swap — "tags survive" was true, but the profile selection quietly
        wasn't, with nothing in the GUI reflecting the reset.
        """
        tags = list(self._geometry)
        profile = self.profile_name
        self._teardown()
        self.load_pack(pack_path, profile=profile)
        for nodepath, render_type in tags:
            self.set_render_type(nodepath, render_type)

    def apply_profile(self, name: str) -> None:
        self.options.apply_profile(name, self.props)
        self.profile_name = name
        self.recompile()

    def set_option(self, name: str, value: object) -> None:
        self.options.set(name, value)

    def recompile(self) -> None:
        """Rebuild shaders/passes after option changes."""
        tags = list(self._geometry)
        self._teardown()
        # Re-parse shaders.properties with the *current* option values too —
        # not just build_graph's own program_enabled resolution — so a live
        # option change (profile switch, settings-panel edit) that flips a
        # `#if`-gated program.*.enabled block re-resolves the same way
        # OptiFine/Iris does on every shader reload.
        self.props = self.pack.properties(self.options.values())
        self.graph = build_graph(self.pack, self.world, self.options.values())
        self._build()
        for nodepath, render_type in tags:
            self.set_render_type(nodepath, render_type)

    # -- object tagging (the "shader id" surface) -----------------------
    def set_render_type(self, nodepath: Any, render_type: str) -> None:
        """Shade ``nodepath`` as a Minecraft render type (e.g. "terrain")."""
        program = self.resolver.program(render_type)
        if program is None:
            raise ValueError(
                f"pack {self.pack.name!r} has no program for render type "
                f"{render_type!r}; available: {self.resolver.types()}"
            )
        nodepath.set_shader_input("mcEntityId", 0)
        self._geometry.append((nodepath, render_type))
        # Respect a standing set_enabled(False): re-tagging (a rebuild after
        # an option change, a swap_pack() replay) must not silently re-shade
        # geometry the caller explicitly turned the pack off for.
        if self._enabled:
            shader = self._compiled_geometry.get(program)
            if shader is not None:
                nodepath.set_shader(shader)
        else:
            nodepath.clear_shader()

    def set_enabled(self, enabled: bool) -> None:
        """Toggle between the pack's full shaded pipeline and Panda3D's own
        plain (unshaded) rendering of the same tagged scene — without
        reloading the pack or losing render-type tags, so you can A/B the
        shading against vanilla geometry.

        The offscreen gbuffer/composite chain keeps running underneath
        either way (cheap to leave alone); only the final composited
        quad(s) are hidden so the window falls through to Panda's own
        default display region — which renders the same scene graph, so
        clearing every tagged NodePath's shader is what actually changes
        its look there.
        """
        self._enabled = enabled
        for quad in self._final_quads:
            (quad.show if enabled else quad.hide)()
        for nodepath, render_type in self._geometry:
            if enabled:
                program = self.resolver.program(render_type)
                shader = self._compiled_geometry.get(program)
                if shader is not None:
                    nodepath.set_shader(shader)
            else:
                nodepath.clear_shader()

    def set_block_id(self, nodepath: Any, block_id: int) -> None:
        """Feed the block id a gbuffers shader keys waving/material off of.

        Minecraft delivers this per-vertex via ``mc_Entity.x``; for engine-authored
        geometry we expose it as the ``mcEntityId`` shader input (see README —
        this is an approximation until per-vertex attributes are wired).
        """
        nodepath.set_shader_input("mcEntityId", int(block_id))

    def clear(self, nodepath: Any) -> None:
        nodepath.clear_shader()
        self._geometry = [(np, rt) for np, rt in self._geometry if np != nodepath]

    def build_sky(self, radius: float = 900.0) -> Any:
        """Attach a working procedural sky — gradient, sun, moon, stars, aurora.

        BSL's ``gbuffers_skybasic`` computes all of that entirely from the
        screen-space ray direction it reconstructs per pixel (see
        ``GetSkyColor``/``ShaderSunMoon``/``DrawStars`` in
        ``lib/atmospherics/sky.glsl`` and friends) — the geometry's only job
        is to cover the sky so the fragment shader runs there. So this is
        just a big sphere, tagged with the "sky_basic" render type, kept
        centred on the camera every frame (see ``_update``) like a
        conventional skybox: background bin, no depth test/write, so it
        never has to be precisely sized relative to real geometry.

        Sized to 90% of the camera's own far clip distance by default so it
        works without the caller having to coordinate lens settings; pass a
        smaller ``radius`` if you want the horizon closer.
        """
        far = self.base.camLens.get_far() if self.base.camLens else radius
        radius = min(radius, far * 0.9)

        sky = self.base.loader.load_model("models/misc/sphere")
        sky.reparent_to(self.base.render)
        sky.set_scale(radius)
        sky.set_bin("background", 0)
        sky.set_depth_test(False)
        sky.set_depth_write(False)
        sky.set_two_sided(True)  # the camera sits inside the sphere
        sky.set_light_off(1)
        # gbuffers_skybasic.glsl hides vanilla's own sunset-gradient quad by
        # checking for a *grayscale* vertex colour (`gl_Color.r==g==b`, the
        # branch taken here since we don't feed an MC_VERSION >= 1.16.5
        # renderStage) — sphere.egg's default white vertex colour would trip
        # that and force alpha to 0, making the whole dome invisible. Any
        # non-grayscale colour sidesteps it; the shader ignores vertex
        # colour entirely otherwise; it's computing everything itself.
        sky.set_color(1, 0, 0, 1)

        self.set_render_type(sky, "sky_basic")
        self._sky_np = sky
        return sky

    # -- build / teardown -----------------------------------------------
    def _build(self) -> None:
        self._uniform_types: dict[str, str] = {}   # non-sampler uniforms -> glsl type
        self._sampler_names: set[str] = set()       # every declared sampler name
        self._fallback = None
        self._fallback_normal = None
        self._maxattach = self._max_color_attachments()
        self._gbuffer_map = self._compute_gbuffer_map()
        self._alloc_buffers()
        self._compiled_geometry = self._compile_geometry_programs()
        self._build_shadow_pass()
        self._build_scene_target()
        self._build_fullscreen_chain()
        # Bind samplers on render so all tagged geometry inherits them; per-frame
        # scalar/matrix uniforms are pushed by the update task.
        self._bind_inputs(self.base.render)
        self._ensure_task()

    def _max_color_attachments(self) -> int:
        """Usable simultaneous render targets (Panda binds 1 color + 4 aux)."""
        try:
            n = self.base.win.get_gsg().get_max_color_targets()
            return max(1, min(int(n), 5))
        except Exception:
            return 5

    def _compute_gbuffer_map(self) -> dict[int, int]:
        """Pack the distinct colortex the gbuffers write into compact attachment
        slots (0..maxattach-1). Buffers beyond the limit are dropped (their writes
        are discarded — see README on GPU attachment limits)."""
        distinct = sorted({o for p in self.graph.geometry_passes() for o in p.outputs})
        distinct = distinct[: self._maxattach]
        return {colortex: slot for slot, colortex in enumerate(distinct)}

    def _teardown(self) -> None:
        from panda3d.core import GraphicsOutput  # noqa: F401

        for quad in self._quads:
            try:
                quad.remove_node()
            except Exception:
                pass
        for buf in getattr(self, "_buffers", []):
            try:
                self.base.graphicsEngine.remove_window(buf)
            except Exception:
                pass
        # _build_shadow_pass attaches a fresh shadow camera to base.render on
        # every (re)build; without removing the old one first, every profile
        # switch / option change / pack swap left a stale Camera node (and,
        # by extension, its shader/RenderState) permanently in the scene
        # graph. Confirmed by instrumentation: 8 profile switches -> 8
        # leaked cameras under render, never fewer. That unbounded growth is
        # what eventually corrupts rendering and forces a relaunch.
        shadow_cam = getattr(self, "_shadow_cam", None)
        if shadow_cam is not None:
            try:
                shadow_cam.remove_node()
            except Exception:
                pass
            self._shadow_cam = None
        self._quads.clear()
        self._final_quads.clear()
        self._colortex.clear()
        self._colortex_back.clear()
        self._buffers = []
        for nodepath, _ in self._geometry:
            try:
                nodepath.clear_shader()
            except Exception:
                pass
        self._geometry.clear()

    def _compile_geometry_programs(self) -> dict[str, Any]:
        """Compile every gbuffers program this pack ships, keyed by name."""
        from panda3d.core import Shader

        compiled: dict[str, Any] = {}
        wanted = {
            self.resolver.program(rt)
            for rt in self.resolver.types()
        } - {None}
        for program in wanted:
            shader = self._compile(program, mode="gbuffer")
            if shader is not None:
                compiled[program] = shader
        return compiled

    def _compile(self, name: str, *, mode: str = "compact",
                 allow_fallback_vertex: bool = False, return_info: bool = False) -> Any:
        """Translate + make a Panda3D shader for a program (None on failure).

        ``mode`` picks the fragment-output routing ("gbuffer" for the shared
        geometry FBO, "compact" for a single fullscreen pass). ``allow_fallback_vertex``
        lets a fullscreen pass with no vertex stage borrow the canonical quad vertex.
        ``return_info`` additionally returns the :class:`TranslatedPass` (needed
        to detect a pass that reads the same ``colortex`` it writes).
        """
        from panda3d.core import Shader
        from ..glsl.fullscreen import FULLSCREEN_VERTEX

        fail = (None, None) if return_info else None
        try:
            tp = translate_pass(
                self.pack, self.options, name, world=self.world, target=self.target,
                mode=mode, gbuffer_map=self._gbuffer_map, max_location=8)
            if not tp.fragment:
                return fail
            vertex = tp.vertex
            if vertex is None and allow_fallback_vertex:
                vertex = FULLSCREEN_VERTEX
            if vertex is None:
                return fail
            self._record_uniforms(vertex)
            self._record_uniforms(tp.fragment)
            shader = Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=tp.fragment)
            return (shader, tp) if return_info else shader
        except Exception as exc:  # a pass that won't compile degrades to skip
            print(f"[mcshader] program {name!r} skipped: {str(exc)[:200]}")
            return fail

    # -- buffers ---------------------------------------------------------
    def _tex_format(self, mc_format: str):
        from panda3d.core import Texture

        return getattr(Texture, _MC_FORMATS.get(mc_format, "F_rgba16"))

    def _alloc_colortex(self, index: int, mc_format: str) -> Any:
        from panda3d.core import Texture

        w = self.base.win.get_x_size() if self.base.win else 1280
        h = self.base.win.get_y_size() if self.base.win else 720
        tex = Texture(f"colortex{index}")
        tex.setup_2d_texture(w, h, Texture.T_float, self._tex_format(mc_format))
        tex.set_wrap_u(Texture.WM_clamp)
        tex.set_wrap_v(Texture.WM_clamp)
        tex.set_minfilter(Texture.FT_linear)
        tex.set_magfilter(Texture.FT_linear)
        return tex

    def _alloc_buffers(self) -> None:
        self._buffers = []
        for index, fmt in self.graph.buffers.formats.items():
            self._colortex[index] = self._alloc_colortex(index, fmt)
            # Ping-pong twin for composite read-after-write.
            self._colortex_back[index] = self._alloc_colortex(index, fmt)

    def _make_buffer(self, name: str, colortex_indices: list[int | None], want_depth: bool,
                     sort: int = -10) -> Any:
        """Create an offscreen MRT buffer with one attachment per entry of
        ``colortex_indices`` (primary color, then aux 0..3) — ``None`` for a
        slot with no meaningful colortex (cleared normally, nothing else
        cares). ``sort`` orders passes: more-negative renders earlier, so the
        gbuffer renders before composites, which render before the window.
        """
        from panda3d.core import (FrameBufferProperties, GraphicsPipe,
                                   GraphicsOutput, WindowProperties)

        n_color = len(colortex_indices)
        fb = FrameBufferProperties()
        fb.set_rgba_bits(16, 16, 16, 16)
        fb.set_float_color(True)
        # set_aux_rgba() requests plain 8-bit-per-channel AUX attachments, but
        # every colortex we allocate (_alloc_colortex) is a T_float texture —
        # RGBA16/RGB16F/etc, whatever the pack's colortexNFormat says.
        # set_aux_hrgba() requests half-float (16-bit-per-channel) AUX
        # attachments, matching what we actually allocate. This must stay in
        # lockstep with the RTP_aux_hrgba_* plane enum used below and in
        # _bind_scene_attachments/_render_quad — Panda has THREE distinct aux
        # attachment families (RTP_aux_rgba_N / RTP_aux_hrgba_N /
        # RTP_aux_float_N), one per precision, and a buffer only actually
        # has the planes matching whichever set_aux_* it was asked for.
        # Requesting hrgba here while the slot arrays elsewhere still used
        # RTP_aux_rgba_N pointed every "aux" attachment at a plane the
        # buffer never allocated — that produced a total white-screen
        # failure, not a partial one, which is exactly what surfaced this.
        fb.set_aux_hrgba(max(0, n_color - 1))
        if want_depth:
            fb.set_depth_bits(24)
        win = self.base.win
        buf = self.base.graphicsEngine.make_output(
            win.get_pipe(), name, sort, fb, WindowProperties.size(
                win.get_x_size(), win.get_y_size()),
            GraphicsPipe.BF_refuse_window, win.get_gsg(), win)
        # Every attachment (primary + aux) starts as driver-allocated VRAM,
        # not zeroed — unlike the fixed-point 8-bit format this used to
        # request, a genuine float attachment can hold NaN/Inf bit patterns,
        # and a single NaN sampled anywhere in a lighting calculation
        # poisons the whole expression (propagates through every add/mul
        # touching it). Force an explicit, real clear on every buffer we
        # create so a pass's first read of a not-yet-written aux target
        # sees defined zeros, not whatever was previously in that memory —
        # EXCEPT for a colortex the pack declares ``colortexNClear = false``
        # (BSL's colortex2/5/9): those are genuine cross-frame history
        # (TAA color, auto-exposure, lens-flare visibility, DOF focus — see
        # composite5/7.glsl), and clearing them every frame was the actual
        # cause of "colortex2 always reads back zero" — a GL clear on a
        # buffer runs unconditionally before that buffer's own pass draws,
        # regardless of what the pass's shader later overwrites, so it wiped
        # frame N's history before frame N+1's earliest reader ever saw it.
        # Leaving the plane uncleared is sufficient (no extra double-buffer
        # bookkeeping needed): each `colortex_indices` entry is the same
        # persistent Texture object every frame (never reallocated outside a
        # rebuild — see _alloc_buffers), and _render_quad's existing
        # self-read ping-pong (for a pass that both samples and writes the
        # same colortex) already gives BSL's real multi-pass history chains
        # (composite5 writes, composite7 reads-and-rewrites) a stable
        # 2-texture cycle that keeps working frame over frame for exactly
        # the same reason a single physical texture does.
        buf.set_clear_color_active(True)
        buf.set_clear_color((0, 0, 0, 0))
        plane_ids = [GraphicsOutput.RTP_color] + [
            getattr(GraphicsOutput, f"RTP_aux_hrgba_{i}") for i in range(4)
        ]
        for slot, plane in enumerate(plane_ids[:max(1, n_color)]):
            idx = colortex_indices[slot] if slot < len(colortex_indices) else None
            should_clear = idx is None or self.graph.buffers.clear.get(idx, True)
            buf.set_clear_active(plane, should_clear)
            if should_clear:
                buf.set_clear_value(plane, (0, 0, 0, 0))
        self._buffers.append(buf)
        return buf

    def _bind_scene_attachments(self, buf: Any) -> None:
        """Bind each packed gbuffer colortex to its attachment slot."""
        from panda3d.core import GraphicsOutput

        slots = [GraphicsOutput.RTP_color] + [
            getattr(GraphicsOutput, f"RTP_aux_hrgba_{i}") for i in range(4)
        ]
        for colortex, slot in sorted(self._gbuffer_map.items(), key=lambda kv: kv[1]):
            if slot < len(slots) and colortex in self._colortex:
                buf.add_render_texture(
                    self._colortex[colortex], GraphicsOutput.RTM_bind_or_copy, slots[slot])

    def _build_scene_target(self) -> None:
        """Point the main camera at the gbuffer MRT instead of the window."""
        slot_of = {slot: colortex for colortex, slot in self._gbuffer_map.items()}
        n_targets = max(1, len(self._gbuffer_map))
        gbuffer_indices = [slot_of.get(i) for i in range(n_targets)]
        # Render the gbuffer before every composite pass.
        self._scene_buf = self._make_buffer(
            "mcshader-gbuffer", gbuffer_indices, want_depth=True, sort=-100)
        self._scene_buf.set_clear_color((0, 0, 0, 1))
        self._bind_scene_attachments(self._scene_buf)
        dr = self._scene_buf.make_display_region()
        dr.set_camera(self.base.cam)

    def _sun_direction(self, time_angle: float = 0.25) -> Any:
        """Direction from the scene toward the sun, at a given ``timeAngle``
        (0..1, the fraction of a day — see `_day_cycle_uniforms`), in
        **Minecraft's own Y-up world convention** (Y = the vertical arc).

        Shares the shape of BSL's own in-shader sun-arc remap (see
        ``lib/atmospherics/sunmoon.glsl``'s ``ang`` calculation, and
        ``gbuffers_terrain.glsl``'s vertex-shader ``sunVec`` construction,
        which likewise puts the rise/set arc in the Y component) so the sun
        rises/sets at roughly the same pace the shaders animate it at.

        Every caller needing this in *this engine's* native Z-up scene graph
        (to position a real Panda camera/light node) must convert it first
        via :meth:`_cs_conversion` — see the matching comment in
        `_build_shadow_pass`/`_dynamic_uniforms` for why: Minecraft shaders
        (BSL confirmed directly — ``gbufferModelView[1]`` is read as the
        view-space up vector, ``GetCloudShadow``'s ``worldLightVec.y`` as the
        vertical component) assume world-space Y is up throughout, which this
        engine's native Panda convention (Y-forward, Z-up) is not.
        """
        import math
        from panda3d.core import LVecBase3

        ang = (time_angle + 0.0001 - 0.25) % 1.0
        ang = (ang + (math.cos(ang * math.pi) * -0.5 + 0.5 - ang) / 3.0) * 2.0 * math.pi
        d = LVecBase3(-math.sin(ang), math.cos(ang), 0.3)
        d.normalize()
        return d

    @staticmethod
    def _cs_conversion() -> tuple[Any, Any]:
        """Rotation-only matrices reconciling this engine's native Z-up,
        Y-forward world with the Y-up world every Minecraft shader assumes.

        This is the crux of a bug confirmed directly in BSL's own source:
        ``gbuffers_terrain.glsl``'s vertex shader reads ``gbufferModelView[1]``
        (the matrix's Y column) straight out as the view-space "up" vector,
        and ``lib/lighting/shadows.glsl``'s ``GetCloudShadow`` reads a
        reconstructed world vector's ``.y`` component as its vertical
        (height) component — both assume world-space Y *is* up, matching
        Minecraft's own convention. Feeding these uniforms unconverted Panda
        matrices (Y-forward, Z-up) silently swaps "up" onto a horizontal
        axis instead: the sky's vertical falloff runs along the wrong axis,
        shadows / sun direction come out rotated relative to the real scene,
        and every per-object effect keyed off world Y (waving amplitude,
        world curvature) is broken.

        Panda's own scene graph, and every transform that actually drives
        real rendering (the automatic ``p3d_ModelViewMatrix``/lens
        projection Panda supplies, untouched here), stay native Z-up
        throughout — this conversion only touches the shader-facing
        ``gbuffer*``/``shadow*`` uniforms computed in `_dynamic_uniforms`.
        Built from Panda's own well-tested coordinate-system utility
        (``LMatrix4.convert_mat``), not hand-derived, and applied as a pure
        rotation (no translation) — see `_rotation_only`.
        """
        from panda3d.core import LMatrix4, CS_zup_right, CS_yup_right

        return (LMatrix4.convert_mat(CS_zup_right, CS_yup_right),
                LMatrix4.convert_mat(CS_yup_right, CS_zup_right))

    @staticmethod
    def _rotation_only(mat: Any) -> Any:
        """The pure-rotation part of an affine ``LMatrix4`` (row-vector
        convention: ``v * M``, translation lives in row 3) — zeroes the
        translation, keeping just the 3x3 linear part.

        Minecraft's own ``gbufferModelView`` carries no translation (its
        vertex data is already camera-relative before any matrix applies);
        replicating that here, from Panda's real (translation-including)
        camera transform, is what keeps the shader-facing view matrices
        semantically correct — see `_cs_conversion`.
        """
        from panda3d.core import LMatrix4, LVecBase4

        m = LMatrix4(mat)
        m.set_row(3, LVecBase4(0, 0, 0, 1))
        return m

    @staticmethod
    def _safe_up(view_direction: Any) -> Any:
        """A Z-up-preferring ``up`` hint for ``look_at()`` that falls back to
        a perpendicular axis when ``view_direction`` is nearly parallel to
        the default up vector ``(0, 0, 1)``.

        ``look_at()`` with no explicit ``up`` builds its basis from
        ``cross(up, forward)``; as ``forward`` approaches parallel to
        ``(0, 0, 1)`` that cross product shrinks toward zero and the
        resulting orientation's *roll* becomes numerically unstable —
        hypersensitive to the exact input values (including tiny
        floating-point differences), not just genuinely undefined at the
        limit. The shadow camera looks down the sun direction, which is
        close to vertical for a large fraction of every simulated day (not
        an edge case near noon) — a plausible root cause for shadows
        reported as "wrongly shaped/huge" and appearing to reorient with
        camera movement despite the view direction itself only depending on
        the (slowly-changing) sun angle.
        """
        from panda3d.core import LVecBase3

        d = LVecBase3(view_direction)
        if d.length_squared() < 1e-12:
            return LVecBase3(0, 0, 1)
        d.normalize()
        if abs(d.dot(LVecBase3(0, 0, 1))) > 0.95:
            return LVecBase3(0, 1, 0)
        return LVecBase3(0, 0, 1)

    def _build_shadow_pass(self) -> None:
        from panda3d.core import (Camera, OrthographicLens, NodePath, GraphicsOutput,
                                  FrameBufferProperties, GraphicsPipe, WindowProperties)

        shadow_shader = self._compile("shadow", mode="gbuffer")
        if shadow_shader is None:
            self._shadow_buf = None
            self._shadow_tex = None
            self._shadow_cam = None
            return
        res = int(self.options.get("shadowMapResolution")) if "shadowMapResolution" in self.options.options else 2048
        # Packs declare shadowtex0/1 as sampler2DShadow (hardware depth-compare),
        # so the shadow buffer needs a real depth texture attached — a bare
        # make_texture_buffer() only gives a colour texture, which sampled as a
        # shadow comparison is meaningless (permanently in-shadow or garbage).
        from panda3d.core import Texture
        fb = FrameBufferProperties()
        fb.set_depth_bits(24)
        # shadowcolor0 (colored/translucent shadow casting — stained glass,
        # leaves, BSL's SHADOW_COLOR and WATER_CAUSTICS features) needs a real
        # RGBA color attachment: shadow.glsl's fragment stage always writes
        # gl_FragData[0] = albedo, but a depth-only FBO has nowhere for that
        # write to land, so it was silently discarded and every shadow was
        # opaque/uncolored regardless of what SHADOW_COLOR requested.
        fb.set_rgba_bits(8, 8, 8, 8)
        win = self.base.win
        self._shadow_buf = self.base.graphicsEngine.make_output(
            win.get_pipe(), "mcshader-shadow", -200, fb,
            WindowProperties.size(res, res), GraphicsPipe.BF_refuse_window,
            win.get_gsg(), win)
        self._shadow_tex = Texture("mcshader-shadow-depth")
        self._shadow_tex.set_wrap_u(Texture.WM_border_color)
        self._shadow_tex.set_wrap_v(Texture.WM_border_color)
        self._shadow_tex.set_border_color((1, 1, 1, 1))  # outside shadow bounds = lit
        self._shadow_tex.set_minfilter(Texture.FT_shadow)
        self._shadow_tex.set_magfilter(Texture.FT_shadow)
        self._shadow_buf.add_render_texture(
            self._shadow_tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_depth)
        self._shadow_color_tex = Texture("mcshader-shadow-color")
        self._shadow_color_tex.set_wrap_u(Texture.WM_border_color)
        self._shadow_color_tex.set_wrap_v(Texture.WM_border_color)
        # Outside the shadow frustum, transmit full light with no tint — must
        # NOT reuse shadowtex's (1,1,1,1) border verbatim by accident sharing
        # state; this is its own Texture with its own border, set explicitly.
        self._shadow_color_tex.set_border_color((1, 1, 1, 1))
        self._shadow_color_tex.set_minfilter(Texture.FT_linear)
        self._shadow_color_tex.set_magfilter(Texture.FT_linear)
        self._shadow_buf.add_render_texture(
            self._shadow_color_tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_color)
        self._buffers.append(self._shadow_buf)
        dist = float(self.options.get("shadowDistance")) if "shadowDistance" in self.options.options else 256.0
        self._shadow_dist = dist
        lens = OrthographicLens()
        lens.set_film_size(dist, dist)
        lens.set_near_far(1.0, dist * 2.0)
        cam = Camera("mcshader-shadow-cam", lens)
        # Reuse the same "gbuffer"-mode shader checked for existence above —
        # this used to call self._compile("shadow") again with no mode
        # argument (silently defaulting to "compact", the fullscreen-quad
        # output-numbering scheme), recompiling the *same* program a second
        # time with the *wrong* mode for real scene geometry. Harmless for
        # BSL (whose shadow.glsl only ever writes gl_FragData[0], where both
        # modes number identically) but not general, and wasteful either way.
        from panda3d.core import ShaderAttrib, RenderState
        cam.set_initial_state(
            RenderState.make(ShaderAttrib.make(shadow_shader)))
        self._shadow_cam = self.base.render.attach_new_node(cam)
        # Point the shadow camera down the sun direction, centred on the
        # PLAYER (base.cam), not the world origin. Minecraft's own shadow
        # frustum is always player-relative — the world has no meaningful
        # origin to center on — and shadow.vsh's distortion formula (see
        # Shaders/shaders/program/shadow.glsl) concentrates virtually all of
        # the shadow map's resolution within a small disc around distortion
        # "dist=0", i.e. around wherever this frustum is centered. Centering
        # on a fixed (0,0,0) instead of the camera meant shadows only ever
        # looked right for geometry sitting near the world origin, and get
        # more visibly wrong the further the camera roams — and *more* wrong
        # at higher shadowDistance profiles (HIGH/ULTRA), since a bigger
        # shadowDistance shrinks the well-resolved disc's share of the frustum
        # even further relative to the (still misplaced) visible scene. See
        # the matching per-frame update in `_dynamic_uniforms`.
        from panda3d.core import LPoint3
        center = (self.base.cam.get_pos(self.base.render) if self.base.cam
                  else LPoint3(0, 0, 0))
        # _sun_direction() is in Minecraft's Y-up convention (see its
        # docstring); this engine's actual scene-graph node needs it back in
        # native Z-up to be positioned correctly — see `_cs_conversion`.
        _, cs_yup_to_zup = self._cs_conversion()
        sun_panda = cs_yup_to_zup.xform_vec(self._sun_direction())
        self._shadow_cam.set_pos(center + sun_panda * (dist * 0.5))
        self._shadow_cam.look_at(center, self._safe_up(-sun_panda))
        dr = self._shadow_buf.make_display_region()
        dr.set_camera(self._shadow_cam)

    # -- fullscreen chain -----------------------------------------------
    _COLORTEX_REF_RE = re.compile(r"\bcolortex(\d+)\b")

    def _self_read_outputs(self, tp: Any) -> set[int]:
        """``colortex`` indices a pass both writes and samples as an input.

        BSL-style composite chains refine one buffer across several passes
        (e.g. colortex1 through composite/composite5/composite6/composite7),
        each reading the previous pass's result while also writing the next.
        Sampling and rendering into the *same* texture in one draw call is a
        feedback loop (undefined in GL) — it's what produced speckled noise
        in place of the refined image. These indices must render into the
        ping-pong back buffer instead; see :meth:`_render_quad`.

        Applies uniformly, including to buffers the pack marks
        ``colortexNClear = false`` (BSL's colortex2/5/9 — real cross-frame
        history: auto-exposure, TAA color, lens-flare visibility, DOF focus).
        Exempting those from the swap was tried once and made things worse —
        two different passes (composite5 and composite7, both writing
        colortex2) then both targeted the exact same Texture as their own
        buffer's render attachment, which Panda doesn't support (the write
        went nowhere). The swap must stay; what those buffers actually needed
        was for `_make_buffer` to stop *clearing* the plane each frame (a GL
        clear runs unconditionally before that buffer's own pass draws,
        which was wiping frame N's history before frame N+1's earliest
        reader ever saw it — the real cause of "colortex2 always reads back
        zero", not this ping-pong). With clearing fixed, this build-time
        swap is *sufficient* for correct frame-to-frame persistence on its
        own: composite5's/composite7's fixed sampler and render-target
        bindings (established once here, at build time, from whichever two
        physical Textures the swap leaves them pointing at) never change
        again, so every subsequent frame composite5 reads whatever
        composite7 wrote last frame and vice versa — a stable, self-sustaining
        2-texture cycle, no per-frame bookkeeping needed.
        """
        if tp is None or not tp.fragment:
            return set()
        referenced = {int(m) for m in self._COLORTEX_REF_RE.findall(tp.fragment)}
        return referenced & set(tp.outputs)

    def _build_fullscreen_chain(self) -> None:
        from panda3d.core import CardMaker, NodePath, Camera, OrthographicLens

        values = self.options.values()
        passes = [p for p in self.graph.fullscreen_passes()
                  if p.kind != "shadowcomp" and p.enabled(values)]
        for i, p in enumerate(passes):
            shader, tp = self._compile(p.name, mode="compact", allow_fallback_vertex=True,
                                       return_info=True)
            if shader is None:
                continue
            cm = CardMaker(f"mcshader-{p.name}")
            cm.set_frame_fullscreen_quad()
            quad = NodePath(cm.generate())
            quad.set_shader(shader)
            self._bind_inputs(quad, is_quad=True)
            is_final = (p.kind == "final")
            hazards = self._self_read_outputs(tp)
            self._render_quad(quad, p, is_final=is_final, order=i, hazards=hazards)
            self._quads.append(quad)
            if is_final:
                self._final_quads.append(quad)
                if not self._enabled:  # a rebuild while pack is toggled off
                    quad.hide()

    def _render_quad(self, quad: Any, p: Any, *, is_final: bool, order: int,
                     hazards: frozenset[int] | set[int] = frozenset()) -> None:
        """Render one fullscreen pass into its target colortex (or the window)."""
        from panda3d.core import (Camera, OrthographicLens, NodePath, GraphicsOutput)

        quad.set_depth_test(False)
        quad.set_depth_write(False)

        if is_final:
            # set_frame_fullscreen_quad() cards are made for render2d, which
            # ShowBase already draws to the window — the reliable way to blit.
            quad.reparent_to(self.base.render2d)
            quad.set_bin("fixed", 100 + order)
            return

        # Intermediate pass: render the quad into its target colortex via a small
        # 2D camera (render2d-style: XZ-plane card viewed down +Y).
        scene = NodePath(f"mcshader-scene-{p.name}")
        quad.reparent_to(scene)
        lens = OrthographicLens()
        lens.set_film_size(2, 2)
        lens.set_near_far(-1000, 1000)
        cam = scene.attach_new_node(Camera(f"mcshader-cam-{p.name}", lens))

        # Composites render after the gbuffer, in pipeline order, before the window.
        # A pass's DRAWBUFFERS can name several colortex (e.g. deferred1 writes
        # colortex0/4/5/6) — the fragment shader's location k is k-th in that list
        # (see translate.py's "compact" mode), so bind every one of them, not just
        # the first, or every buffer past the first silently never gets written.
        targets = p.outputs or [0]
        n_color = max(1, min(len(targets), self._maxattach))
        buf = self._make_buffer(f"mcshader-{p.name}", targets[:n_color], want_depth=False,
                                sort=-50 + order)
        slots = [GraphicsOutput.RTP_color] + [
            getattr(GraphicsOutput, f"RTP_aux_hrgba_{i}") for i in range(4)
        ]
        for slot, colortex in enumerate(targets[:n_color]):
            # A pass that samples the same colortex it writes (BSL's colortex1
            # refinement chain) must render into the ping-pong back buffer, not
            # the one it (and this quad's own shader input) is reading from —
            # sampling and rendering the same texture in one draw is a GL
            # feedback loop and was the source of the speckled-noise artifact.
            dest = (self._colortex_back[colortex] if colortex in hazards
                    else self._colortex[colortex])
            buf.add_render_texture(dest, GraphicsOutput.RTM_bind_or_copy, slots[slot])
        dr = buf.make_display_region()
        dr.set_camera(cam)
        for colortex in hazards:
            if colortex in targets[:n_color]:
                self._colortex[colortex], self._colortex_back[colortex] = (
                    self._colortex_back[colortex], self._colortex[colortex])

    _UNIFORM_RE = None

    def _record_uniforms(self, source: str) -> None:
        """Collect ``uniform`` declarations so every input can be fed a value."""
        import re

        if PipelineRenderer._UNIFORM_RE is None:
            # Capture the type and the full (possibly comma-separated) name list.
            PipelineRenderer._UNIFORM_RE = re.compile(
                r"^\s*uniform\s+(\w+)\s+([^;]+);", re.M)
        for gtype, names in PipelineRenderer._UNIFORM_RE.findall(source):
            for raw in names.split(","):
                name = raw.strip().split("[")[0].strip()  # drop any array suffix
                if not name or not name.isidentifier():
                    continue
                if name.startswith(("p3d_", "osg_")):
                    continue  # auto-provided by Panda3D
                if "sampler" in gtype:
                    self._sampler_names.add(name)
                else:
                    self._uniform_types[name] = gtype

    def _fallback_tex(self) -> Any:
        from panda3d.core import Texture

        if self._fallback is None:
            tex = Texture("mcshader-fallback")
            tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_rgba)
            tex.set_ram_image(bytes([0, 0, 0, 255]))
            self._fallback = tex
        return self._fallback

    def _fallback_normal_tex(self) -> Any:
        """A flat tangent-space normal map (decodes to (0,0,1), i.e. "no bump").

        Packs decode their ``normals`` PBR map as ``texture(normals, uv).xyz *
        2.0 - 1.0``; our generic black fallback decodes to (-1,-1,-1), an
        invalid normal that kills every N·L lighting term touching it — a
        surface with no normal map would render solid black instead of
        matching its unbumped geometric normal. (0,0,0,255) stays correct as
        the fallback for the unrelated ``specular``/PBR map — LabPBR's own
        convention for "no data" is all-zero, matching vanilla resource packs.
        """
        from panda3d.core import Texture

        if getattr(self, "_fallback_normal", None) is None:
            tex = Texture("mcshader-fallback-normal")
            tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_rgba)
            tex.set_ram_image(bytes([128, 128, 255, 255]))
            self._fallback_normal = tex
        return self._fallback_normal

    def _bind_inputs(self, node: Any, *, is_quad: bool = False) -> None:
        """Bind every sampler the shaders declare (colortex/shadow/fallback)."""
        bound: set[str] = set()

        def bind(name: str, tex: Any) -> None:
            node.set_shader_input(name, tex)
            bound.add(name)

        for index, tex in self._colortex.items():
            bind(f"colortex{index}", tex)
        aliases = {"gcolor": 0, "gdepth": 1, "gnormal": 2, "composite": 3,
                   "gaux1": 4, "gaux2": 5, "gaux3": 6, "gaux4": 7}
        for alias, idx in aliases.items():
            bind(alias, self._colortex.get(idx, self._fallback_tex()))
        shadow_depth = getattr(self, "_shadow_tex", None) or self._fallback_tex()
        for s in ("shadowtex0", "shadowtex1"):
            bind(s, shadow_depth)
        # shadowcolor0/1 are plain (non depth-compare) sampler2D uniforms that
        # hold the shadow pass's rendered albedo (colored/translucent shadow
        # casting). Binding them to the SAME Texture object as shadowtex0/1
        # is wrong twice over: semantically they're a different image, and
        # that texture's sampler state is configured for hardware depth
        # comparison (FT_shadow) — sampling it as a plain color would read
        # back a 0/1 comparison result, not albedo, wherever the GSG doesn't
        # give the two bindings independent sampler state.
        shadow_color = getattr(self, "_shadow_color_tex", None)
        opaque_white = getattr(self, "_opaque_white_tex", None)
        if opaque_white is None:
            from panda3d.core import Texture as _Texture
            opaque_white = _Texture("mcshader-opaque-white")
            opaque_white.setup_2d_texture(1, 1, _Texture.T_unsigned_byte, _Texture.F_rgba)
            opaque_white.set_ram_image(bytes([255, 255, 255, 255]))
            self._opaque_white_tex = opaque_white
        bind("shadowcolor0", shadow_color if shadow_color is not None else opaque_white)
        # shadow.glsl only ever writes gl_FragData[0]; shadowcolor1 has no
        # producer in this pack, so it gets the semantically-neutral
        # "fully lit, no tint" default rather than aliasing shadowcolor0.
        bind("shadowcolor1", opaque_white)
        for s in ("depthtex0", "depthtex1", "depthtex2"):
            bind(s, self._colortex.get(0, self._fallback_tex()))
        bind("noisetex", self._fallback_tex())
        # A fullscreen quad has no model texture; give its base sampler colortex0.
        if is_quad:
            bind("p3d_Texture0", self._colortex.get(0, self._fallback_tex()))
        # Anything else the shaders declared but we don't recognise -> fallback,
        # WITHOUT clobbering the real bindings above. A name containing
        # "normal" gets the flat-normal fallback (see _fallback_normal_tex);
        # everything else gets the neutral black/zero fallback.
        for name in self._sampler_names:
            if name not in bound and name != "p3d_Texture0":
                fallback = self._fallback_normal_tex() if "normal" in name.lower() else self._fallback_tex()
                node.set_shader_input(name, fallback)

    # -- per-frame uniforms ---------------------------------------------
    def _ensure_task(self) -> None:
        if self._task_started or self.base is None:
            return
        self.base.taskMgr.add(self._update, "mcshader-pipeline-uniforms")
        self._task_started = True

    #: Real seconds per in-game day, driving worldTime (and everything derived
    #: from it — sun position, sky colour, shadowFade) so lighting actually
    #: changes instead of sitting at whatever a single fixed constant gave it.
    _DAY_LENGTH_SECONDS = 120.0

    def _day_cycle_uniforms(self, t: float, frame_count: int) -> dict[str, Any]:
        """The handful of ``shaders.properties`` custom uniforms BSL actually
        needs (see shaders.properties' "Custom Time/Blindness/Frame Jitter
        Uniform" blocks) — hand-evaluated rather than through a general
        expression-DSL evaluator (not implemented), since this pack only
        needs this fixed, small set. Formulas copied verbatim from
        Shaders/shaders/shaders.properties.

        Getting these fed at all matters far more than getting worldTime's
        pacing "authentic": left at their generic zero default, `shadowFade`
        zeroes the direct-light term in BSL's `GetLighting()` (`shadowMult =
        shadowFade * ...`), which is why terrain/entities rendered solid
        black — an ambient-only, un-lit-by-the-sun scene is indistinguishable
        from a broken one.
        """
        import math

        world_time = (t / self._DAY_LENGTH_SECONDS * 24000.0) % 24000.0
        time_angle = world_time / 24000.0

        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))

        shadow_fade_out1 = clamp01((world_time - 12330) / 230)
        shadow_fade_in1 = clamp01((world_time - 13010) / 220)
        shadow_fade_out2 = clamp01((world_time - 22770) / 220)
        shadow_fade_in2 = clamp01((world_time - 23440) / 230)
        shadow_fade = 1.0 - (shadow_fade_out1 - shadow_fade_in1 + shadow_fade_out2 - shadow_fade_in2)
        time_brightness = max(math.sin(time_angle * 2.0 * math.pi), 0.0)

        return {
            "worldTime": world_time, "worldDay": int(world_time // 24000),
            "sunAngle": time_angle, "timeAngle": time_angle,
            "shadowFade": shadow_fade, "timeBrightness": time_brightness,
            "framemod8": float(frame_count % 8), "framemod2": float(frame_count % 2),
            "blindFactor": 0.0, "blindness": 0.0, "moonPhase": 0,
        }

    def _dynamic_uniforms(self) -> dict[str, Any]:
        """Compute the frame-varying uniform values we can derive from the scene."""
        from panda3d.core import ClockObject, LMatrix4, LVecBase3

        clock = ClockObject.get_global_clock()
        t = clock.get_frame_time() % 3600.0
        # base.cam (not base.camera, its parent — see _build_scene_target's
        # dr.set_camera) is the node actually driving the render; base.camera
        # sits at the origin by default and only base.cam moves when callers
        # (rightly) pose the camera via base.cam.set_pos/look_at. Using the
        # wrong one here meant gbufferModelView/cameraPosition/etc. stayed
        # fixed at the origin no matter where the camera actually was.
        render, cam_np, lens = self.base.render, self.base.cam, self.base.camLens
        w = float(self.base.win.get_x_size()); h = float(self.base.win.get_y_size())
        cam_pos = cam_np.get_pos(render)
        frame_count = int(clock.get_frame_count())
        day = self._day_cycle_uniforms(t, frame_count)

        cs_zup_to_yup, cs_yup_to_zup = self._cs_conversion()

        # gbufferModelView/Inverse: OpenGL-standard view space (X-right,
        # Y-up, Z-backward — the space every BSL shader's own math assumes
        # its "view space" to be in) <-> Minecraft-Y-up world (relative to
        # the PLAYER camera). This is a *three*-matrix rotation, not two:
        # `cs_yup_to_zup` first relabels the OpenGL-view-space input back
        # into Panda's native camera-local axes (X-right, Y-forward, Z-up —
        # what Panda's own camera actually looks down), THEN `rot_player`
        # carries it into native Panda world space, THEN `cs_zup_to_yup`
        # relabels that into Minecraft's Y-up world. Dropping the first
        # term (an earlier version of this code did) leaves the *shape* of
        # gbufferModelView looking plausible — it's still a pure rotation,
        # still self-cancels with its own inverse — but silently mismatches
        # gbufferProjection, whose diagonal-shortcut consumers assume the
        # standard convention (see the `proj` comment below): confirmed by
        # reconstructing Panda's own CS_yup_right-lens projection matrix
        # numerically as `cs_yup_to_zup * <Panda's native projection>` and
        # checking it against Panda's own output with that coordinate
        # system set — they match exactly.
        #
        # Built as a true inverse pair (gbuffer_mv is the *numeric* inverse
        # of gbuffer_mv_inv) so every self-cancelling
        # `gbufferModelView * gbufferModelViewInverse * ...` chain BSL's own
        # vertex shaders use (see gbuffers_terrain.glsl) reduces to Panda's
        # own real per-object transform, regardless of this matrix's
        # semantics — real rendered geometry position is provably unaffected
        # by this reconciliation, only the shaders' own world-space math is.
        rot_player = self._rotation_only(cam_np.get_mat(render))  # player-view -> world
        gbuffer_mv_inv = cs_yup_to_zup * rot_player * cs_zup_to_yup
        gbuffer_mv = LMatrix4(gbuffer_mv_inv)
        gbuffer_mv.invert_in_place()

        # gbufferProjection must be in the *standard* OpenGL projection
        # matrix convention, not Panda's native one. This isn't just about
        # axis orientation for full `proj * modelview * v` chains (those
        # would tolerate any consistent convention) — a lot of BSL's own
        # code (`ToNDC`/`ToShadow`'s `projMAD`/`diagonal3` macros in
        # spaceConversion.glsl, `GetLinearDepth` used throughout AO/light
        # shafts/outline/bloom) reads specific *cells* of this matrix
        # assuming the sparse layout of a standard symmetric-frustum
        # projection (`m[0].x`/`m[1].y`/`m[2].zw`/`m[3]`). Panda's native
        # `lens.get_projection_mat()` is mathematically a valid projection
        # but puts those same terms in *different* cells (its camera-local
        # convention is X-right/Y-forward/Z-up, not X-right/Y-up/
        # Z-backward) — those shortcuts silently read zeros/garbage instead
        # of the real depth/aspect terms. This was the root cause behind
        # screen-space effects (AO, light shafts, SSR, bloom, DOF) and
        # shadow-space reconstruction (`ToShadow`, see `shadowProjection`
        # below) all being visibly broken despite the shadow map and gbuffer
        # geometry themselves rendering correctly.
        proj = cs_yup_to_zup * LMatrix4(lens.get_projection_mat())
        proj_inv = LMatrix4(proj); proj_inv.invert_in_place()

        camera_position_mc = cs_zup_to_yup.xform_point(cam_pos)

        sun_mc = self._sun_direction(day["timeAngle"])
        sun_view = gbuffer_mv.xform_vec(sun_mc) * 100.0
        up_view = gbuffer_mv.xform_vec(LVecBase3(0, 1, 0)) * 100.0

        shadow_cam_np = getattr(self, "_shadow_cam", None)
        if shadow_cam_np is not None:
            # Track the moving sun AND the player each frame, so the shadow
            # map (and the shadowModelView/Projection derived from it below)
            # stays centred on the camera — see the matching comment in
            # `_build_shadow_pass` for why a fixed world-origin center is
            # wrong. Re-centering every frame is what makes shadows follow
            # the player around an arbitrarily large world instead of only
            # ever looking right near (0,0,0).
            dist = getattr(self, "_shadow_dist", 256.0)
            sun_panda = cs_yup_to_zup.xform_vec(sun_mc)
            shadow_cam_np.set_pos(cam_pos + sun_panda * (dist * 0.5))
            shadow_cam_np.look_at(cam_pos, self._safe_up(-sun_panda))

            # Same standard-convention fix as `proj` above — confirmed the
            # identical relabeling (`cs_yup_to_zup * <native>`) reproduces
            # Panda's own CS_yup_right output for OrthographicLens too, not
            # just PerspectiveLens. `ToShadow` (spaceConversion.glsl) uses
            # the same `projMAD`/`diagonal3` cell-shortcut on
            # shadowProjection that `ToNDC` uses on gbufferProjection — this
            # was the actual cause of `GetShadow()` reading a garbage
            # reference depth (see shadows.glsl), not a bias/acne issue as
            # previously suspected.
            s_proj = cs_yup_to_zup * LMatrix4(shadow_cam_np.node().get_lens().get_projection_mat())
            s_proj_inv = LMatrix4(s_proj); s_proj_inv.invert_in_place()

            # Same rotation-only reconciliation as gbufferModelView, but the
            # shadow camera sits at a different real position than the
            # player — `position`/`worldPos` in every BSL program (shadow
            # pass included: see shadow.glsl's `worldPos = position.xyz +
            # cameraPosition.xyz`) is always relative to the PLAYER's
            # cameraPosition, so this also carries the
            # (shadow-cam -> player-cam) offset, re-expressed in Minecraft's
            # Y-up axes, as a translation. Still a true inverse pair with
            # s_mv (numeric inverse below), so — exactly as with
            # gbufferModelView — the real rendered shadow-map geometry
            # (shadow.glsl's own self-cancelling
            # `shadowProjection*shadowModelView*shadowModelViewInverse*
            # shadowProjectionInverse*ftransform()` chain) is provably
            # unaffected by this reconciliation even if it were wrong; only
            # the shadow pass's own world-space math (waving, water
            # caustics) depends on getting the offset right.
            rot_shadow = self._rotation_only(shadow_cam_np.get_mat(render))
            offset = cs_zup_to_yup.xform_vec(shadow_cam_np.get_pos(render) - cam_pos)
            s_mv_inv = cs_yup_to_zup * rot_shadow * cs_zup_to_yup * LMatrix4.translate_mat(offset)
            s_mv = LMatrix4(s_mv_inv)
            s_mv.invert_in_place()
        else:
            s_mv, s_mv_inv, s_proj, s_proj_inv = gbuffer_mv, gbuffer_mv_inv, proj, proj_inv

        # gbufferPreviousModelView/Projection and previousCameraPosition must
        # be genuinely LAST frame's values, not a copy of this frame's — TAA
        # (taa.glsl's Reprojection) unprojects the current pixel with the
        # *current* inverse matrices, offsets by `cameraPosition -
        # previousCameraPosition`, then reprojects with the *previous*
        # forward matrices to find where that same world point was on
        # screen last frame. Feeding identical current/previous matrices
        # (as this did before) makes `cameraOffset` permanently zero and
        # the reprojection a no-op — every reprojected sample lands back at
        # the *current* screen position regardless of real camera motion,
        # so TAA blends the new frame with a history texel that (whenever
        # the camera actually moved) represents a different world point.
        # That mismatch is exactly what reads as persistent speckle/noise
        # ("staticy") under camera motion, on top of AO leaning on the same
        # history buffer. Cached on `self` and updated at the end of every
        # call, one frame behind on purpose.
        prev_gbuffer_mv = getattr(self, "_prev_gbuffer_mv", gbuffer_mv)
        prev_proj = getattr(self, "_prev_proj", proj)
        prev_camera_position_mc = getattr(self, "_prev_camera_position_mc", camera_position_mc)
        self._prev_gbuffer_mv = gbuffer_mv
        self._prev_proj = proj
        self._prev_camera_position_mc = camera_position_mc

        return {
            "frameTimeCounter": t, "frameTime": clock.get_dt(),
            "frameCounter": frame_count,
            "viewWidth": w, "viewHeight": h, "aspectRatio": w / max(h, 1.0),
            "near": lens.get_near(), "far": lens.get_far(),
            **day, "rainStrength": 0.0, "wetness": 0.0,
            "cameraPosition": camera_position_mc, "previousCameraPosition": prev_camera_position_mc,
            "gbufferModelView": gbuffer_mv, "gbufferModelViewInverse": gbuffer_mv_inv,
            "gbufferPreviousModelView": prev_gbuffer_mv,
            "gbufferProjection": proj, "gbufferProjectionInverse": proj_inv,
            "gbufferPreviousProjection": prev_proj,
            "shadowModelView": s_mv, "shadowModelViewInverse": s_mv_inv,
            "shadowProjection": s_proj, "shadowProjectionInverse": s_proj_inv,
            "sunPosition": sun_view, "moonPosition": sun_view * -1.0,
            "shadowLightPosition": sun_view, "upPosition": up_view,
        }

    # Uniforms whose semantically-correct "unbound" default differs from the
    # generic per-type default below — e.g. entityColor is a (tint, blend)
    # pair consumed as `mix(albedo, entityColor.rgb, entityColor.a)`; alpha=0
    # means "no tint, keep the real texture", but the generic vec4 default's
    # alpha=1 means "always fully overridden by black", blacking out every
    # entity/actor regardless of its own texture.
    _NAMED_DEFAULTS = {
        "entityColor": (0.0, 0.0, 0.0, 0.0),
    }

    def _default_for(self, gtype: str) -> Any:
        from panda3d.core import (LMatrix3, LMatrix4, LVecBase2, LVecBase3, LVecBase4,
                                  LVecBase2i, LVecBase3i, LVecBase4i)

        return {
            "float": 0.0, "int": 0, "bool": 0, "uint": 0,
            "vec2": LVecBase2(0, 0), "vec3": LVecBase3(0, 0, 0),
            "vec4": LVecBase4(0, 0, 0, 1),
            "ivec2": LVecBase2i(0, 0), "ivec3": LVecBase3i(0, 0, 0),
            "ivec4": LVecBase4i(0, 0, 0, 0),
            "mat3": LMatrix3.ident_mat(), "mat4": LMatrix4.ident_mat(),
        }.get(gtype, 0.0)

    #: GLSL scalar types whose Panda3D shader input must be a Python `int`
    #: (or bool) — the values this class *computes* (`_dynamic_uniforms`/
    #: `_day_cycle_uniforms`) are written assuming BSL's own declared types
    #: (e.g. `worldTime` as a `float`), but Minecraft's real convention (and
    #: what other packs — Complementary confirmed — actually declare) is
    #: `int`. Panda3D's `set_shader_input` doesn't coerce for you: feeding a
    #: Python `float` to a shader's `int`/`uint` uniform is a hard GL error
    #: ("Cannot pass floating-point data to integer shader input"), not a
    #: silent truncation, so a pack that declares a *different* type than
    #: BSL for the same semantic uniform previously broke outright.
    _INT_GLSL_TYPES = frozenset({"int", "uint", "bool"})

    def _coerce(self, value: Any, gtype: str) -> Any:
        """Adapt a computed value to whatever GLSL type *this pack* declared
        the uniform as, rather than assuming BSL's own declared type."""
        if gtype in self._INT_GLSL_TYPES and isinstance(value, (int, float, bool)):
            return int(round(value))
        if gtype == "float" and isinstance(value, (int, bool)):
            return float(value)
        return value

    def _update(self, task: Any) -> Any:
        from direct.task import Task

        dynamic = self._dynamic_uniforms()
        sky_np = getattr(self, "_sky_np", None)
        if sky_np is not None:
            # A skybox must stay centred on the viewer, not the world origin,
            # or the camera would eventually pass through its wall.
            sky_np.set_pos(self.base.render, dynamic["cameraPosition"])
        for node in [self.base.render] + self._quads:
            for name, gtype in self._uniform_types.items():
                value = dynamic.get(name)
                if value is None:
                    value = self._NAMED_DEFAULTS.get(name)
                if value is None:
                    value = self._default_for(gtype)
                node.set_shader_input(name, self._coerce(value, gtype))
        return Task.cont

    # -- inspection (headless) ------------------------------------------
    def describe(self) -> str:
        """A text summary of the resolved pipeline (no GPU needed)."""
        values = self.options.values()
        lines = [
            f"Pack: {self.pack.name}   world: {self.world}",
            f"Buffers: " + ", ".join(
                f"colortex{i}={self.graph.buffers.format_of(i)}"
                for i in self.graph.buffers.indices()),
            "Passes (enabled under current options):",
        ]
        for p in self.graph.enabled_passes(values):
            tag = "geo" if p.kind == "geometry" else "FS"
            outs = f" -> colortex{p.outputs}" if p.outputs else ""
            lines.append(f"  [{p.kind:10}] {p.name:24} {tag}{outs}")
        lines.append(f"Render types available: {', '.join(self.resolver.types())}")
        return "\n".join(lines)

    def debug_textures(self) -> dict[str, Any]:
        """Every ``colortex`` buffer, by name, for on-screen inspection tools.

        Lets a caller (see ``examples/pipeline_demo.py``'s buffer viewer)
        display any intermediate stage directly instead of only the final
        composited frame — the fastest way to confirm a given pass is really
        writing what it should. The shadow depth texture is deliberately
        excluded: it's bound to ``sampler2DShadow`` uniforms with GL's
        depth-compare mode enabled, which doesn't display sensibly through a
        plain textured quad (and toggling that mode to view it would corrupt
        the live shadow pass, since it's the same GPU texture object).
        """
        return {f"colortex{i}": tex for i, tex in sorted(self._colortex.items())}
