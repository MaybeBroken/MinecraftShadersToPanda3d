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
        self._colortex: dict[int, Any] = {}
        self._colortex_back: dict[int, Any] = {}
        self._task_started = False
        self.load_pack(pack_path, world=world, profile=profile)

    # -- pack / options lifecycle ---------------------------------------
    def load_pack(self, pack_path: str, *, world: str | None = None,
                  profile: str | None = None) -> None:
        """Load (or reload) a shaderpack and rebuild everything."""
        self.pack = ShaderPack.from_path(pack_path)
        self.world = world or self.world
        self.props = self.pack.properties()
        self.options = ShaderOptions.from_pack(self.pack.option_sources(), self.props)
        if profile:
            self.options.apply_profile(profile, self.props)
        self.graph: PipelineGraph = build_graph(self.pack, self.world)
        self.resolver = RenderTypeResolver(self.pack)
        self._build()

    def swap_pack(self, pack_path: str) -> None:
        """Re-shade the whole scene with a different pack, keeping tags."""
        tags = list(self._geometry)
        self._teardown()
        self.load_pack(pack_path)
        for nodepath, render_type in tags:
            self.set_render_type(nodepath, render_type)

    def apply_profile(self, name: str) -> None:
        self.options.apply_profile(name, self.props)
        self.recompile()

    def set_option(self, name: str, value: object) -> None:
        self.options.set(name, value)

    def recompile(self) -> None:
        """Rebuild shaders/passes after option changes."""
        tags = list(self._geometry)
        self._teardown()
        self.graph = build_graph(self.pack, self.world)
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
        shader = self._compiled_geometry.get(program)
        if shader is not None:
            nodepath.set_shader(shader)
        nodepath.set_shader_input("mcEntityId", 0)
        self._geometry.append((nodepath, render_type))

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

    def _make_buffer(self, name: str, n_color: int, want_depth: bool,
                     sort: int = -10) -> Any:
        """Create an offscreen MRT buffer with ``n_color`` color attachments.

        ``sort`` orders passes: more-negative renders earlier, so the gbuffer
        renders before composites, which render before the window.
        """
        from panda3d.core import (FrameBufferProperties, GraphicsPipe,
                                   GraphicsOutput, WindowProperties)

        fb = FrameBufferProperties()
        fb.set_rgba_bits(16, 16, 16, 16)
        fb.set_float_color(True)
        fb.set_aux_rgba(max(0, n_color - 1))
        if want_depth:
            fb.set_depth_bits(24)
        win = self.base.win
        buf = self.base.graphicsEngine.make_output(
            win.get_pipe(), name, sort, fb, WindowProperties.size(
                win.get_x_size(), win.get_y_size()),
            GraphicsPipe.BF_refuse_window, win.get_gsg(), win)
        self._buffers.append(buf)
        return buf

    def _bind_scene_attachments(self, buf: Any) -> None:
        """Bind each packed gbuffer colortex to its attachment slot."""
        from panda3d.core import GraphicsOutput

        slots = [GraphicsOutput.RTP_color] + [
            getattr(GraphicsOutput, f"RTP_aux_rgba_{i}") for i in range(4)
        ]
        for colortex, slot in sorted(self._gbuffer_map.items(), key=lambda kv: kv[1]):
            if slot < len(slots) and colortex in self._colortex:
                buf.add_render_texture(
                    self._colortex[colortex], GraphicsOutput.RTM_bind_or_copy, slots[slot])

    def _build_scene_target(self) -> None:
        """Point the main camera at the gbuffer MRT instead of the window."""
        n_targets = max(1, len(self._gbuffer_map))
        # Render the gbuffer before every composite pass.
        self._scene_buf = self._make_buffer(
            "mcshader-gbuffer", n_targets, want_depth=True, sort=-100)
        self._scene_buf.set_clear_color((0, 0, 0, 1))
        self._bind_scene_attachments(self._scene_buf)
        dr = self._scene_buf.make_display_region()
        dr.set_camera(self.base.cam)

    def _sun_direction(self) -> Any:
        """World-space direction from the scene toward the sun.

        No day/night cycle is modelled yet (see README), so this is a fixed
        mid-morning angle shared by the shadow camera and the per-frame
        sunPosition/shadowLightPosition uniforms, so shadows actually point
        the way the lighting says they should.
        """
        from panda3d.core import LVecBase3

        d = LVecBase3(0.3, 0.6, 0.75)
        d.normalize()
        return d

    def _build_shadow_pass(self) -> None:
        from panda3d.core import (Camera, OrthographicLens, NodePath, GraphicsOutput,
                                  FrameBufferProperties, GraphicsPipe, WindowProperties)

        if self._compile("shadow", mode="gbuffer") is None:
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
        self._buffers.append(self._shadow_buf)
        dist = float(self.options.get("shadowDistance")) if "shadowDistance" in self.options.options else 256.0
        lens = OrthographicLens()
        lens.set_film_size(dist, dist)
        lens.set_near_far(1.0, dist * 2.0)
        cam = Camera("mcshader-shadow-cam", lens)
        shadow_shader = self._compile("shadow")
        if shadow_shader is not None:
            from panda3d.core import ShaderAttrib, RenderState
            cam.set_initial_state(
                RenderState.make(ShaderAttrib.make(shadow_shader)))
        self._shadow_cam = self.base.render.attach_new_node(cam)
        # Point the shadow camera down the sun direction at the scene origin so
        # its depth map — and the shadowModelView/Projection uniforms derived
        # from it — actually match the lighting direction shaders are fed.
        self._shadow_cam.set_pos(self._sun_direction() * (dist * 0.5))
        self._shadow_cam.look_at(0, 0, 0)
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
        buf = self._make_buffer(f"mcshader-{p.name}", n_color=n_color, want_depth=False,
                                sort=-50 + order)
        slots = [GraphicsOutput.RTP_color] + [
            getattr(GraphicsOutput, f"RTP_aux_rgba_{i}") for i in range(4)
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
        shadow = getattr(self, "_shadow_tex", None) or self._fallback_tex()
        for s in ("shadowtex0", "shadowtex1", "shadowcolor0", "shadowcolor1"):
            bind(s, shadow)
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
        mv = render.get_mat(cam_np)              # world -> view
        mv_inv = cam_np.get_mat(render)          # view -> world
        proj = LMatrix4(lens.get_projection_mat())
        proj_inv = LMatrix4(proj); proj_inv.invert_in_place()
        cam_pos = cam_np.get_pos(render)
        sun_world = self._sun_direction()
        sun_view = mv.xform_vec(sun_world) * 100.0
        up_view = mv.xform_vec(LVecBase3(0, 0, 1)) * 100.0

        shadow_cam_np = getattr(self, "_shadow_cam", None)
        if shadow_cam_np is not None:
            s_mv = render.get_mat(shadow_cam_np)
            s_mv_inv = shadow_cam_np.get_mat(render)
            s_proj = LMatrix4(shadow_cam_np.node().get_lens().get_projection_mat())
            s_proj_inv = LMatrix4(s_proj); s_proj_inv.invert_in_place()
        else:
            s_mv, s_mv_inv, s_proj, s_proj_inv = mv, mv_inv, proj, proj_inv
        return {
            "frameTimeCounter": t, "frameTime": clock.get_dt(),
            "frameCounter": int(clock.get_frame_count()),
            "viewWidth": w, "viewHeight": h, "aspectRatio": w / max(h, 1.0),
            "near": lens.get_near(), "far": lens.get_far(),
            "sunAngle": 0.25, "timeAngle": 0.25, "rainStrength": 0.0, "wetness": 0.0,
            "cameraPosition": cam_pos, "previousCameraPosition": cam_pos,
            "gbufferModelView": mv, "gbufferModelViewInverse": mv_inv,
            "gbufferPreviousModelView": mv,
            "gbufferProjection": proj, "gbufferProjectionInverse": proj_inv,
            "gbufferPreviousProjection": proj,
            "shadowModelView": s_mv, "shadowModelViewInverse": s_mv_inv,
            "shadowProjection": s_proj, "shadowProjectionInverse": s_proj_inv,
            "sunPosition": sun_view, "moonPosition": sun_view * -1.0,
            "shadowLightPosition": sun_view, "upPosition": up_view,
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

    def _update(self, task: Any) -> Any:
        from direct.task import Task

        dynamic = self._dynamic_uniforms()
        for node in [self.base.render] + self._quads:
            for name, gtype in self._uniform_types.items():
                value = dynamic.get(name)
                if value is None:
                    value = self._default_for(gtype)
                node.set_shader_input(name, value)
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
