# MinecraftShaderDecompiler

Run a **Minecraft shaderpack's entire deferred pipeline** — shadows, gbuffers,
deferred lighting, composite chain, final — inside an OpenGL game engine, and
**re-shade your whole game by swapping the pack**. Tag your engine objects with
Minecraft *render types* (and block ids, even for custom items); drive every
feature through an **Iris-style options system** (profiles, sliders, toggles)
parsed straight from the pack.

Primary engine: **Panda3D**. The pack-parsing, options, render-graph and GLSL
translation layers are engine-independent (and fully unit-tested without a GPU),
so other OpenGL backends can be added.

```python
from mcshader.engine import PipelineRenderer

pipe = PipelineRenderer(base, "Shaders/", world="world0", profile="HIGH")

pipe.set_render_type(ground, "terrain")     # each object renders as an MC type
pipe.set_render_type(water,  "water")
pipe.set_render_type(crystal, "glowing")
pipe.set_block_id(crystal, pipe.resolver.block_id("minecraft:sea_lantern"))

pipe.apply_profile("ULTRA")                  # Iris-style quality profiles
pipe.set_option("SHADOW_FILTER", True); pipe.recompile()
pipe.swap_pack("SomeOtherPack.zip")          # re-shade everything, tags preserved
```

## Install

```bash
pip install -e ".[panda3d]"   # the pipeline runner + demo
pip install -e ".[dev]"       # tests
```

The core (parsing, options, graph, translation) has **no dependencies**; only the
Panda3D runner needs `panda3d`.

## The three subsystems

### 1. The deferred pipeline runner

`PipelineRenderer` loads a pack, builds its render graph, and runs the passes in
the correct order into a shared `colortex` buffer pool, exactly as the pack
declares them:

```
shadow map ─▶ opaque gbuffers ─▶ deferred* ─▶ translucent gbuffers ─▶ composite* ─▶ final
```

- **Object tagging is the API.** You never name a program. `set_render_type(np,
  "terrain")` resolves to the pack's `gbuffers_terrain` (with OptiFine's fallback
  chain — a pack with no `gbuffers_terrain` falls back to `gbuffers_textured` →
  `gbuffers_basic`). Custom items just choose a render type.
- **Block ids** let a custom object render the way the pack renders a given block
  (waving, emissive, material), via `set_block_id`.
- **Per-pass output routing** is read from each program's `/* DRAWBUFFERS:0195 */`
  directive; buffer formats/clear flags from `const int colortexNFormat` /
  `colortexNClear`. Attachment `i` ↔ `colortex i` throughout.

### 2. The Iris-style options manager

Everything a pack exposes in its in-game settings screen is parsed into data you
can read and change:

```python
from mcshader import load_pack, ShaderOptions
pack = load_pack("Shaders/")
props = pack.properties()
opts = ShaderOptions.from_pack(pack.option_sources(), props)

opts.apply_profile("HIGH", props)                 # profiles (with inheritance)
opts.set("shadowMapResolution", "4096")           # validated against allowed values
tree = opts.menu_tree(props)                       # full screen tree for a GUI
opts.dumps() / opts.loads(text)                    # option-file round-trip
```

Changing an option rewrites the pack's GLSL (`#define`/`const` values, toggle
comment/uncomment) — the same mechanism OptiFine/Iris use — and the runner
recompiles. An interactive on-screen menu is a thin layer on top of this data
(planned; the data/API ships now).

### 3. Pack loading + GLSL translation

Loads a pack from a folder or `.zip`, and translates each program's legacy GLSL
(`attribute`/`varying`, `gl_Vertex`, `texture2D`, combined `#ifdef VSH/FSH`
files with `#include`s) into modern `#version 150` GLSL Panda3D can compile, with
fragment outputs routed to the right `colortex` attachments.

## Command line (all GPU-free)

```bash
python -m mcshader pipeline Shaders/ --profile HIGH   # resolved pass order + buffers + render types
python -m mcshader options  Shaders/ --screen LIGHTING # the options menu tree with values
python -m mcshader list     Shaders/                   # programs in the pack
python -m mcshader show     Shaders/ gbuffers_water --stage fragment
python -m mcshader extract  Shaders/ gbuffers_terrain --out ./out
```

Inspect a pack headlessly from Python with `examples/pipeline_describe.py`, or run
the live demo with `examples/pipeline_demo.py` (needs `panda3d` + a display).

## Honest limitations

This is a **faithful, incrementally-complete** port, not a bit-exact
reimplementation of OptiFine/Iris. The parsing/options/graph/translation layers
are solid and tested. The GPU runner has been validated on real hardware
(Apple M1 / Mesa, OpenGL 4.6 core): **all of BSL's world0 programs translate and
compile, and the pipeline renders end-to-end** (geometry → gbuffers → deferred →
composite → final) to the window, and a second hardware pass fixed three
mechanical wiring bugs that were the difference between "runs" and "renders
something recognisable":

- **Fullscreen passes now bind every `DRAWBUFFERS` target, not just the
  first.** A pass like `deferred1` (writes colortex0/4/5/6) or `composite`
  (colortex0/1/5/9) used to only ever get its first output attached to the
  FBO — everything past it silently never got written.
- **The shadow map is a real depth texture, rendered from the sun's
  direction.** The shadow camera used to sit at the world origin with an
  identity rotation (not aimed at the scene, not aimed along the light
  direction shaders were told), and its buffer was a colour texture bound to
  packs' `sampler2DShadow` uniforms — a meaningless comparison. It's now an
  actual depth attachment (`Texture.FT_shadow` compare mode) with the camera
  pointed down the same sun direction fed as `sunPosition`/`shadowLightPosition`,
  and `shadowModelView`/`shadowProjection` are now that camera's real matrices
  instead of a copy of the main camera's.
- **Self-reading composite passes ping-pong instead of feeding back on
  themselves.** BSL refines one buffer (colortex1) across five composite
  passes, each sampling the previous pass's result while writing a new one.
  Sampling and rendering into the *same* texture in one draw call is an
  undefined GL feedback loop — on this hardware it produced a speckled-noise
  artifact instead of the intended image. Passes whose `DRAWBUFFERS` output
  overlaps their own declared input samplers now render into the already-
  allocated ping-pong back buffer and swap it to the front for the next pass.

A third pass, prompted by real on-hardware testing that flagged the render as
still looking "corrupted," found two more bugs that turned out to matter more
than all three above combined:

- **Every render-to-texture buffer was silently padded to the next
  power-of-two by Panda3D** (e.g. a 960×540 window → a 1024×1024 texture,
  with only the top-left 960×540 corner ever actually rendered into). The
  translated shaders sample `colortexN`/`shadowtexN` with plain `[0,1]` UV —
  there's no auto texture-matrix correction for hand-written GLSL — so every
  pass reading a previous pass's output sampled a squashed, offset copy of
  it, compounding across the composite chain into the "nested rectangle"
  corruption. Fixed with one config line (`textures-power-2 none`) set before
  any buffer exists; modern GL has no power-of-two restriction, so there's no
  downside.
- **The per-frame uniform code read `base.camera` instead of `base.cam`.**
  These are different nodes in Panda3D (`cam` is a child of `camera`, which
  sits at the origin unless something else moves it), and every caller poses
  the camera via `base.cam`. Rasterization was always correct, but
  `gbufferModelView`/`cameraPosition`/fog/sky-direction were computed as if
  the camera never left the origin — so moving the camera changed what was
  on screen but never changed the shading.

With all five fixed, a render is now a genuinely coherent scene — correct
perspective convergence at the horizon, a distinctly-colored water plane, a
recognizable object silhouette — not an incremental improvement on the noise.
BSL specifically still looks rough end-to-end under its default
**HIGH**/**ULTRA** profiles, because BSL leans hard on features this runner
doesn't have yet (this part *is* fidelity work, not more wiring bugs):

- **No cross-frame history.** BSL's `composite5` explicitly treats colortex2 as
  "temporal data" (reprojected screen-space reflections/volumetrics) — it
  expects last *frame's* buffer, not just last *pass's*. The runner has no
  concept of frame-to-frame buffer persistence yet, so these effects read
  garbage. The **MINIMUM** profile avoids most of this and looks noticeably
  cleaner — prefer it until history buffers land.
- **No custom-uniform evaluator.** `shaders.properties` defines derived
  uniforms (`timeBrightness`, `shadowFade`, biome flags, …) as a small
  expression DSL; only the literal MC uniforms are fed, so anything pack-side
  computed this way silently defaults to 0. This is the single biggest lever
  left for lighting fidelity/day-night correctness — and its effect is now
  directly visible: BSL's `terrain`/`entity` gbuffer programs run the full
  `GetLighting()` PBR path, whose inputs (lightmap, sky/sun color) sum to
  zero under the current defaults, so those objects render solid black;
  `textured`/`block_entity` skip that path and render correctly (checkered,
  lit). Run `examples/pipeline_demo.py` and compare its four differently-
  tagged pillars to see this directly.
- **No exposure/tone-mapping tuning.** HDR sky content clips to solid white
  and unlit surfaces read near-black in the demo scene; real per-object
  textures/materials (the demo uses flat-shaded primitives) and a look at
  BSL's exposure curve would go a long way.
- **Gbuffer attachment count.** Panda3D binds at most 1 colour + 4 aux render
  targets, so the runner packs the distinct gbuffer `colortex` into ≤5
  attachment slots; a pack whose gbuffers write more than 5 distinct buffers
  has the extras dropped (their writes are discarded).
- **Per-vertex `mc_Entity`.** Block ids are fed as a uniform approximation, not
  a per-vertex attribute, until geometry-attribute wiring lands.
- **Not yet handled:** Distant-Horizons (`dh_*`) programs, TAA jitter/motion
  vectors, and some OptiFine-specific semantics. The architecture adds these
  incrementally rather than by rewrite.

## Simple per-object effects (secondary mode)

A lighter path also exists for when you don't want the whole pipeline — hand-authored
standalone effects (`glow`, `reflection`, `waving`, `movement`) applied to one object:

```python
from mcshader.engine import Panda3DAdapter
fx = Panda3DAdapter(base)
fx.apply(sword, "glow", u_glow_color=(0.2, 0.9, 1.0))
```

See `mcshader.effects` and `python -m mcshader effects`.

## Layout

```
mcshader/
  glsl/        preprocess (includes, VSH/FSH), dialect translator, uniform catalog, fullscreen
  pack/        loader (folder/zip), properties (shaders/block/item/entity)
  config/      options: discovery, profiles, source rewriting, menu tree, option file
  pipeline/    graph (passes/buffers/DRAWBUFFERS), rendertypes (+block ids), translate
  engine/      PipelineRenderer (the runner) + Panda3DAdapter/GenericGLAdapter (simple mode)
  decompiler.py, registry.py, __main__.py (CLI)
examples/      pipeline_demo.py, pipeline_describe.py, list_pack_effects.py, panda3d_demo.py
tests/         pytest — parsing, options, graph, translation, integration (no GPU)
Shaders/       bundled BSL v10 pack used by examples/tests
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
