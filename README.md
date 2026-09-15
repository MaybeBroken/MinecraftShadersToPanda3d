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
import mcshader

app = mcshader.init()                        # window + pack + sky + fly camera
app.load("models/environment", type="terrain", scale=0.25)
app.load_actor("models/panda-model", {"walk": "models/panda-walk4"},
               type="entity", block="minecraft:sea_lantern", loop="walk")

app.profile = "ULTRA"                        # Iris-style quality profiles
app.option("SHADOW_FILTER", True)            # any option the pack declares
app.swap_pack("SomeOtherPack.zip")           # re-shade everything, tags preserved
app.run()
```

That's the whole program — `init()` creates the Panda3D `ShowBase`, finds a
pack, builds the procedural sky and starts the pipeline. Shading a game you
already have is the same call with your own engine handle:

```python
app = mcshader.init(base)                    # your ShowBase, your scene graph
app.attach(ground, type="terrain")
app.attach(crystal, type="glowing", block="minecraft:sea_lantern")
```

Nothing is hidden behind it: `app.pipe` is the full `PipelineRenderer` and
`app.base` your `ShowBase`, so dropping to the low-level API (below) or to raw
Panda3D is never a dead end.

| | |
|---|---|
| `init(base=None, pack=None, …)` | make (or adopt) the window, load a pack, build the sky, start the pipeline |
| `load` / `load_actor` | load a model or animated `Actor`, place it (`pos`/`hpr`/`scale`), tag it |
| `attach` / `set_type` / `set_block` / `set_light` | shade a node you made yourself, or change one later |
| `load(..., tag=[(regex, type), …])` | bulk-tag a model's sub-parts by naming convention |
| `app.profile` / `app.option(name[, value])` / `app.options` | Iris-style profiles and every option the pack declares |
| `swap_pack` / `reload` / `app.enabled` | re-shade with another pack, recompile, or A/B against plain Panda3D |
| `camera` / `fly_camera` / `go_home` | pose and fly the camera the pipeline reads its matrices from |
| `app.buffers` / `view` / `debug_ui` | inspect raw `colortex` buffers; install the whole dev HUD |
| `key` / `every_frame` / `text` / `screenshot` / `run` | the Panda3D bits you'd otherwise write by hand |

Block ids and lightmaps set through the app are re-applied for you after every
recompile, profile switch and pack swap — the raw runner drops those on a
rebuild.

## Install

```bash
pip install ".[panda3d]"      # the pipeline runner + demo
pip install -e ".[dev]"       # editable, with tests
```

Then, to see it running on real geometry:

```bash
python -m mcshader demo       # the bundled scene, any pack it can find
python -m mcshader demo Shaders/BSL_v10.0.zip --profile ULTRA
```

The core (parsing, options, graph, translation) has **no dependencies**; only the
Panda3D runner needs `panda3d`, and importing `mcshader` never imports Panda3D
until you actually call `init()`.

## The three subsystems

### 1. The deferred pipeline runner

`mcshader.init()` is the front door (see above); `PipelineRenderer` is what it
drives, and what you'd reach for to own the wiring yourself. It loads a pack,
builds its render graph, and runs the passes in the correct order into a shared
`colortex` buffer pool, exactly as the pack declares them:

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

One command there does need a GPU — `python -m mcshader demo [pack]` (same as
`examples/pipeline_demo.py`): the pipeline over Panda3D's own bundled
"environment" island (real ground/rock/tree/bamboo textures) plus an animated
actor, tagged in bulk by name pattern — the same way a real game's assets would
be tagged, not object-by-object. Its source (`mcshader/demo.py`) is also the
worked example for the one-call API, and its on-screen HUD (profile hotkeys,
the pack's live settings panel, pack on/off, the raw-buffer viewer) is one call
you can put on your own app: `app.debug_ui()`.

Inspect a pack headlessly from Python with `examples/pipeline_describe.py`.

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

A fourth pass found a deeper, more consequential bug than any of the above:
**every Minecraft shader assumes world-space Y is up** (confirmed directly in
BSL's own source — `gbuffers_terrain.glsl`'s vertex shader reads
`gbufferModelView[1]` straight out as the view-space "up" vector, and
`shadows.glsl`'s `GetCloudShadow` reads a reconstructed world vector's `.y` as
its vertical component), but this engine's native world is Z-up/Y-forward, and
the `gbufferModelView`/`shadowModelView`-family uniforms were being fed
unconverted Panda matrices. The practical effect: "up" silently landed on a
horizontal axis for every piece of shader math that used it directly — the
sky's vertical falloff ran along the wrong axis (dark bands where there
should be zenith/horizon gradient), the sun/shadow direction was rotated
relative to the real scene (shadows not lining up with objects, appearing to
swing as the camera turned), and every per-object effect keyed off world Y
(foliage waving amplitude, world curvature) was broken. Fixed with a
rotation-only Z-up↔Y-up reconciliation (`PipelineRenderer._cs_conversion`)
applied to the shader-facing `gbufferModelView(Inverse)`/`shadowModelView
(Inverse)`/`cameraPosition`/sun-direction uniforms — Panda's own scene graph
and everything that drives *actual* rendered geometry position
(`p3d_ModelViewMatrix`, the real camera/lens) stay native Z-up throughout and
are provably unaffected (verified numerically: the self-cancelling identity
every BSL vertex shader relies on, `gbufferModelView *
gbufferModelViewInverse * gl_ModelViewMatrix * gl_Vertex`, reduces to exactly
Panda's real per-object transform regardless of this matrix's contents — only
the shaders' own world-space math depends on getting the convention right).
Visually confirmed: the sky now renders a normal gradient at every profile
(previously a solid/axis-inverted dark patch), and terrain/geometry show
coherent directional lighting. **Not separately re-verified this pass:**
whether a ground-contact shadow is now visibly cast under an object — see the
existing shadow-bias caveat below, which predates (and is logically
independent of) this fix.

A fifth pass fixed cross-frame history — colortex2 (BSL's TAA color history,
auto-exposure, lens-flare visibility, and DOF focus, all packed into one
`colortexNClear = false` buffer) was reading back as exactly zero on every
frame, which was the direct cause of the persistent speckled grain at
**HIGH**/**ULTRA** (AO and TAA both lean on this history to denoise their
per-frame-noisy samples) and silently disabled lens flare outright (its own
gating check, `tempVisibleSun`, is read from a corner texel of this same
buffer — permanently zero meant `if (lensFlareFactor > 0.0001)` never ran).
Root cause: `_make_buffer` was unconditionally clearing every render target's
color/aux planes at the start of each frame (added earlier as a real, correct
fix for NaN/Inf propagating from uninitialized float VRAM) — for a genuine
`Clear = false` buffer, that GL clear runs *before* that buffer's own pass
draws, regardless of what the pass's shader later overwrites, so it wiped
frame N's history before frame N+1's earliest reader (`composite5`) ever saw
it. Fixed by making the clear decision per attachment slot
(`self.graph.buffers.clear`), skipping it for `Clear = false` colortex
indices. No other change was needed: `_render_quad`'s existing self-read
ping-pong (for `composite5`/`composite7`, which both sample and write
colortex2 in the same pass — a hazard the pack-wide `_self_read_outputs`
already detects) already gives that a stable, self-sustaining 2-texture
cycle across frames once the clear stopped wiping it — confirmed both by
direct pixel inspection of the write pattern and, more convincingly, by
rendering the same camera position for 3 vs. 120 frames at **ULTRA**: the
speckle noise visibly converges/smooths out instead of staying constant.
Also enables `SHADER_SUN_MOON` (BSL's real sun/moon disc, geometrically
correct against the coordinate fix above — confirmed via screenshot) and
`LENS_FLARE` (on by default in BSL, previously always-inert) to actually
work now that their shared gating buffer holds real data.

A sixth addition: `mcshader/ui/settings_panel.py`'s `SettingsPanel` renders the
pack's *entire* Iris-style options menu (`ShaderOptions.menu_tree` — every
screen/toggle/slider it declares) as a live, navigable DirectGUI panel in the
demo (`[o]` to open) — change a value, watch `set_option()` + `recompile()`
apply it in real time. Pure demo/dev-tool convenience on top of the existing
engine-side options API; touches no pipeline internals. Its panel is docked
against the window's *actual* aspect ratio (`SettingsPanel._reposition`,
re-run on every `window-event`) rather than a fixed literal tuned for a wide
window — the fixed version left roughly a third of the panel clipped off the
right edge of Panda3D's own 800x600 default window.

A seventh pass found the single bug behind four separate symptoms — broken
shadows, screen-space effects not lining up with the camera, and (per report)
"nothing mapping to the camera right, might be FOV": **`gbufferProjection`/
`shadowProjection` were fed Panda's *native* projection matrix, in a
different convention than BSL's own code assumes.** Confirmed numerically:
Panda's `lens.get_projection_mat()` depends on `lens.get_coordinate_system()`
— left at Panda's default (`CS_zup_right`, matching the engine's native
Y-forward/Z-up camera space) it returns a mathematically valid but
differently-*shaped* matrix than the standard OpenGL one (`CS_yup_right`) a
lens with that coordinate system set explicitly returns; verified by
reconstructing the latter as `cs_yup_to_zup * <native matrix>` and checking
it against Panda's own `CS_yup_right` output (exact match, both for
`PerspectiveLens` and the shadow camera's `OrthographicLens`). That
distinction is invisible to code that only ever multiplies the full
matrix — but a good deal of BSL's own code doesn't: `ToNDC`/`ToShadow`'s
`projMAD`/`diagonal3` macros (`lib/util/spaceConversion.glsl`) and
`GetLinearDepth` (used throughout AO, light shafts, outlines, and
`shadows.glsl`'s `GetShadow`) all read specific *cells* of the projection
matrix assuming the standard sparse layout — Panda's native matrix puts the
same terms in different cells, so those shortcuts silently read zeros
instead of the real depth/aspect terms. Fixed by reconstructing the
standard-convention matrix (`cs_yup_to_zup * <native>`) for both
`gbufferProjection` and `shadowProjection`, and correspondingly folding an
extra `cs_yup_to_zup` rotation into `gbufferModelView`/`shadowModelView` so
both halves of every `projection * modelview` pair agree on the same "view
space" (previously they silently didn't, even though each half looked
internally plausible on its own). Confirmed by rendering the real demo scene
offscreen on the actual test hardware, before and after, at the same camera
pose: terrain/rocks/sky went from solid black (the long-standing
unexplained "black pillar"/no-sky/no-shadow symptom) to correctly lit with a
clearly visible cast shadow next to a rock cluster, matching the sun
direction. The `MEDIUM` profile (shadows + AO + `SHADOW_FILTER`, no
`LIGHT_SHAFT`/`TAA`) is the cleanest demonstration of this fix.

The same pass fixed a second, independent bug behind the persistent
"static"/noisy look: **`gbufferPreviousModelView`/`gbufferPreviousProjection`/
`previousCameraPosition` were fed *this* frame's values, not last frame's.**
`taa.glsl`'s `Reprojection()` (and the multicolored-blocklight/SSR code that
reuses the same pattern) computes `cameraOffset = cameraPosition -
previousCameraPosition` and reprojects through the *previous* frame's
matrices to find where a world point was on screen a frame ago; feeding it
identical current/previous values made `cameraOffset` permanently zero and
the reprojection a no-op regardless of real camera motion, so temporal
blending mixed each new frame with a history texel that (whenever the camera
had actually moved) represented a different world point. `PipelineRenderer`
now caches the previous call's `gbufferModelView`/`gbufferProjection`/
`cameraPosition` and feeds those, one frame behind, instead of duplicating
the current frame's values.

**A newly-found, separate, pre-existing bug, not fixed this pass:** with
`TAA` on (the default at `HIGH`/`ULTRA`), `composite7`'s `TemporalAA()`
renders `terrain`/`block_entity`-tagged geometry solid black while
`entity`-tagged geometry (foliage, actors) stays correctly lit — confirmed
present before the projection-matrix fix too (bisected against a stashed
pre-fix build), so it isn't a regression from the work above. Bisected by
selectively disabling each option `HIGH` adds over a verified-good `MEDIUM`
baseline (`AO`, `SHADOW_FILTER`, `LIGHT_SHAFT`, `SHADOW_COLOR`, `TAA`
individually): only `TAA` reproduces it. Further bisected *inside*
`TemporalAA()` by patching the compiled shader to bypass pieces of it one at
a time: forcing `prvCoord = texCoord` (skipping `Reprojection()` entirely)
does **not** fix it, and outputting the raw `textureCatmullRom(colortex2,
prvCoord, view)` sample directly shows it's already black *before*
`NeighbourhoodClipping`/`ClipAABB` or the final blend ever run — so the bug
is that `colortex2`'s persisted TAA-history channels (`.gba`) are
genuinely, persistently black for these pixels, and `ClipAABB` isn't
correcting it back into the current frame's valid color range the way its
own logic implies it should once contaminated. Leading unconfirmed
hypothesis: a NaN entering `colortex2` on an early frame (composite5's
`temporalData = mix(sqrt(exposure), tempExposure, ...)` computing
`sqrt` of a negative intermediate is one plausible source) would explain
every observation — it fails the `tempColor == vec3(0.0)` fallback check
(`NaN != 0.0`), and `ClipAABB`'s own `if (ma_unit > 1.0)` guard is false for
a NaN `ma_unit` (all comparisons with NaN are false), so it falls through to
`return q` and passes the contamination straight through unclipped, forever
self-sustaining in the persistent buffer — but this was not confirmed before
the investigation budget for this pass ran out. Until fixed, prefer
`MEDIUM` profile (or `pipe.set_option("TAA", False); pipe.recompile()`) over
`HIGH`/`ULTRA`.

**Cross-pack portability: first real bugs found testing a second pack.**
Everything above was validated against BSL only; pointed at Complementary
Unbound (`Shaders/ComplementaryUnbound.zip`) for the first time, the runner
crashed outright. Three genuine, BSL-blind-spot bugs, all fixed in
`mcshader/glsl/dialect.py` and `panda3d_pipeline.py`, none of them
Complementary-specific hacks:

- **OptiFine/Iris "uniform with a fallback" initializers weren't stripped.**
  `uniform bool heavyFog = false;` is legal in a pack's own source (the value
  is a default for when the engine doesn't recognise the name) but not in
  real GLSL; left in place, our own uniform recorder captured `"heavyFog =
  false"` as a name instead of `heavyFog`, so nothing ever bound it and
  Panda3D asserted `Shader input heavyFog is not present`. `dialect.py` now
  strips the initializer from any `uniform TYPE NAME = ...;` declaration
  before translation.
- **`gl_Fog.*` (legacy fixed-function fog state) had no translation.**
  BSL never uses it; Complementary reads `gl_Fog.start/.end/.density/.color`
  to blend with vanilla fog, and it has no core-profile declaration, so the
  fragment shader failed to compile (`gl_Fog undeclared`). `dialect.py` now
  rewrites each member to the real Minecraft uniform it mirrors
  (`fogStart`/`fogEnd`/`fogDensity`/`fogColor`), injecting a declaration
  only for whichever of those the pack hadn't already declared itself
  (`fogColor` usually was; the others never had a reason to be).
- **Computed uniform values weren't coerced to the pack's own declared
  type.** `_day_cycle_uniforms` produces `worldTime` as a Python `float`
  (fine for BSL, which declares it `float`); Complementary declares it
  `int` (Minecraft's actual convention), and Panda3D's `set_shader_input`
  errors outright on a type mismatch rather than truncating
  (`Cannot pass floating-point data to integer shader input`).
  `PipelineRenderer._coerce` now casts a computed value to match whatever
  GLSL type *this* pack actually declared before feeding it, instead of
  assuming BSL's declared type is universal.

A follow-up pass, asked to make the *whole* pack work (not just stop
crashing), found the actual reason its render looked flat/wrong even after
the above: **`shaders.properties`' own `#if`/`#else`/`#endif` preprocessor
blocks were never evaluated.** BSL's `shaders.properties` never uses them, so
this was invisible; every line was read unconditionally (`#` was treated as
a plain end-of-line comment, same as vanilla `.properties`). Complementary
gates several `program.*.enabled` lines this way — critically
`#if SHADOW_QUALITY == -1 / program.world0/shadow.enabled=false / #endif` —
and reading that unconditionally permanently disabled the shadow program
regardless of `SHADOW_QUALITY`'s real value (2 by default, not -1). Two
fixes, both in `mcshader/pack/properties.py`:

- `eval_condition` (the same evaluator `program.*.enabled` expressions
  already used) gained comparison operators (`==`/`!=`/`<`/`>`/`<=`/`>=`)
  against numeric option values — BSL's own `program.*.enabled` expressions
  never needed more than `&&`/`||`/`!` over boolean toggles, but a
  `.properties`-file-level `#if` is a richer grammar. Moved here from
  `pipeline/graph.py` (which now imports it back) so `pack/` doesn't have to
  depend on `pipeline/` to preprocess its own conditional blocks.
- `parse_properties`/`read_pairs` now take the pack's current option values
  and resolve `#if`/`#ifdef`/`#ifndef`/`#else`/`#endif` blocks before parsing
  key=value lines. Re-evaluated on every rebuild (`PipelineRenderer.recompile`
  now re-parses `shaders.properties` with the live option values, not just
  once at load), matching OptiFine/Iris re-resolving these on every shader
  reload — so flipping the relevant option later (`SHADOW_QUALITY` back to
  `-1`, say) correctly disables the program again, not just at startup.

Also fixed along the way: `_build_shadow_pass` compiled the `shadow` program
twice — once correctly with `mode="gbuffer"` just to check it exists, then
again with the *default* mode (`"compact"`, meant for fullscreen quads, not
real scene geometry) for the shader actually bound to the shadow camera.
Harmless for BSL (`shadow.glsl` only ever writes `gl_FragData[0]`, where both
modes number identically) but wrong in general and wasteful either way; now
compiled once and reused. And two demo GUI bugs
assuming every pack has a `SHADOW` toggle option by that exact name
(Complementary gates shadows on the numeric `SHADOW_QUALITY` instead): `[z]`
(toggle shadow) raised a bare `KeyError`, and the on-screen status HUD had
lost its `setText` call entirely (computed the status string, never
displayed it) — both fixed to degrade to `"n/a"` for a pack without that
option instead of crashing/silently doing nothing.

**Still open, and worse than first assessed: an intermittent GL link failure
(`vertex shader lacks 'main'`) that takes the whole app down**, not a benign
warning — confirmed directly on real hardware with the actual
`examples/pipeline_demo.py` window (not a headless script): the window opens
and renders, then dies with no Python traceback partway through the first
run, roughly half the time, not tied to any one program or profile found so
far. One real, general bug fixed while chasing it — `resolve_includes`
(`mcshader/glsl/preprocess.py`) only broke *cycles* (A includes B includes
A), not the far more common "diamond" case (two unrelated files both
`#include` the same shared `lib/` file), so a file could get textually
re-expanded many times over through a deep include graph; confirmed this
inflated Complementary's combined sources to 10,000+ lines each. Real
Iris/OptiFine `#include` expands each file at most once per program (like
`#pragma once`); `resolve_includes` now does too (tracks every path already
expanded anywhere in the call tree, not just on the current branch),
shrinking `gbuffers_terrain` from ~10,460 to ~8,830 lines and `composite`
from ~12,380 to ~9,730 — real, and possibly relevant to a compiler choking on
an oversized source, but **not confirmed to be the fix**: the crash still
needs to be reproduced and pinned to a specific program before it can be
called fixed. Whoever picks this up next: don't trust a single run either
way (it's intermittent) — reproduce it a few times first, then bisect by
disabling one program's compile at a time (`PipelineRenderer._compile`) to
find which one, before assuming any particular cause.

## Demo GUI fixes (pack on/off toggle + button-interaction bugs)

Went through the demo and `SettingsPanel` (then `examples/pipeline_demo.py`
and `examples/_settings_panel.py`; now `mcshader/demo.py` and `mcshader/ui/`)
looking for buttons/hotkeys that don't do what they claim, and added the
requested pack on/off toggle:

- **New `[y]` hotkey and `PipelineRenderer.set_enabled(bool)`** — toggles
  between the pack's full shaded output and Panda3D's own plain rendering of
  the *same* tagged scene, live, without reloading anything. The offscreen
  gbuffer/composite chain keeps running underneath either way; disabling
  just hides the final composited quad(s) (so the window falls through to
  Panda's own default display region) and clears every tagged NodePath's
  shader (since that default display region draws the same scene graph, so
  what's actually on the geometry is what determines its look there).
  `set_render_type` now also respects a standing `set_enabled(False)`, so a
  rebuild (option change, profile switch, `swap_pack()`) while the pack is
  toggled off doesn't silently re-shade things.
- **`[1]`-`[5]` profile keys were compiling everything twice.**
  `PipelineRenderer.apply_profile()` already calls `recompile()` internally;
  the demo's `_profile()` called it *again* right after — every profile
  hotkey rebuilt every gbuffers/composite program and every buffer twice for
  no reason. Also deleted a stale comment left over from a since-removed
  `SHADER_SUN_MOON` override it no longer described.
- **The settings panel's own profile stepper (`< >` on the "profile:" row)
  desynced the HUD.** Changing profile through the panel updates
  `pipe.options` correctly but never told the demo, so the on-screen
  `profile=` label kept showing whatever was last set via `[1]`-`[5]` (or
  the startup default) regardless of what was actually applied. Added a
  `SettingsPanel.profile_name` property and now resync the demo's own
  `_profile_name` from it on every settings-panel change.
- **`[p]` (`swap_pack()`) silently reset the profile.** `load_pack()` was
  called with no `profile` argument on a pack swap, dropping back to the new
  pack's raw per-option defaults regardless of what profile (MINIMUM..ULTRA)
  was active — "tags survive" was true, the profile selection wasn't, and
  nothing in the GUI reflected the reset. `PipelineRenderer` now tracks its
  own `profile_name` (set on load/`apply_profile`) and `swap_pack()` passes
  it through.

Complementary's actual visual *fidelity* (it does voxel GI and ships
compute-shader programs — `.csh` —
this pipeline has no compute-dispatch machinery for at all, so anything
depending on those degrades to whatever fallback its shaders happen to fall
back to) has not been assessed — that needs a human actually looking at the
live GUI, not a headless screenshot diff.

Remaining gaps, none of them wiring bugs — genuine fidelity/feature work:

- **`shaders.properties`' custom-uniform DSL is hand-implemented, not
  evaluated generally.** `PipelineRenderer._day_cycle_uniforms` computes
  `worldTime` (animated), `timeAngle`, `shadowFade`, `timeBrightness`,
  `framemod8`/`framemod2` by copying this pack's specific formulas — a
  real, verified fix (confirmed `shadowFade=1.0`/`timeBrightness=1.0` at
  simulated noon). A general expression evaluator for arbitrary packs is
  still not implemented.
- **Entities/foliage were losing their real texture — fixed.** BSL's
  `gbuffers_entities(_glowing)` does `albedo.rgb = mix(albedo.rgb,
  entityColor.rgb, entityColor.a)`; the runner's generic default for an
  unbound `vec4` uniform is `(0,0,0,1)`, and alpha=1 there means "always
  fully overridden by black". Every `entity`-tagged object — foliage,
  actors, anything not `terrain`/`textured`/`block_entity` — was losing its
  texture to this. Fixed with a per-name default override
  (`entityColor` → `(0,0,0,0)`, "no tint"). Confirmed: bamboo, tree
  branches, and an animated actor all render with their real textures now.
- **Shadow ground-contact visibility — confirmed working.** The
  `GetShadow()`/`GetLinearDepth` "fully occluded everywhere" symptom
  previously suspected to be a bias problem was actually the projection-matrix
  convention bug described in the seventh pass above (`ToShadow`'s
  `projMAD`/`diagonal3` shortcut was reading the wrong matrix cells). Fixed
  there; visually confirmed via an on-hardware screenshot showing a clear
  cast shadow next to a rock cluster.
- **Sky now renders** (`PipelineRenderer.build_sky()` attaches a
  camera-following sphere tagged `sky_basic`; BSL's `gbuffers_skybasic.glsl`
  computes the gradient/sun/moon/stars/aurora entirely from the
  screen-space ray direction, so the geometry only needs to cover the sky).
  **Clouds are not implemented** — BSL's cloud programs need real vertex
  UVs/normals/a cloud texture, unlike the sky's purely-procedural approach.
- **No exposure/tone-mapping tuning.** HDR sky content can clip to solid
  white; a look at BSL's exposure curve would go a long way.
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
  ui/          fly camera, bulk name-pattern tagging, the live settings panel
  app.py       the one-call facade: init() -> ShaderApp (load/attach/profile/view/debug_ui)
  demo.py      the bundled demo scene (python -m mcshader demo)
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
