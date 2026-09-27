# Minecraft Shader Decompiler

Run a **Minecraft shaderpack's entire deferred pipeline** (shadows, gbuffers,
deferred lighting, composite chain, final) inside an OpenGL game engine, and
re-shade your whole game by swapping the pack.

You tag your objects with Minecraft *render types* (`terrain`, `entity`, …) and
the pack does the rest. Quality profiles and every option the pack exposes are
driven through an Iris-style options system, live.

Primary engine is **Panda3D**; the pack parsing, options, render-graph and GLSL
translation layers are engine-independent.

## What it looks like

Panda3D's bundled "environment" scene, same camera, same geometry, rendered by
Panda3D alone, then by BSL v10 through this pipeline at the `MEDIUM` profile:

| Without | With |
|---|---|
| ![Plain Panda3D rendering](docs/images/without-shaders.jpg) | ![The same scene through BSL v10](docs/images/with-shaders.jpg) |

## Install

```bash
pip install ".[panda3d]"
```
OR run the build.py program with the --install flag

## Demo

```bash
python -m mcshader demo
```

That opens the scene above. `WASD`/`QE` to fly, `[1]`-`[5]` to switch quality
profile, `[o]` for the pack's live settings panel, `[y]` to toggle the pack off
and back on.

## Use it

```python
import mcshader

app = mcshader.init()                        # window + pack + sky + fly camera
app.load("models/environment", type="terrain", scale=0.25)
app.load_actor("models/panda-model", {"walk": "models/panda-walk4"},
               type="entity", loop="walk")

app.profile = "MEDIUM"
app.run()
```

To shade a game you already have, hand `init()` your own `ShowBase` and tag the
nodes you already made:

```python
app = mcshader.init(base)
app.attach(ground, type="terrain")
app.attach(crystal, type="glowing", block="minecraft:sea_lantern")
```

`app.pipe` is the full pipeline renderer and `app.base` your `ShowBase`, so
nothing is a dead end.

## VR

With [Panda3d-VR](https://github.com/MaybeBroken/Panda3d-VR) installed (`pip install -e path/to/Panda3d-VR`), pass its `VRManager` and the pack renders in the headset:

```python
from panda3d_vr import BaseVrApp
import mcshader

class Game(BaseVrApp):
    def __init__(self):
        super().__init__(msaa=0)            # the pack does its own anti-aliasing
        self.shaders = mcshader.init(self, pack="BSL_v10.0.zip", profile="MEDIUM", vr=self.vr)
```

How it works:

- **Headset connected.** Each eye runs the pack's whole pipeline from its own camera: gbuffers, deferred lighting, composites and TAA history. The compiled programs, the shadow map and the sky are shared between the eyes.
- **Handover to the headset.** Each eye's final image, plus its float depth for the compositor's reprojection, is handed to the headset on the GPU.
- **No headset.** The window keeps rendering through Panda3d-VR's desktop simulator. Connecting a headset later switches over automatically.
- **Camera.** `app.camera(near=..., far=...)` sets the headset's clip planes. The FOV and camera pose come from the headset.

`examples/vr_demo.py` is a runnable version.

Underneath is `PipelineView`, which works with any number of cameras (not just VR):

```python
from mcshader.engine.panda3d_pipeline import PipelineView

tex = Texture()
app.pipe.set_views([PipelineView("mirror", mirror_cam, size=(1024, 1024), output=tex)])
```

Geometry programs read the camera from Panda's per-camera built-ins (`p3d_ViewMatrix`...), rewritten at translation time (`mcshader.glsl.views`). Per-view images are bound on each view's camera. As a result, extra views cost nothing per frame beyond their own rendering.

## Notes

Validated against BSL v10 on Apple M1 / Mesa, OpenGL 4.6 core. BSL is the only shaderpack confirmed to work, more will be fixed and implemented later

Python 3.10+. MIT licensed.
