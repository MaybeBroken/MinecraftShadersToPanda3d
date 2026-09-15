"""Run a Minecraft shaderpack's deferred pipeline over a real Panda3D scene.

    pip install panda3d
    python examples/pipeline_demo.py [path-to-pack]

Identical to ``python -m mcshader demo`` — the scene lives in the package now
(``mcshader/demo.py``), which is also the worked example for the one-call API:

    app = mcshader.init(profile="LOW")                    # window + pack + sky
    app.load("models/environment", tag=RULES, scale=0.25)  # place + tag a model
    app.load_actor("models/panda-model", {"walk": ...}, type="entity")
    app.camera(pos=..., look_at=..., fov=100)
    app.debug_ui(focus=panda)                              # the whole dev HUD
    app.run()

This file just keeps it runnable straight from a source checkout (no install).
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# noqa: E402 (run from a source checkout without installing)

from mcshader.demo import run

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
