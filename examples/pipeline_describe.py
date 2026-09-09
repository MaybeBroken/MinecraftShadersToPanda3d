"""Print a pack's resolved pipeline + options — no GPU, no Panda3D required.

    python examples/pipeline_describe.py [path-to-pack] [profile]
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  # noqa: E402

from mcshader import load_pack, build_graph, ShaderOptions, RenderTypeResolver


def main():
    pack_path = sys.argv[1] if len(sys.argv) > 1 else "Shaders/"
    profile = sys.argv[2] if len(sys.argv) > 2 else "HIGH"

    pack = load_pack(pack_path)
    props = pack.properties()
    opts = ShaderOptions.from_pack(pack.option_sources(), props)
    opts.apply_profile(profile, props)
    graph = build_graph(pack, "world0")
    resolver = RenderTypeResolver(pack)

    print(f"{pack.name}  (profile: {profile})")
    print(f"  {len(opts.options)} options, profiles: {', '.join(props.profiles)}")
    print(f"  buffers: {[f'colortex{i}' for i in graph.buffers.indices()]}")
    print("  enabled passes:")
    for p in graph.enabled_passes(opts.values()):
        print(f"    {p.kind:10} {p.name:24} -> colortex{p.outputs}")
    print(f"  render types: {', '.join(resolver.types())}")


if __name__ == "__main__":
    main()
