"""Inspect a Minecraft shaderpack and decompile one program.

    python examples/list_pack_effects.py [path-to-pack]

Defaults to the bundled ``Shaders/`` pack. Needs no OpenGL context.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  # noqa: E402 (run from a source checkout without installing)


import sys

from mcshader import load_pack, decompile_program, builtin_effects


def main() -> None:
    pack_path = sys.argv[1] if len(sys.argv) > 1 else "Shaders/"

    print("Built-in effect ids you can apply by name:")
    for eid, e in builtin_effects().items():
        print(f"  {eid:11} - {e.description}")

    print(f"\nLoading pack: {pack_path}")
    pack = load_pack(pack_path)
    programs = pack.programs()
    print(f"  {pack.name}: {len(programs)} programs")
    print("  gbuffers programs:",
          ", ".join(p for p in programs if p.startswith("gbuffers")))

    target = "gbuffers_terrain"
    if target in programs:
        print(f"\nDecompiling {target!r} to modern GLSL:")
        prog = decompile_program(pack, target)
        print(f"  vertex:   {len(prog.vertex.source.splitlines())} lines")
        print(f"  fragment: {len(prog.fragment.source.splitlines())} lines")
        print(f"  Minecraft uniforms to feed: {', '.join(prog.mc_uniforms)}")


if __name__ == "__main__":
    main()
