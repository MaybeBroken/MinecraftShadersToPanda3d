"""Command-line access to the decompiler and pipeline (GPU-free except ``demo``).

    python -m mcshader pipeline <pack> [--world world0] [--profile HIGH]
    python -m mcshader options  <pack> [--profile HIGH] [--screen LIGHTING]
    python -m mcshader list     <pack>
    python -m mcshader show     <pack> <program> [--world world0] [--stage vertex|fragment]
    python -m mcshader demo     [pack]   (needs panda3d + a display)
    python -m mcshader effects
    python -m mcshader extract  <pack> <program> --out <dir> [--target panda3d|generic]
"""

from __future__ import annotations

import argparse
import os
import sys

from . import load_pack, decompile_program, builtin_effects
from . import build_graph, ShaderOptions, RenderTypeResolver


def _load_options(pack, profile: str | None):
    props = pack.properties()
    opts = ShaderOptions.from_pack(pack.option_sources(), props)
    if profile:
        opts.apply_profile(profile, props)
    return props, opts


def _cmd_pipeline(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    props, opts = _load_options(pack, args.profile)
    graph = build_graph(pack, args.world)
    resolver = RenderTypeResolver(pack)
    values = opts.values()

    print(f"Pack: {pack.name}   world: {args.world}"
          + (f"   profile: {args.profile}" if args.profile else ""))
    print("Buffers: " + ", ".join(
        f"colortex{i}={graph.buffers.format_of(i)}" for i in graph.buffers.indices()))
    print("\nPass order (enabled under current options):")
    for p in graph.enabled_passes(values):
        tag = "geo" if p.kind == "geometry" else "FS "
        outs = f" -> colortex{p.outputs}" if p.outputs else ""
        cond = f"   [{p.enable_expr}]" if p.enable_expr else ""
        print(f"  {tag} [{p.kind:10}] {p.name:24}{outs}{cond}")
    disabled = [p.name for p in graph.passes if not p.enabled(values)]
    if disabled:
        print("\nDisabled by current options:", ", ".join(disabled))
    print("\nRender types this pack serves:")
    for rt in resolver.types():
        print(f"  {rt:14} -> {resolver.program(rt)}")
    return 0


def _print_screen(node, opts, indent=0):
    pad = "  " * indent
    for el in node["elements"]:
        if el["type"] == "spacer":
            continue
        if el["type"] == "screen":
            print(f"{pad}[{el['name']}]")
            _print_screen(el["screen"], opts, indent + 1)
        elif el["type"] == "option":
            val = el.get("value")
            allowed = f"  choices={el['allowed']}" if el.get("allowed") else ""
            kind = el.get("kind", "?")
            print(f"{pad}- {el['name']:26} = {val!s:8} ({kind}){allowed}")
        else:
            print(f"{pad}- {el.get('name', el)}")


def _cmd_options(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    props, opts = _load_options(pack, args.profile)
    print(f"{pack.name}: {len(opts.options)} options"
          + (f"   profile: {args.profile}" if args.profile else ""))
    print(f"Profiles: {', '.join(props.profiles)}\n")
    tree = opts.menu_tree(props, args.screen or "")
    _print_screen(tree, opts)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    progs = pack.programs()
    print(f"{pack.name}: {len(progs)} programs, {len(pack.files)} source files")
    for name in progs:
        print(f"  {name}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    prog = decompile_program(pack, args.program, world=args.world, target=args.target)
    if prog.mc_uniforms:
        print("// Minecraft uniforms this program expects (feed via shader inputs):")
        print("//   " + ", ".join(prog.mc_uniforms))
    for note in prog.notes:
        print(f"// note: {note}")
    tr = prog.vertex if args.stage == "vertex" else prog.fragment
    if tr is None:
        print(f"(no {args.stage} stage)", file=sys.stderr)
        return 1
    print(tr.source)
    return 0


def _cmd_effects(_args: argparse.Namespace) -> int:
    for eid, e in builtin_effects().items():
        params = ", ".join(e.params) or "(none)"
        print(f"{eid:12} {e.name}")
        print(f"             {e.description}")
        print(f"             params: {params}")
        print(f"             from:   {e.mc_origin}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    pack = load_pack(args.pack)
    prog = decompile_program(pack, args.program, world=args.world, target=args.target)
    os.makedirs(args.out, exist_ok=True)
    written = []
    if prog.vertex:
        path = os.path.join(args.out, f"{args.program}.vert")
        with open(path, "w") as fh:
            fh.write(prog.vertex.source)
        written.append(path)
    if prog.fragment:
        path = os.path.join(args.out, f"{args.program}.frag")
        with open(path, "w") as fh:
            fh.write(prog.fragment.source)
        written.append(path)
    for path in written:
        print(f"wrote {path}")
    return 0 if written else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    """The one command here that needs a GPU: run the bundled demo scene."""
    from .demo import run

    run(args.pack, profile=args.profile or "LOW")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcshader", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pipeline", help="show the resolved deferred pipeline")
    p.add_argument("pack")
    p.add_argument("--world", default="world0")
    p.add_argument("--profile", default=None)
    p.set_defaults(func=_cmd_pipeline)

    p = sub.add_parser("options", help="show the shader options menu tree")
    p.add_argument("pack")
    p.add_argument("--profile", default=None)
    p.add_argument("--screen", default=None, help="a sub-screen name, e.g. LIGHTING")
    p.set_defaults(func=_cmd_options)

    p = sub.add_parser("list", help="list programs in a shaderpack")
    p.add_argument("pack")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("show", help="print a decompiled shader stage")
    p.add_argument("pack")
    p.add_argument("program")
    p.add_argument("--world", default="world0")
    p.add_argument("--stage", choices=("vertex", "fragment"), default="fragment")
    p.add_argument("--target", choices=("panda3d", "generic"), default="panda3d")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("demo", help="run the bundled demo scene (needs panda3d)")
    p.add_argument("pack", nargs="?", default=None,
                   help="a pack dir/zip; searched for if omitted")
    p.add_argument("--profile", default=None)
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("effects", help="list built-in effects")
    p.set_defaults(func=_cmd_effects)

    p = sub.add_parser("extract", help="write decompiled stages to files")
    p.add_argument("pack")
    p.add_argument("program")
    p.add_argument("--out", required=True)
    p.add_argument("--world", default="world0")
    p.add_argument("--target", choices=("panda3d", "generic"), default="panda3d")
    p.set_defaults(func=_cmd_extract)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
