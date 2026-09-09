"""High-level entry point: turn a Minecraft shaderpack program into an Effect.

This is the "decompiler" façade tying the pieces together:

    pack -> read program -> resolve #include -> split VSH/FSH ->
    translate each stage to a modern dialect -> wrap as an Effect

The result is an :class:`~mcshader.effects.base.Effect` you can drop into a
:class:`~mcshader.registry.ShaderRegistry` under any id and then apply to an
object. Note the honest boundary (see README): a lifted program compiles and
runs as a per-object surface shader, but it will *not* reproduce the pack's
whole-scene deferred pipeline — the composite/shadow passes have no per-object
equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .effects.base import Effect
from .glsl import preprocess
from .glsl.dialect import translate_stage, TranslationResult
from .pack import ProgramSource, ShaderPack

__all__ = ["decompile_program", "DecompiledProgram", "load_pack"]


@dataclass
class DecompiledProgram:
    name: str
    vertex: TranslationResult | None
    fragment: TranslationResult | None

    @property
    def mc_uniforms(self) -> list[str]:
        found: set[str] = set()
        for tr in (self.vertex, self.fragment):
            if tr:
                found.update(tr.mc_uniforms)
        return sorted(found)

    @property
    def notes(self) -> list[str]:
        out: list[str] = []
        for tr in (self.vertex, self.fragment):
            if tr:
                out.extend(tr.notes)
        return out

    def as_effect(self, shader_id: str, *, description: str = "") -> Effect:
        """Wrap the translated sources as a registrable :class:`Effect`."""
        if not (self.vertex and self.fragment):
            raise ValueError(
                f"program {self.name!r} is missing a "
                f"{'vertex' if not self.vertex else 'fragment'} stage; "
                "cannot build a complete effect"
            )
        return Effect(
            id=shader_id,
            name=self.name,
            description=description or f"Decompiled from Minecraft program {self.name!r}",
            vertex=self.vertex.source,
            fragment=self.fragment.source,
            auto_inputs=[],  # MC uniforms are reported separately; feed as needed
            dialect=self.vertex.target,  # matches the translation target
            mc_origin=self.name,
            tags=["decompiled"],
        )


def load_pack(path: str) -> ShaderPack:
    """Convenience wrapper around :meth:`ShaderPack.from_path`."""
    return ShaderPack.from_path(path)


def decompile_program(
    pack: ShaderPack,
    name: str,
    *,
    world: str = "world0",
    target: str = "panda3d",
) -> DecompiledProgram:
    """Decompile one program from ``pack`` into translated stage sources."""
    program: ProgramSource = pack.read_program(name, world=world)
    base_dir = pack.base_dir or "."

    def _prep(src: str) -> str:
        return preprocess.resolve_includes(src, base_dir=base_dir, files=pack.files)

    # A stage's raw source is its own file when present, otherwise the combined
    # .glsl. Iris stubs (#define VSH + #include) resolve to the combined file
    # too, so select_stage runs in every case to strip the other stage's blocks.
    vert_raw = program.vertex or program.combined
    frag_raw = program.fragment or program.combined

    vertex = None
    if vert_raw is not None:
        src = preprocess.select_stage(_prep(vert_raw), "vertex")
        vertex = translate_stage(src, "vertex", target)

    fragment = None
    if frag_raw is not None:
        src = preprocess.select_stage(_prep(frag_raw), "fragment")
        fragment = translate_stage(src, "fragment", target)

    return DecompiledProgram(name=name, vertex=vertex, fragment=fragment)
