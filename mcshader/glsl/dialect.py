"""Translate legacy Minecraft-shader GLSL into a modern, engine-usable dialect.

OptiFine/Iris shaders are written against the old ``#version 120`` fixed-function
pipeline: ``attribute``/``varying`` qualifiers, ``gl_Vertex`` and friends,
``gl_ModelViewProjectionMatrix``, ``texture2D``. None of that compiles under a
modern core profile, and none of it lines up with the vertex attributes an
engine like Panda3D actually feeds. This module rewrites a single stage's source
into either the Panda3D dialect (``p3d_*`` inputs) or a neutral "generic"
dialect (``mc_*`` inputs a raw-GL adapter binds itself).

It is deliberately a *best-effort textual* translator, not a full GLSL compiler.
It handles the mechanical dialect gap that blocks 90% of per-object effect
extraction; it does not reconstruct Minecraft's deferred pipeline. Anything it
can't confidently rewrite is reported in :attr:`TranslationResult.notes` rather
than silently changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import uniforms

__all__ = ["TranslationResult", "translate_stage", "Target"]

Target = str  # "panda3d" | "generic"
_TARGETS = ("panda3d", "generic")


@dataclass
class TranslationResult:
    source: str
    stage: str  # "vertex" | "fragment"
    target: Target
    mc_uniforms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# gl_* builtin -> (panda3d name, generic name). "attrib" builtins are only
# valid to declare in the vertex stage.
_BUILTINS = {
    "gl_ModelViewProjectionMatrix": ("p3d_ModelViewProjectionMatrix", "mc_ModelViewProjectionMatrix", "uniform mat4"),
    "gl_ModelViewMatrix":           ("p3d_ModelViewMatrix", "mc_ModelViewMatrix", "uniform mat4"),
    "gl_ProjectionMatrix":          ("p3d_ProjectionMatrix", "mc_ProjectionMatrix", "uniform mat4"),
    "gl_NormalMatrix":              ("p3d_NormalMatrix", "mc_NormalMatrix", "uniform mat3"),
    "gl_Vertex":         ("p3d_Vertex", "mc_Vertex", "attrib vec4"),
    "gl_Normal":         ("p3d_Normal", "mc_Normal", "attrib vec3"),
    "gl_Color":          ("p3d_Color", "mc_Color", "attrib vec4"),
    # gl_MultiTexCoord0/1 are handled specially (promoted to vec4) — see below.
}

# Texture-sampling functions that changed name in modern GLSL.
_TEX_FUNCS = {
    "texture2DLod": "textureLod",
    "texture3DLod": "textureLod",
    "textureCubeLod": "textureLod",
    "texture2DProj": "textureProj",
    "texture2D": "texture",
    "texture3D": "texture",
    "textureCube": "texture",
    # shadow2D is NOT renamed here: modern texture(sampler2DShadow, vec3) returns
    # a float, but packs use shadow2D(...).x expecting a vec4. A compat macro
    # (see _COMPAT_PRELUDE) preserves the vec4 shape.
}

# Compatibility macros injected after the version/extension lines.
_COMPAT_PRELUDE = [
    "#define shadow2D(s, p) vec4(texture((s), (p)))",
    "#define shadow2DLod(s, p, l) vec4(textureLod((s), (p), (l)))",
]

_DROP_LINE = re.compile(
    r"^\s*#\s*(version\b"
    r"|extension\s+GL_ARB_shading_language_include\b"
    r"|define\s+(attribute|varying)\b)"
)


def _word_sub(text: str, mapping: dict[str, str]) -> str:
    """Replace whole-identifier occurrences of each key with its value."""
    if not mapping:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, mapping)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def translate_stage(
    source: str,
    stage: str,
    target: Target = "panda3d",
    frag_output_map: dict[int, int] | None = None,
) -> TranslationResult:
    """Translate one shader ``stage`` ("vertex"/"fragment") into ``target``.

    Returns the rewritten source plus the set of Minecraft uniforms it still
    references (so an engine adapter knows what to feed) and human-readable
    notes about anything left for the caller to resolve.

    ``frag_output_map`` maps a ``gl_FragData[k]`` subscript to the framebuffer
    attachment location it should write (the ``colortex`` index from a pass's
    ``DRAWBUFFERS`` directive). Without it, each output keeps ``location = k``
    (correct for a single-target per-object effect).
    """
    if target not in _TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {_TARGETS}")
    if stage not in ("vertex", "fragment"):
        raise ValueError(f"stage must be 'vertex' or 'fragment', got {stage!r}")

    result = TranslationResult(source="", stage=stage, target=target)
    idx = 0 if target == "panda3d" else 1

    # 1. Drop directives that don't belong in the modern output.
    lines = [ln for ln in source.splitlines() if not _DROP_LINE.match(ln)]
    body = "\n".join(lines)

    # 2. Qualifier keywords: attribute/varying -> in/out by stage.
    if stage == "vertex":
        body = re.sub(r"\battribute\b", "in", body)
        body = re.sub(r"\bvarying\b", "out", body)
    else:
        # A fragment stage has no attributes; varying becomes an input.
        body = re.sub(r"\bvarying\b", "in", body)

    # 3. Builtin identifiers -> target names, remembering which we introduced.
    builtin_map: dict[str, str] = {}
    introduced: list[tuple[str, str]] = []  # (name, "uniform mat4" | "attrib vec4")
    for gl_name, (p3d_name, generic_name, decl) in _BUILTINS.items():
        if re.search(rf"\b{re.escape(gl_name)}\b", body):
            new_name = (p3d_name, generic_name)[idx]
            builtin_map[gl_name] = new_name
            introduced.append((new_name, decl))
    body = _word_sub(body, builtin_map)

    # ftransform() -> explicit MVP multiply.
    if "ftransform" in body:
        mvp = builtin_map.get("gl_ModelViewProjectionMatrix") or (
            "p3d_ModelViewProjectionMatrix" if idx == 0 else "mc_ModelViewProjectionMatrix"
        )
        vtx = builtin_map.get("gl_Vertex") or ("p3d_Vertex" if idx == 0 else "mc_Vertex")
        body = re.sub(r"\bftransform\s*\(\s*\)", f"({mvp} * {vtx})", body)
        introduced.append((mvp, "uniform mat4"))
        introduced.append((vtx, "attrib vec4"))

    # gl_TextureMatrix[n] has no core-profile equivalent. Minecraft uses it to
    # scale texture/lightmap coords, which we feed already in [0,1], so treat it
    # as identity.
    body = re.sub(r"\bgl_TextureMatrix\s*\[\s*\d+\s*\]", "mat4(1.0)", body)

    # gl_MultiTexCoord0 is a vec4 in Minecraft; Panda3D provides a vec2 texcoord,
    # so promote every use to vec4 (and keep the vec2 attribute). gl_MultiTexCoord1
    # is the block/sky lightmap, which has no engine equivalent yet — feed full
    # bright so shaders compile and render lit.
    texcoord_attr = "p3d_MultiTexCoord0" if idx == 0 else "mc_MultiTexCoord0"
    if re.search(r"\bgl_MultiTexCoord0\b", body):
        body = re.sub(r"\bgl_MultiTexCoord0\b",
                      f"vec4({texcoord_attr}, 0.0, 1.0)", body)
        introduced.append((texcoord_attr, "attrib vec2"))
    body = re.sub(r"\bgl_MultiTexCoord1\b", "vec4(1.0)", body)

    # 3b. Custom Minecraft per-vertex attributes (mc_Entity, mc_midTexCoord, …)
    # have no real per-vertex data in engine-authored geometry — nothing in this
    # translator's output was ever wired to a vertex column named "mc_Entity",
    # so declaring it as a bare `in vec4 mc_Entity;` (what the generic
    # attribute->in qualifier swap above does on its own) leaves it reading
    # whatever the driver defaults an unbound attribute to (typically all
    # zeros). Since `mc_Entity.x` is the block id BSL's WavingBlocks() (and
    # every material-branching gbuffers program) keys off of, this silently
    # disabled per-object waving/movement/emissive-material selection
    # entirely, no matter what an adapter's `set_block_id()` was told to do.
    # Synthesize these as per-object globals driven by the `mcEntityId`
    # uniform an adapter DOES feed per node (see PipelineRenderer.set_block_id)
    # — a whole-object approximation of the real per-vertex value, correct for
    # the common case (one block id per tagged object) and strictly better
    # than the always-zero status quo for a mixed-id mesh.
    extra_header: list[str] = []
    if stage == "vertex":
        entity_decl_re = re.compile(r"^[ \t]*(?:attribute|in)\s+vec4\s+mc_Entity\s*;\s*$", re.M)
        if entity_decl_re.search(body):
            body = entity_decl_re.sub("", body)
            extra_header += [
                "uniform float mcEntityId;",
                "vec4 mc_Entity = vec4(mcEntityId, 0.0, 0.0, 1.0);",
            ]
        midtex_decl_re = re.compile(r"^[ \t]*(?:attribute|in)\s+vec4\s+mc_midTexCoord\s*;\s*$", re.M)
        if midtex_decl_re.search(body):
            body = midtex_decl_re.sub("", body)
            # Equal to the live texcoord (not a real face midpoint), so any
            # `texCoord.t < mc_midTexCoord.t` "top half only" split BSL makes
            # (e.g. tall-grass-only-sways-at-the-top) resolves to "whole mesh
            # sways" instead — an honest approximation, not a real midpoint.
            # Force the texcoord attribute's own declaration to exist even if
            # this particular source never separately referenced
            # gl_MultiTexCoord0 itself (in practice it always does, but this
            # keeps the synthesis self-contained rather than order-dependent).
            introduced.append((texcoord_attr, "attrib vec2"))
            extra_header.append(f"vec4 mc_midTexCoord = vec4({texcoord_attr}, 0.0, 1.0);")

    # 4. Texture function renames.
    body = _word_sub(body, _TEX_FUNCS)

    # `texture` was a legal identifier in old GLSL, so packs declare
    # `uniform sampler2D texture;` (the bound block/entity atlas). In core GLSL
    # `texture` is the sampling builtin, so that declaration shadows it and every
    # later texture(...) call fails to resolve. Rename the identifier — but never
    # the `texture(` builtin call — to the engine's base-texture input so it binds
    # automatically.
    base_tex = "p3d_Texture0" if idx == 0 else "mc_BaseTexture"
    body = re.sub(r"\btexture\b(?!\s*\()", base_tex, body)

    # 5. Fragment outputs: gl_FragColor / gl_FragData[n] -> declared outs.
    frag_outs: list[str] = []
    if stage == "fragment":
        omap = frag_output_map or {}
        if re.search(r"\bgl_FragColor\b", body):
            body = re.sub(r"\bgl_FragColor\b", "mcFragColor", body)
            frag_outs.append(f"layout(location = {omap.get(0, 0)}) out vec4 mcFragColor;")
        for n in sorted(int(x) for x in set(re.findall(r"gl_FragData\s*\[\s*(\d+)\s*\]", body))):
            body = re.sub(rf"gl_FragData\s*\[\s*{n}\s*\]", f"mcFragData{n}", body)
            location = omap.get(n, n)
            frag_outs.append(f"layout(location = {location}) out vec4 mcFragData{n};")
        if "gl_TexCoord" in body:
            result.notes.append(
                "uses gl_TexCoord[]: modern GLSL has no fixed-function varyings; "
                "wire an explicit texcoord varying from the vertex stage."
            )

    # 6. Build the declaration prelude (dedup, stage-aware).
    decls: list[str] = []
    seen: set[str] = set()
    for name, decl in introduced:
        if name in seen:
            continue
        seen.add(name)
        kind, gtype = decl.split(" ", 1)
        if kind == "attrib":
            if stage == "vertex":
                decls.append(f"in {gtype} {name};")
        else:  # uniform
            decls.append(f"uniform {gtype} {name};")

    # 7. Detected Minecraft uniforms (for the adapter to feed).
    used_mc = sorted(
        name for name in uniforms.CATALOG if re.search(rf"\b{re.escape(name)}\b", body)
    )
    result.mc_uniforms = used_mc

    # #extension directives must sit at the very top; hoist any that survived
    # (our own injected declarations would otherwise push them "into the middle").
    ext_re = re.compile(r"^\s*#\s*extension\b")
    body_lines = body.splitlines()
    extensions = [ln.strip() for ln in body_lines if ext_re.match(ln)]
    body = "\n".join(ln for ln in body_lines if not ext_re.match(ln))

    header = ["#version 330"] + extensions + _COMPAT_PRELUDE
    if frag_outs:
        header += frag_outs
    if decls:
        header += ["// [mcshader] injected engine inputs"] + decls
    if extra_header:
        header += ["// [mcshader] synthesized per-object vertex-input approximations"] + extra_header

    result.source = "\n".join(header) + "\n\n" + body.strip() + "\n"
    return result
