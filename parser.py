import pprint
import glfw
from OpenGL.GL import glUseProgram, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
from OpenGL.GL.ARB.shading_language_include import (
    glNamedStringARB,
    GL_SHADER_INCLUDE_ARB,
)
from OpenGL.GL.shaders import compileProgram, compileShader
import re


def _ensure_gl_context():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW; cannot create OpenGL context.")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    # Request a context that supports ARB include; macOS core 4.1 typically works
    # but keep compatibility where possible. If 2.1 fails to expose ARB include,
    # try a higher context by allowing GLFW to pick best available.
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    # Do NOT set core profile; we want compatibility for GL 2.1
    # macOS typically supports 2.1 compatibility without forward-compat.
    try:
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.FALSE)
    except AttributeError:
        pass
    window = glfw.create_window(1, 1, "shader-compile", None, None)
    if not window:
        err = glfw.get_error()
        glfw.terminate()
        raise RuntimeError(
            f"Failed to create GLFW window/context. "
            f"Ensure calls run on the main thread on macOS. GLFW error: {err}"
        )
    glfw.make_context_current(window)
    return window


def _destroy_gl_context(window):
    if window:
        glfw.destroy_window(window)
    glfw.terminate()


def _preprocess_includes(source: str, base_dir: str, files: dict) -> str:
    """Resolve simple GLSL #include directives.
    Supports lines like: #include "path/to/file.glsl"
    - First tries to load from provided files dict by key matching the include path.
    - Then tries the filesystem relative to base_dir.
    - Strips BOM if present; preserves line order.
    """
    import os

    resolved_lines = []
    for line in source.splitlines():
        striped = line.strip()
        if striped.startswith("#include"):
            # Extract quoted path
            start = striped.find('"')
            end = striped.rfind('"')
            # Support angle brackets too: #include <path>
            if start == -1 or end == -1 or end <= start:
                start = striped.find("<")
                end = striped.rfind(">")
            include_path = None
            if start != -1 and end != -1 and end > start:
                include_path = striped[start + 1 : end]
            if not include_path:
                resolved_lines.append(line)
                continue
            # Try files dict
            if include_path in files:
                included = files[include_path]
            else:
                fs_path = include_path
                # Treat leading '/' as shader-pack root relative to base_dir
                if os.path.isabs(fs_path):
                    # Some packs use absolute-like paths; remap to base_dir
                    fs_path = os.path.join(base_dir, fs_path.lstrip("/"))
                else:
                    fs_path = os.path.join(base_dir, fs_path)
                try:
                    with open(fs_path, "r", encoding="utf-8") as f:
                        included = f.read()
                except Exception:
                    # Could not resolve; keep original line to surface error
                    resolved_lines.append(line)
                    continue
            # Recursively resolve nested includes
            resolved = _preprocess_includes(included, base_dir, files)
            resolved_lines.append(resolved)
        else:
            resolved_lines.append(line)
    return "\n".join(resolved_lines)


def _normalize_preprocessor_directives(source: str) -> str:
    """Normalize GLSL preprocessor conditionals for GLSL 120.
    Transform `#if FEATURE` and `#elif FEATURE` into `#if defined(FEATURE)` forms.
    Avoid touching lines with parentheses, operators, or numeric constants.
    """
    import re

    def repl_if(match):
        ident = match.group(1)
        return f"#if defined({ident})"

    def repl_elif(match):
        ident = match.group(1)
        return f"#elif defined({ident})"

    # Process line by line to allow trailing comments and avoid complex expressions
    out_lines = []
    for ln in source.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("//"):
            out_lines.append(ln)
            continue
        m = re.match(r"^\s*#if\s+([A-Za-z_][A-Za-z0-9_]*)\b(.*)$", ln)
        if m:
            ident = m.group(1)
            tail = m.group(2).rstrip()
            if "(" not in tail and ")" not in tail and not tail:
                ln = f"#if defined({ident})"
                out_lines.append(ln)
                continue
        m = re.match(r"^\s*#elif\s+([A-Za-z_][A-Za-z0-9_]*)\b(.*)$", ln)
        if m:
            ident = m.group(1)
            tail = m.group(2).rstrip()
            if "(" not in tail and ")" not in tail and not tail:
                ln = f"#elif defined({ident})"
                out_lines.append(ln)
                continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def split_shader_stages(source: str) -> dict[str, str]:
    """Split combined shader source into 'vertex' and 'fragment' stages.

    Supports blocks opened by:
      - #ifdef FSH / #ifdef VSH
      - #if FSH / #if VSH
      - #if defined(FSH) / #if defined(VSH)
      - #elif FSH / #elif defined(FSH) (same for VSH)

    Collects content until the corresponding #endif at the same nesting level.
    Nested conditionals inside a stage block are preserved.
    Multiple stage blocks are concatenated.
    """

    stages: dict[str, list[str]] = {"vertex": [], "fragment": []}
    current_stage: str | None = None
    depth: int = 0  # nesting depth within the current stage block

    # Regexes to detect entering/exiting stage blocks
    re_ifdef = re.compile(r"^\s*#\s*ifdef\s+([A-Za-z_][A-Za-z0-9_]*)\b")
    re_if = re.compile(r"^\s*#\s*if\s+(.*)$")
    re_elif = re.compile(r"^\s*#\s*elif\s+(.*)$")
    re_endif = re.compile(r"^\s*#\s*endif\b")

    def expr_targets_stage(expr: str) -> str | None:
        """Return 'fragment' if expr refers to FSH, 'vertex' if VSH, else None."""
        # Normalize whitespace
        e = expr.strip()
        # Direct identifiers: FSH or VSH
        if re.fullmatch(r"(defined\s*\(\s*FSH\s*\)|FSH)\b", e):
            return "fragment"
        if re.fullmatch(r"(defined\s*\(\s*VSH\s*\)|VSH)\b", e):
            return "vertex"
        # Handle simple expressions like "defined(FSH) && SOMETHING"
        if re.search(r"\bdefined\s*\(\s*FSH\s*\)\b|\bFSH\b", e):
            return "fragment"
        if re.search(r"\bdefined\s*\(\s*VSH\s*\)\b|\bVSH\b", e):
            return "vertex"
        return None

    for line in source.splitlines():
        stripped = line.strip()

        m = re_ifdef.match(line)
        if m:
            ident = m.group(1)
            stage = (
                "fragment" if ident == "FSH" else "vertex" if ident == "VSH" else None
            )
            if stage is not None:
                # Enter stage block
                if current_stage is None:
                    current_stage = stage
                    depth = 1
                else:
                    # Nested #ifdef inside a stage block
                    depth += 1
                # Do not include the outermost stage directive; include nested ones
                if depth > 1:
                    stages[current_stage].append(line)
            else:
                # Non-stage #ifdef: if inside a stage, include; else ignore
                if current_stage is not None:
                    stages[current_stage].append(line)
            continue

        m = re_if.match(line)
        if m:
            stage = expr_targets_stage(m.group(1))
            if stage is not None:
                if current_stage is None:
                    current_stage = stage
                    depth = 1
                else:
                    depth += 1
                # Do not include the outermost stage directive; include nested ones
                if depth > 1:
                    stages[current_stage].append(line)
            else:
                if current_stage is not None:
                    stages[current_stage].append(line)
            continue

        m = re_elif.match(line)
        if m:
            # #elif switches only if we're at depth==1 and the expr targets other stage
            if current_stage is not None:
                target = expr_targets_stage(m.group(1))
                if depth == 1 and target is not None and target != current_stage:
                    # Switch stage but keep collecting at same depth
                    current_stage = target
                # Include nested elifs, but skip the outer stage-switch elif directive
                if depth > 1 or target is None:
                    stages[current_stage].append(line)
            continue

        if re_endif.match(line):
            if current_stage is not None:
                depth -= 1
                # Include nested endifs, but skip closing outermost stage block
                if depth >= 1:
                    stages[current_stage].append(line)
                if depth == 0:
                    current_stage = None
            # If not in a stage, just ignore endifs outside
            continue

        # Regular line
        if current_stage is not None:
            stages[current_stage].append(line)

    # Join collected lines and drop empty results
    result: dict[str, str] = {}
    for k, v in stages.items():
        joined = "\n".join(v).strip()
        if joined:
            result[k] = joined
    return result


def compile_shader(shader_pack, files, base_dir: str = "."):
    print(f"Rendering shader pack: {shader_pack}")

    window = _ensure_gl_context()
    try:
        vert_source = None
        frag_source = None

        for filename, content in files.items():
            if filename.endswith(".vsh") or filename.endswith(".vert"):
                vert_source = _preprocess_includes(content, base_dir, files)
            elif filename.endswith(".fsh") or filename.endswith(".frag"):
                frag_source = _preprocess_includes(content, base_dir, files)
            elif filename.endswith(".glsl"):
                preprocessed = _preprocess_includes(content, base_dir, files)
                normalized = _normalize_preprocessor_directives(preprocessed)
                stages = split_shader_stages(normalized)
                if "vertex" in stages:
                    if vert_source is None:
                        vert_source = stages["vertex"]
                    else:
                        vert_source += "\n" + stages["vertex"]
                if "fragment" in stages:
                    if frag_source is None:
                        frag_source = stages["fragment"]
                    else:
                        frag_source += "\n" + stages["fragment"]

        if not vert_source and not frag_source:
            raise ValueError(
                "No shader sources provided. Expected .vsh/.vert and/or .fsh/.frag files."
            )
        else:
            pprint.pprint(
                {
                    "vertex_source_present": vert_source,
                    "fragment_source_present": frag_source,
                }
            )

        # Register includes via ARB_shading_language_include if available
        try:
            for filename, content in files.items():
                # Use pack-relative paths as include names
                if filename.endswith(
                    (".glsl", ".inc", ".vsh", ".vert", ".fsh", ".frag")
                ):
                    glNamedStringARB(GL_SHADER_INCLUDE_ARB, f"/{filename}", content)
        except Exception:
            # If ARB include registration fails, proceed with preprocessed sources
            pass

        # Inject header enabling ARB include so all shaders share same context
        header = (
            "\n".join(
                [
                    "#version 120",
                    "#extension GL_ARB_shading_language_include : enable",
                ]
            )
            + "\n"
        )

        shaders = []
        if vert_source:
            print("  Compiling vertex shader…")
            shaders.append(compileShader(header + vert_source, GL_VERTEX_SHADER))
        if frag_source:
            print("  Compiling fragment shader…")
            shaders.append(compileShader(header + frag_source, GL_FRAGMENT_SHADER))

        if not shaders:
            raise ValueError("No valid shader stages to compile.")

        print("  Linking program…")
        program = compileProgram(*shaders)
        print("  Successfully linked program.")
        return program
    except Exception as e:
        print(f"  Failed to compile/link program: {str(e)[:800]}...")
    finally:
        _destroy_gl_context(window)


def render_shader(program):
    print("  Rendering with shader program…")
    try:
        glUseProgram(program)
        print("  Shader program is now active.")
    except Exception as e:
        print(f"  Failed to use shader program: {e}")
        raise
    finally:
        glUseProgram(0)
        print("  Shader program deactivated.")
