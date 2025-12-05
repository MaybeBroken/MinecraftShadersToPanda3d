import glfw
from OpenGL.GL import glUseProgram, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
from OpenGL.GL.shaders import compileProgram, compileShader


def _ensure_gl_context():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW; cannot create OpenGL context.")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    # Request legacy compatibility context for OpenGL 2.1 (GLSL 1.20)
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

        if not vert_source and not frag_source:
            raise ValueError(
                "No shader sources provided. Expected .vsh/.vert and/or .fsh/.frag files."
            )

        shaders = []
        if vert_source:
            print("  Compiling vertex shader…")
            shaders.append(compileShader(vert_source, GL_VERTEX_SHADER))
        if frag_source:
            print("  Compiling fragment shader…")
            shaders.append(compileShader(frag_source, GL_FRAGMENT_SHADER))

        if not shaders:
            raise ValueError("No valid shader stages to compile.")

        print("  Linking program…")
        program = compileProgram(*shaders)
        print("  Successfully linked program.")
        return program
    except Exception as e:
        print(f"  Failed to compile/link program: {e}")
        raise
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
