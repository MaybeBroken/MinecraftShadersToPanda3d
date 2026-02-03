import ctypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DEPTH_ATTACHMENT,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_LINK_STATUS,
    GL_RENDERBUFFER,
    GL_RGBA,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_TRUE,
    GL_DEPTH_COMPONENT24,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glAttachShader,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBindRenderbuffer,
    glBufferData,
    glClear,
    glClearColor,
    glCheckFramebufferStatus,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteRenderbuffers,
    glDeleteBuffers,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteShader,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDisable,
    glDrawElements,
    glEnable,
    glEnableVertexAttribArray,
    glFramebufferRenderbuffer,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glTexImage2D,
    glTexParameteri,
    glRenderbufferStorage,
    glUniform1f,
    glUniform1i,
    glUniform2f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMatrix4x4, QVector3D, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

SHADER_PATH = Path("Shaders/")


def modify_shader_sources(vertex_src: str, fragment_src: str) -> tuple[str, str]:
    # Ensure version directive is present and at the top; avoid duplicating if already set.
    version_line = "#version 330 core"
    defaults = """
#ifndef MC_VERSION
#define MC_VERSION 12000
#endif
"""
    compat = """
#ifdef VSH
#define varying out
#define attribute in
#else
#define varying in
#define attribute in
#ifndef FRAG_COLOR_DEFINED
#define FRAG_COLOR_DEFINED
out vec4 FragColor;
#define gl_FragColor FragColor
#endif
#endif
"""

    def _normalize(src: str, stage_define: str) -> str:
        stripped = src.lstrip()
        injected = defaults + stage_define + compat
        if stripped.startswith("#version"):
            return stripped + "\n" + injected
        return f"{version_line}\n" + injected + stripped

    vertex_norm = _normalize(vertex_src, "#define VSH\n")
    fragment_norm = _normalize(fragment_src, "#define FSH\n")
    return vertex_norm, fragment_norm


def _resolve_includes(
    src: str, pack_root: Path, seen: set[Path], depth: int = 0
) -> str:
    if depth > 10:
        raise RuntimeError("Include depth too deep (possible cycle)")

    lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#include") and '"' in stripped:
            try:
                include_target = Path(stripped.split('"')[1])
            except IndexError:
                raise RuntimeError(f"Invalid include directive: {line}")
            include_target = Path(stripped.split('"')[1])
            if include_target.is_absolute():
                include_target = include_target.relative_to("/")
            include_path = pack_root / include_target
            if not include_path.exists():
                matches = list(pack_root.rglob(include_target.name))
                if matches:
                    include_path = matches[0]
                else:
                    raise RuntimeError(f"Include file not found: {include_target}")
            if include_path in seen:
                raise RuntimeError(f"Include cycle detected at {include_path}")
            if not include_path.exists():
                raise RuntimeError(f"Include file not found: {include_path}")
            seen.add(include_path)
            included = include_path.read_text()
            lines.append(_resolve_includes(included, pack_root, seen, depth + 1))
            seen.remove(include_path)
            continue
        lines.append(line)
    return "\n".join(lines)


# Utility to freeze program and view shader sources as text files
def sView(vertex_src: str, fragment_src: str) -> None:
    import tempfile
    import webbrowser
    import os

    with tempfile.TemporaryDirectory() as tmpdirname:
        vert_path = os.path.join(tmpdirname, "vertex_shader.glsl")
        frag_path = os.path.join(tmpdirname, "fragment_shader.glsl")
        with open(vert_path, "w") as vert_file:
            vert_file.write(vertex_src)
        with open(frag_path, "w") as frag_file:
            frag_file.write(fragment_src)
        webbrowser.open(f"file://{vert_path}")
        webbrowser.open(f"file://{frag_path}")


@dataclass
class ShaderPass:
    name: str
    shader_type: str  # "3d" or "2d"
    vertex_path: Optional[Path]
    fragment_path: Path
    program: Optional[int] = None
    uniform_name: str = "u_strength"
    strength: float = 1.0


DEFAULT_2D_VERTEX = """
#version 330 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(a_position.xy, 0.0, 1.0);
    v_uv = a_uv;
}
"""


class GLCanvas(QOpenGLWidget):
    def __init__(self, pack_root: Path) -> None:
        super().__init__()
        self.pack_root = pack_root
        self.passes: List[ShaderPass] = []
        self.quad_vao: Optional[int] = None
        self.quad_vbo: Optional[int] = None
        self.quad_ebo: Optional[int] = None
        self.cube_vao: Optional[int] = None
        self.cube_vbo: Optional[int] = None
        self.cube_ebo: Optional[int] = None
        self.fbos: List[tuple[int, int, int]] = []  # (fbo, color_tex, depth_rbo)
        self.start_time = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    # Public API ---------------------------------------------------------
    def add_shader_pass(self, shader_pass: ShaderPass) -> None:
        # Compile first; only append if successful to avoid dangling None programs.
        if self.context() is not None:
            self.makeCurrent()
            self._compile_pass(shader_pass)
            self.doneCurrent()
        else:
            # QOpenGLWidget not yet initialized; store and compile later in initializeGL.
            pass
        self.passes.append(shader_pass)
        self.update()

    def remove_shader_pass(self, index: int) -> None:
        if 0 <= index < len(self.passes):
            shader_pass = self.passes.pop(index)
            if shader_pass.program:
                self.makeCurrent()
                glDeleteProgram(shader_pass.program)
                self.doneCurrent()
            self.update()

    def move_shader_pass(self, index: int, offset: int) -> None:
        new_index = index + offset
        if 0 <= index < len(self.passes) and 0 <= new_index < len(self.passes):
            self.passes[index], self.passes[new_index] = (
                self.passes[new_index],
                self.passes[index],
            )
            self.update()

    # QOpenGLWidget lifecycle ------------------------------------------
    def initializeGL(self) -> None:
        glClearColor(0.05, 0.05, 0.05, 1.0)
        glEnable(GL_DEPTH_TEST)
        self._init_geometry()
        self._init_fbos()
        for shader_pass in self.passes:
            self._compile_pass(shader_pass)

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, width, height)
        self._init_fbos()

    def paintGL(self) -> None:
        default_fbo = self.defaultFramebufferObject()
        if not self.passes:
            glBindFramebuffer(GL_FRAMEBUFFER, default_fbo)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            return

        last_texture = None
        for idx, shader_pass in enumerate(self.passes):
            if shader_pass.program is None:
                continue  # skip passes that failed to compile
            is_last = idx == len(self.passes) - 1
            target_fbo, target_tex = self._target_for_pass(default_fbo, is_last, idx)

            glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
            glViewport(0, 0, self.width(), self.height())
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUseProgram(shader_pass.program or 0)
            self._apply_common_uniforms(shader_pass, last_texture)

            if shader_pass.shader_type == "3d":
                glEnable(GL_DEPTH_TEST)
                self._draw_cube(shader_pass)
            else:
                glDisable(GL_DEPTH_TEST)
                self._draw_quad()

            if not is_last:
                last_texture = target_tex
            else:
                last_texture = None

        glBindFramebuffer(GL_FRAMEBUFFER, default_fbo)
        glViewport(0, 0, self.width(), self.height())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(0)
        # Present final texture if at least one offscreen pass existed.
        if last_texture is not None:
            glDisable(GL_DEPTH_TEST)
            glBindFramebuffer(GL_FRAMEBUFFER, default_fbo)
            glClear(GL_COLOR_BUFFER_BIT)
            self._blit_texture(last_texture)

    # Helpers -----------------------------------------------------------
    def _init_geometry(self) -> None:
        quad_vertices = [
            # x, y, z, u, v
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            0.0,
            1.0,
        ]
        quad_indices = [0, 1, 2, 2, 3, 0]

        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        self.quad_ebo = glGenBuffers(1)

        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            len(quad_vertices) * 4,
            (ctypes.c_float * len(quad_vertices))(*quad_vertices),
            GL_STATIC_DRAW,
        )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            len(quad_indices) * 4,
            (ctypes.c_uint * len(quad_indices))(*quad_indices),
            GL_STATIC_DRAW,
        )
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))

        cube_vertices = [
            # Front
            -1,
            -1,
            1,
            0,
            0,
            1,
            -1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            1,
            0,
            1,
            # Back
            -1,
            -1,
            -1,
            1,
            0,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            0,
            1,
            1,
            -1,
            -1,
            0,
            0,
            # Left
            -1,
            -1,
            -1,
            0,
            0,
            -1,
            -1,
            1,
            1,
            0,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            0,
            1,
            # Right
            1,
            -1,
            -1,
            1,
            0,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            -1,
            1,
            0,
            0,
            # Top
            -1,
            1,
            -1,
            0,
            0,
            -1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            0,
            # Bottom
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            0,
            1,
            1,
            -1,
            1,
            0,
            0,
            -1,
            -1,
            1,
            1,
            0,
        ]

        cube_indices: list[int] = []
        for face in range(6):
            offset = face * 4
            cube_indices.extend(
                [offset, offset + 1, offset + 2, offset + 2, offset + 3, offset]
            )

        self.cube_vao = glGenVertexArrays(1)
        self.cube_vbo = glGenBuffers(1)
        self.cube_ebo = glGenBuffers(1)

        glBindVertexArray(self.cube_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            len(cube_vertices) * 4,
            (ctypes.c_float * len(cube_vertices))(*cube_vertices),
            GL_STATIC_DRAW,
        )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube_ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            len(cube_indices) * 4,
            (ctypes.c_uint * len(cube_indices))(*cube_indices),
            GL_STATIC_DRAW,
        )
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))

        glBindVertexArray(0)

    def _init_fbos(self) -> None:
        # Clean up old
        for fbo, tex, depth in self.fbos:
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            glDeleteRenderbuffers(1, [depth])
        self.fbos.clear()

        width = max(1, self.width())
        height = max(1, self.height())
        for _ in range(2):
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                width,
                height,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                None,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            depth = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depth)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)

            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0
            )
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth
            )

            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"FBO incomplete, status {status}")

            self.fbos.append((fbo, tex, depth))

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _target_for_pass(
        self, default_fbo: int, is_last: bool, idx: int
    ) -> tuple[int, Optional[int]]:
        if is_last:
            return default_fbo, None
        fbo, tex, _depth = self.fbos[idx % len(self.fbos)]
        return fbo, tex

    def _compile_pass(self, shader_pass: ShaderPass) -> None:
        vertex_source = None
        if shader_pass.shader_type == "2d":
            vertex_source = DEFAULT_2D_VERTEX
        elif shader_pass.vertex_path and shader_pass.vertex_path.exists():
            vertex_source = shader_pass.vertex_path.read_text()
        else:
            raise ValueError("3D shader requires a vertex shader file")

        fragment_source = shader_pass.fragment_path.read_text()
        print(self.pack_root)
        # Resolve includes relative to each shader file
        if shader_pass.vertex_path:
            vertex_source = _resolve_includes(
                vertex_source,
                self.pack_root,
                {shader_pass.vertex_path},
            )
        fragment_source = _resolve_includes(
            fragment_source,
            self.pack_root,
            {shader_pass.fragment_path},
        )

        vertex_source, fragment_source = modify_shader_sources(
            vertex_source, fragment_source
        )

        sView(vertex_source, fragment_source)

        shader_pass.program = self._create_program(vertex_source, fragment_source)

    def _create_program(self, vertex_src: str, fragment_src: str) -> int:
        vert = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vert, vertex_src)
        glCompileShader(vert)
        self._check_compile(vert, "vertex", vertex_src)

        frag = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(frag, fragment_src)
        glCompileShader(frag)
        self._check_compile(frag, "fragment", fragment_src)

        program = glCreateProgram()
        glAttachShader(program, vert)
        glAttachShader(program, frag)
        glLinkProgram(program)
        self._check_link(program)

        glDeleteShader(vert)
        glDeleteShader(frag)
        return program

    def _format_shader_snippet(self, source: str, line_no: int, radius: int = 4) -> str:
        lines = source.splitlines()
        start = max(0, line_no - radius - 1)
        end = min(len(lines), line_no + radius)
        snippet = []
        for idx in range(start, end):
            prefix = "->" if idx + 1 == line_no else "  "
            snippet.append(f"{prefix} {idx + 1:04d}: {lines[idx]}")
        return "\n".join(snippet)

    def _check_compile(self, shader: int, stage: str, source: str) -> None:
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if status != GL_TRUE:
            log = glGetShaderInfoLog(shader).decode()
            line_no = None
            for token in log.split():
                if token.startswith("0:"):
                    try:
                        line_no = int(token.split(":")[1])
                        break
                    except ValueError:
                        line_no = None
            snippet = self._format_shader_snippet(source, line_no, 4) if line_no else ""
            raise RuntimeError(f"{stage} shader error: {log}\n{snippet}")

    def _check_link(self, program: int) -> None:
        status = glGetProgramiv(program, GL_LINK_STATUS)
        if status != GL_TRUE:
            log = glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"program link error: {log}")

    def _apply_common_uniforms(
        self, shader_pass: ShaderPass, last_texture: Optional[int]
    ) -> None:
        if shader_pass.program is None:
            return
        now = time.time() - self.start_time
        loc_time = glGetUniformLocation(shader_pass.program, "u_time")
        if loc_time != -1:
            glUniform1f(loc_time, float(now))

        loc_res = glGetUniformLocation(shader_pass.program, "u_resolution")
        if loc_res != -1:
            glUniform2f(loc_res, float(self.width()), float(self.height()))

        if last_texture is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, last_texture)
            loc_prev = glGetUniformLocation(shader_pass.program, "u_prev")
            if loc_prev != -1:
                glUniform1i(loc_prev, 0)

        if shader_pass.uniform_name:
            loc_strength = glGetUniformLocation(
                shader_pass.program, shader_pass.uniform_name
            )
            if loc_strength != -1:
                glUniform1f(loc_strength, float(shader_pass.strength))

    def _draw_quad(self) -> None:
        if self.quad_vao is None:
            return
        glBindVertexArray(self.quad_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _draw_cube(self, shader_pass: ShaderPass) -> None:
        if self.cube_vao is None:
            return
        mvp = self._mvp_matrix()
        loc_mvp = glGetUniformLocation(shader_pass.program, "u_mvp")
        if loc_mvp != -1:
            data = (ctypes.c_float * 16)()
            mvp.copyDataTo(data)
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, data)

        glBindVertexArray(self.cube_vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _mvp_matrix(self) -> QMatrix4x4:
        aspect = max(0.1, self.width() / max(1.0, float(self.height())))
        projection = QMatrix4x4()
        projection.perspective(45.0, aspect, 0.1, 100.0)

        view = QMatrix4x4()
        view.translate(0.0, 0.0, -4.0)
        view.rotate(time.time() * 15.0, QVector3D(0.2, 1.0, 0.3))

        model = QMatrix4x4()
        model.rotate(time.time() * 10.0, QVector3D(1.0, 0.4, 0.2))

        return projection * view * model

    def _blit_texture(self, texture: int) -> None:
        # Use a tiny inline program to present the texture to the default FBO.
        vertex_src = DEFAULT_2D_VERTEX
        fragment_src = """
        #version 330 core
        out vec4 FragColor;
        in vec2 v_uv;
        uniform sampler2D u_prev;
        void main() {
            FragColor = texture(u_prev, v_uv);
        }
        """
        program = self._create_program(vertex_src, fragment_src)
        glUseProgram(program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        loc_prev = glGetUniformLocation(program, "u_prev")
        if loc_prev != -1:
            glUniform1i(loc_prev, 0)
        self._draw_quad()
        glDeleteProgram(program)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # Clean up GL resources
        for shader_pass in self.passes:
            if shader_pass.program:
                glDeleteProgram(shader_pass.program)
        for fbo, tex, depth in self.fbos:
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            glDeleteRenderbuffers(1, [depth])
        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
        if self.quad_vbo:
            glDeleteBuffers(1, [self.quad_vbo])
        if self.quad_ebo:
            glDeleteBuffers(1, [self.quad_ebo])
        if self.cube_vao:
            glDeleteVertexArrays(1, [self.cube_vao])
        if self.cube_vbo:
            glDeleteBuffers(1, [self.cube_vbo])
        if self.cube_ebo:
            glDeleteBuffers(1, [self.cube_ebo])
        super().closeEvent(event)


class ShaderViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Shader Stack Viewer")

        self.pack_root = self._choose_pack()
        if self.pack_root is None:
            raise SystemExit(0)

        self.canvas = GLCanvas(self.pack_root)
        self.shader_list = QListWidget()
        self.uniform_name_edit = QLineEdit("u_strength")
        self.uniform_value_spin = QDoubleSpinBox()
        self.uniform_value_spin.setRange(-1000.0, 1000.0)
        self.uniform_value_spin.setSingleStep(0.1)
        self.uniform_value_spin.setValue(1.0)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        add_3d_btn = QPushButton("Add 3D pass")
        add_2d_btn = QPushButton("Add 2D pass")
        remove_btn = QPushButton("Remove")
        move_up_btn = QPushButton("Move Up")
        move_down_btn = QPushButton("Move Down")

        add_3d_btn.clicked.connect(self._on_add_3d)
        add_2d_btn.clicked.connect(self._on_add_2d)
        remove_btn.clicked.connect(self._on_remove)
        move_up_btn.clicked.connect(lambda: self._on_move(-1))
        move_down_btn.clicked.connect(lambda: self._on_move(1))

        form = QFormLayout()
        form.addRow(QLabel("Uniform name"), self.uniform_name_edit)
        form.addRow(QLabel("Strength"), self.uniform_value_spin)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(add_3d_btn)
        buttons_row.addWidget(add_2d_btn)

        move_row = QHBoxLayout()
        move_row.addWidget(move_up_btn)
        move_row.addWidget(move_down_btn)

        right_panel = QVBoxLayout()
        right_panel.addLayout(buttons_row)
        right_panel.addWidget(QLabel("Shader stack"))
        right_panel.addWidget(self.shader_list)
        right_panel.addWidget(remove_btn)
        right_panel.addLayout(move_row)
        right_panel.addWidget(QLabel("Per-pass controls"))
        right_panel.addLayout(form)
        right_panel.addStretch(1)

        root = QHBoxLayout()
        root.addWidget(self.canvas, 3)
        side = QWidget()
        side.setLayout(right_panel)
        root.addWidget(side, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def _connect_signals(self) -> None:
        self.shader_list.currentRowChanged.connect(self._on_select)
        self.uniform_name_edit.editingFinished.connect(self._on_uniform_changed)
        self.uniform_value_spin.valueChanged.connect(self._on_uniform_changed)

    def _choose_pack(self) -> Optional[Path]:
        packs = [p for p in SHADER_PATH.iterdir() if p.is_dir()]
        if not packs:
            QMessageBox.critical(
                self, "No packs", f"No shader packs found in {SHADER_PATH}"
            )
            return None

        dialog = QDialog(self)
        dialog.setWindowTitle("Select shader pack")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Choose a shader pack"))
        combo = QComboBox(dialog)
        for p in packs:
            combo.addItem(p.name, p)
        layout.addWidget(combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog
        )
        layout.addWidget(buttons)

        chosen: Optional[Path] = None

        def accept() -> None:
            nonlocal chosen
            idx = combo.currentIndex()
            chosen = combo.itemData(idx)
            dialog.accept()

        def reject() -> None:
            dialog.reject()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(reject)

        if dialog.exec() == QDialog.Accepted and chosen:
            print(f"Selected shader pack: {chosen}")
            return Path(chosen).absolute()
        return None

    def _on_add_3d(self) -> None:
        vert_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select vertex shader",
            str(self.pack_root),
            "GLSL Files (*.glsl *.vert)",
        )
        if not vert_path:
            return
        frag_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select fragment shader",
            str(Path(vert_path).parent),
            "GLSL Files (*.glsl *.frag)",
        )
        if not frag_path:
            return
        self._add_pass(Path(vert_path), Path(frag_path), "3d")

    def _on_add_2d(self) -> None:
        frag_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 2D fragment shader",
            str(self.pack_root),
            "GLSL Files (*.glsl *.frag)",
        )
        if not frag_path:
            return
        self._add_pass(None, Path(frag_path), "2d")

    def _add_pass(
        self, vertex: Optional[Path], fragment: Path, shader_type: str
    ) -> None:
        name = fragment.stem
        shader_pass = ShaderPass(
            name=name,
            shader_type=shader_type,
            vertex_path=vertex,
            fragment_path=fragment,
        )
        try:
            self.canvas.add_shader_pass(shader_pass)
        except Exception as exc:  # noqa: BLE001
            # Do not keep the pass if compilation failed.
            if shader_pass in self.canvas.passes:
                self.canvas.passes.remove(shader_pass)
            QMessageBox.critical(self, "Shader error", str(exc))
            return
        self._refresh_list()
        self.shader_list.setCurrentRow(len(self.canvas.passes) - 1)

    def _on_remove(self) -> None:
        row = self.shader_list.currentRow()
        if row >= 0:
            self.canvas.remove_shader_pass(row)
            self._refresh_list()

    def _on_move(self, offset: int) -> None:
        row = self.shader_list.currentRow()
        if row >= 0:
            self.canvas.move_shader_pass(row, offset)
            self._refresh_list()
            self.shader_list.setCurrentRow(row + offset)

    def _on_select(self, row: int) -> None:
        if 0 <= row < len(self.canvas.passes):
            shader_pass = self.canvas.passes[row]
            self.uniform_name_edit.setText(shader_pass.uniform_name)
            self.uniform_value_spin.blockSignals(True)
            self.uniform_value_spin.setValue(shader_pass.strength)
            self.uniform_value_spin.blockSignals(False)

    def _on_uniform_changed(self) -> None:
        row = self.shader_list.currentRow()
        if 0 <= row < len(self.canvas.passes):
            shader_pass = self.canvas.passes[row]
            shader_pass.uniform_name = self.uniform_name_edit.text()
            shader_pass.strength = float(self.uniform_value_spin.value())
            self.canvas.update()

    def _refresh_list(self) -> None:
        self.shader_list.clear()
        for shader_pass in self.canvas.passes:
            item = QListWidgetItem(f"{shader_pass.name} ({shader_pass.shader_type})")
            self.shader_list.addItem(item)


def main() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    viewer = ShaderViewer()
    viewer.resize(1280, 800)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
