"""A canonical fullscreen-quad vertex shader.

Most composite/deferred/final programs ship their own vertex stage (it often
computes handy varyings like the sun vector), and the pipeline uses that. But
some packs omit it and rely on the loader's default. This provides that default:
a passthrough that covers the screen and exposes the ``texcoord`` varying the
OptiFine default supplies. Written in the Panda3D dialect so it drops straight
into ``Shader.make``.
"""

from __future__ import annotations

__all__ = ["FULLSCREEN_VERTEX", "texcoord_name"]

FULLSCREEN_VERTEX = """\
#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 texcoord;

void main() {
    texcoord = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""


def texcoord_name(fragment_src: str) -> str:
    """Best-guess the texcoord varying a fragment expects ("texcoord"/"texCoord").

    Lets a caller align the fallback vertex's output with the fragment's input
    when a pack uses the capitalised spelling.
    """
    return "texCoord" if "texCoord" in fragment_src else "texcoord"
