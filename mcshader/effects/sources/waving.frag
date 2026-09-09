#version 150
uniform sampler2D p3d_Texture0;
in vec2 v_texcoord;
in vec4 v_color;
in vec3 v_normal;
out vec4 fragColor;

void main() {
    vec4 tex = texture(p3d_Texture0, v_texcoord);
    fragColor = tex * v_color;
}
