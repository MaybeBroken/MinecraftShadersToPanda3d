#version 150
// Animated UV scroll with a sine wobble: flowing water/lava-style movement,
// the technique MC uses for animated block textures and water surfaces.
uniform sampler2D p3d_Texture0;
uniform float u_time;
uniform vec2  u_scroll_speed;  // uv units per second
uniform float u_distort_amp;
uniform float u_distort_freq;
in vec2 v_texcoord;
in vec4 v_color;
out vec4 fragColor;

void main() {
    vec2 uv = v_texcoord + u_scroll_speed * u_time;
    uv.x += sin((v_texcoord.y + u_time * 0.5) * u_distort_freq) * u_distort_amp;
    uv.y += cos((v_texcoord.x + u_time * 0.5) * u_distort_freq) * u_distort_amp;
    fragColor = texture(p3d_Texture0, uv) * v_color;
}
