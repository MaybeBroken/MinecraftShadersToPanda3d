#version 150
// Per-object emissive, in the spirit of Minecraft's gbuffers_entities_glowing
// and emissive texture maps. NOTE: this brightens the surface itself; true
// screen-space bloom needs a post-process pass (see README).
uniform sampler2D p3d_Texture0;
uniform float u_time;
uniform vec3  u_glow_color;
uniform float u_glow_strength;
uniform float u_glow_pulse;   // pulses per second; 0 = steady
in vec2 v_texcoord;
in vec4 v_color;
out vec4 fragColor;

void main() {
    vec4 base = texture(p3d_Texture0, v_texcoord) * v_color;
    float pulse = u_glow_pulse > 0.0
        ? 0.5 + 0.5 * sin(u_time * u_glow_pulse * 6.2831853)
        : 1.0;
    float lum = dot(base.rgb, vec3(0.299, 0.587, 0.114));
    vec3 emissive = u_glow_color * u_glow_strength * pulse * (0.35 + 0.65 * lum);
    fragColor = vec4(base.rgb + emissive, base.a);
}
