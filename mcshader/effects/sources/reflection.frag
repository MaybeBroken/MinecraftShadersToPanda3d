#version 150
// Fresnel-weighted environment reflection, the per-object cousin of Minecraft's
// screen-space water/metal reflections. Samples a cubemap the engine supplies
// (the adapter binds a flat sky-coloured fallback when none is set).
uniform sampler2D  p3d_Texture0;
uniform samplerCube u_env_map;
uniform vec3  u_camera_pos;    // world-space camera position
uniform float u_reflectivity;  // base reflectance at normal incidence
uniform float u_fresnel_power; // sharpness of the grazing-angle boost
uniform vec3  u_reflect_tint;
in vec2 v_texcoord;
in vec4 v_color;
in vec3 v_world_normal;
in vec3 v_world_pos;
out vec4 fragColor;

void main() {
    vec4 base = texture(p3d_Texture0, v_texcoord) * v_color;
    vec3 n        = normalize(v_world_normal);
    vec3 view_dir = normalize(v_world_pos - u_camera_pos);
    vec3 r        = reflect(view_dir, n);
    vec3 env      = texture(u_env_map, r).rgb * u_reflect_tint;
    float fresnel = u_reflectivity
        + (1.0 - u_reflectivity) * pow(1.0 - max(dot(-view_dir, n), 0.0), u_fresnel_power);
    fragColor = vec4(mix(base.rgb, env, clamp(fresnel, 0.0, 1.0)), base.a);
}
