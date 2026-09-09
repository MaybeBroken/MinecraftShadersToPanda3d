#version 150
// Derived from Minecraft gbuffers_terrain waving-grass/leaves vertex logic:
// a horizontal sway that grows toward the top of the geometry.
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;          // model -> world
uniform mat4 p3d_ViewProjectionMatrix; // world -> clip
uniform mat3 p3d_NormalMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;

uniform float u_time;
uniform float u_wave_amplitude;
uniform float u_wave_speed;
uniform float u_wave_scale;

out vec2 v_texcoord;
out vec4 v_color;
out vec3 v_normal;

void main() {
    vec4 world = p3d_ModelMatrix * p3d_Vertex;
    float phase = (world.x + world.z) * u_wave_scale + u_time * u_wave_speed;
    float sway  = sin(phase) + 0.35 * sin(phase * 2.7 + 1.3);
    // Vertical UV as a cheap "distance from the ground" mask, like MC does.
    float mask  = clamp(p3d_MultiTexCoord0.y, 0.0, 1.0);
    world.x += sway * u_wave_amplitude * mask;
    world.z += cos(phase * 0.9) * u_wave_amplitude * 0.5 * mask;

    v_texcoord = p3d_MultiTexCoord0;
    v_color    = p3d_Color;
    v_normal   = p3d_NormalMatrix * p3d_Normal;
    gl_Position = p3d_ViewProjectionMatrix * world;
}
