#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix; // model -> world
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;
out vec2 v_texcoord;
out vec4 v_color;
out vec3 v_world_normal;
out vec3 v_world_pos;

void main() {
    v_texcoord = p3d_MultiTexCoord0;
    v_color    = p3d_Color;
    vec4 world = p3d_ModelMatrix * p3d_Vertex;
    v_world_pos    = world.xyz;
    v_world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal); // assumes uniform scale
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
