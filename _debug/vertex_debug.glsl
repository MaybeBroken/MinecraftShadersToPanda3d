#version 150

//Varyings//
out vec2 texCoord, lmCoord;

out vec3 normal;
out vec3 sunVec, upVec, eastVec;

out vec4 color;

//Uniforms//
uniform int worldTime;

uniform float frameTimeCounter;
uniform float timeAngle;

uniform vec3 cameraPosition;

uniform mat4 gbufferModelView, gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;

#ifdef TAA
uniform int frameCounter;

uniform float viewWidth, viewHeight;