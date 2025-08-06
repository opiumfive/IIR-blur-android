#version 310 es

precision highp float;

layout(location = 0) in vec2 aPosision;
layout(location = 1) in vec2 aUV;

out vec2 vUV;

uniform mat4 uMVP;

void main() {
    vUV = aUV;

    gl_Position = uMVP * vec4(aPosision, 0.0, 1.0);
}