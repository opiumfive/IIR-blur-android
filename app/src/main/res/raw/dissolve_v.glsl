#version 300 es

precision highp float;

layout(location = 0) in vec2 inOffset;
layout(location = 1) in vec2 inVelocity;
layout(location = 2) in float inLifetime;
layout(location = 3) in float inDuration;

out vec2 outOffset;
out vec2 outVelocity;
out float outLifetime;
out float outDuration;
out float alpha;
out vec2 texCoord;

uniform float reset;
uniform float time;
uniform float deltaTime;
uniform vec2 pointsOffset;
uniform vec2 pointsSize;
uniform vec2 pointsCount;
uniform vec2 size;
uniform float r;
uniform float seed;
uniform float noiseScale;
uniform float noiseSpeed;
uniform float noiseMovement;
uniform float dampingMult;
uniform float forceMult;
uniform float velocityMult;
uniform float longevity;
uniform float maxVelocity;

float rand(vec2 n) {
    return fract(sin(dot(n,vec2(0.1898,0.1414-seed*.42)))*58.5453);
}

float rand() {
    return rand(vec2(gl_VertexID, gl_VertexID));
}

vec4 toGlobalCoordinates(vec2 pos) {
    float x = pos.x * 2.0 - 1.0;
    float y = (1.0 - pos.y) * 2.0 - 1.0;
    return vec4(x, y, 0.0, 1.0);
}

float getIDX() {
    return mod(float(gl_VertexID), pointsCount.x);
}

vec2 texturePosition(int id) {
    float countX = pointsCount.x;
    float countY = pointsCount.y;
    float idx = getIDX();
    float x = (idx / countX);
    float y = (float(id - int(idx)) / (countX * countY));
    return vec2(x, y);
}

vec2 startPos(int id) {
    float countX = pointsCount.x;
    float countY = pointsCount.y;
    float idx = getIDX();
    float x = pointsOffset.x + (idx / countX) * pointsSize.x;
    float y = pointsOffset.y + (float(id - int(idx)) / (countX * countY)) * pointsSize.y;
    return vec2(x, y);
}

const float PI2 = 2.0 * 3.14159;

vec2 dissolve() {
    float direction = PI2 * rand();
    float distance = 0.2 + rand() * 0.1;
    return vec2(1.5 * cos(direction) * distance, sin(direction) * distance) * 3.14159;
}

void main() {
    vec2 offset = inOffset;
    vec2 velocity = inVelocity;
    float lifetime = inLifetime;
    float duration = inDuration;
    vec2 startPosition = startPos(gl_VertexID);
    vec2 texturePosition = texturePosition(gl_VertexID);

    if (time == 0.0 || reset > 0.) {
        offset = vec2(0.0, 0.0);
        velocity = dissolve();
        lifetime = 0.2 + rand() * 0.8;
    } else {
        float fractionDistance = max(0.0, time - startPosition.x * 0.2);
        float fractionTime = fractionDistance;

        offset += (dissolve() * deltaTime) * fractionDistance + vec2(0.0, -0.001) * fractionDistance;

        if (fractionDistance > 0.5) {
            offset += vec2(0.0, 2.0 * 0.001 * (fractionDistance - 0.5) * 2.0);
        } else {
            offset += vec2(0.0, 0.001 * fractionDistance * 2.0);
        }

        if (fractionTime > 0.0) {
            alpha = min(1.0 - 1.7 * fractionTime / lifetime, 1.0);
            gl_PointSize = r;
        } else {
            alpha = 1.0;
            gl_PointSize = r * velocityMult;
        }
    }

    outOffset = offset;
    outVelocity = velocity;
    outLifetime = lifetime;
    outDuration = duration;

    gl_Position = toGlobalCoordinates(startPosition + offset);
    texCoord = texturePosition;
}