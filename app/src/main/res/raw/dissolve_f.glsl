#version 300 es

precision highp float;

in float alpha;
out vec4 fragColor;

uniform sampler2D u_Texture;
in vec2 texCoord;

void main() {
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0) {
        discard;
    }
    vec4 textureColor = texture(u_Texture, texCoord);
    fragColor = vec4(textureColor.rgb, textureColor.a * alpha);
}