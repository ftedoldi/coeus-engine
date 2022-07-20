#version 410 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;
// layout (location = 2) in vec4 aColor;

out vec2 Frag_UV;
// out vec4 Frag_Color;

uniform mat4 projection;

void main()
{
    Frag_UV = aTexCoords;
    // Frag_Color = aColor;
    gl_Position = projection * vec4(aPos.xy, 0.0, 1.0); 
}  