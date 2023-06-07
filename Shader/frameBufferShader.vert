#version 450 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 Frag_UV;

uniform mat4 projection;

void main()
{
    Frag_UV = aTexCoords;
    gl_Position = projection * vec4(aPos.xy, 0.0, 1.0); 
}  