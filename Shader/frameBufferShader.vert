#version 330 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;
// layout (location = 2) in int aObjectID;

out vec2 Frag_UV;
// flat out int ObjectID;

uniform mat4 projection;

void main()
{
    Frag_UV = aTexCoords;
    // ObjectID = aObjectID;
    gl_Position = projection * vec4(aPos.xy, 0.0, 1.0); 
}  