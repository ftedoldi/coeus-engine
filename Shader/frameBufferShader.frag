#version 410 core

out vec4 FragColor;
  
in vec2 Frag_UV;
// in vec4 Frag_Color;

uniform sampler2D screenTexture;

void main()
{ 
    FragColor = vec4(1 - texture(screenTexture, Frag_UV.st).xyz, 1);
}