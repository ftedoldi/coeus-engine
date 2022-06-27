#version 330 core
out vec4 FragColor;
  
//uniform vec3 objectColor;
in vec2 TexCoords;

uniform sampler2D diffuse1;

void main()
{
    FragColor = texture(diffuse1, TexCoords);
}