#version 330 core
out vec4 FragColor;
  
//uniform vec3 objectColor;
in vec2 TexCoords;

uniform sampler2D diffuse1;
uniform vec4 materialDiff;
//uniform vec4 materialSpec;
//uniform vec4 materialAmb;
//uniform vec4 materialShin;
uniform bool hasTexture;

void main()
{
    if(hasTexture)
        FragColor = texture(diffuse1, TexCoords);
    else
    {
        vec4 result = materialDiff;
        FragColor = vec4(result);
    }
    
}