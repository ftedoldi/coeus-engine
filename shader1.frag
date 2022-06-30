#version 330 core
out vec4 FragColor;
  
//uniform vec3 objectColor;
in vec2 TexCoords;

struct Material
{
    vec4 diffuse;
    vec4 specular;
    vec4 ambient;
    vec4 shininess;
};

uniform sampler2D diffuse1;
uniform Material material;
uniform bool hasTexture;

void main()
{
    if(hasTexture)
        FragColor = texture(diffuse1, TexCoords);
    else
    {
        vec4 result = material.diffuse;
        FragColor = vec4(result);
    }
    
}