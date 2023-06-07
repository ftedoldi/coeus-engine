#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube environmentMap;

void main()
{    
    vec3 envColor = textureLod(environmentMap, TexCoords, 0.0).rgb;

    FragColor = vec4(envColor, 1.0);
}