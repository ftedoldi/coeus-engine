#version 410 core

out vec4 FragColor;
  
in vec2 Frag_UV;
// in vec4 Frag_Color;

uniform sampler2D screenTexture;

void main()
{ 
    float gamma = 2.2;
    float exposure = 1.0;
    
    vec3 color = texture(screenTexture, Frag_UV.st).rgb;
    // HDR tonemapping
    color = vec3(1.0) - exp(-color * exposure);
    // gamma correction
    color = pow(color, vec3(1.0 / gamma));
    FragColor = vec4(color, 1.0);
}