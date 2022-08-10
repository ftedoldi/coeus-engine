#version 450 core

layout (location = 0) out vec4 fragColor;
  
in vec2 Frag_UV;
// flat in int ObjectID;

uniform sampler2D screenTexture;
// uniform int ObjectID;

void main()
{ 
    float gamma = 2.2;
    float exposure = 1.0;
    
    vec4 color = texture(screenTexture, Frag_UV.st);
    // HDR tonemapping
    color.rgb = vec3(1.0) - exp(-color.rgb * exposure);
    // gamma correction
    color.rgb = pow(color.rgb, vec3(1.0 / gamma));
    // f_objectID = ObjectID;
    fragColor = vec4(color.rgb, 1.0);
}