#version 330 core
out vec4 FragColor;

struct Material
{
    sampler2D diffuseTex;
    sampler2D specularTex;
    sampler2D shininessTex;
    sampler2D ambientTex;
    sampler2D normalTex;
    vec3 diffuse;
    vec3 specular;
    vec3 ambient;
    float shininess;
};

struct DirectionalLight
{
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight
{
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

struct SpotLight
{
    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float cutOff;
    float spotExponent;
};

in VS_OUT
{
    vec2 TexCoords;
    vec3 Normal;
    vec3 FragPos;
    vec3 TangentLightPos;
    vec3 TangentFragPos;
} fs_in;

uniform Material material;
uniform DirectionalLight dirLight;
uniform PointLight pointLight;
uniform SpotLight spotLight;

#define MAX_LIGHTS 128
uniform int numLights;
uniform PointLight pointLights[MAX_LIGHTS];

uniform bool hasTexture;

vec3 calcDirectionalLight(Material material, DirectionalLight ligth, vec3 normal, vec3 viewDir);
vec3 calcPointLight(Material material, PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
//vec3 calcSpotLight(Material material, SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

void main()
{
    vec3 viewDir = normalize(-fs_in.TangentFragPos);
    vec3 result = vec3(0.0, 0.0, 0.0);

    //result += calcDirectionalLight(material, dirLight, fs_in.Normal, viewDir);
    result += calcPointLight(material, pointLight, fs_in.Normal, fs_in.TangentFragPos, viewDir);
    //result += calcSpotLight(material, spotLight, fs_in.Normal, fs_in.FragPos, viewDir);

    /*for(int i = 0; i < numLights; ++i)
    {
        result += calcPointLight(material, pointLights[i], fs_in.Normal, fs_in.FragPos, viewDir);
    }*/

    FragColor = vec4(result, 1.0);
}

vec3 calcDirectionalLight(Material material, DirectionalLight light, vec3 normal, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexAmbient;
    vec3 matTexSpecular;
    float matTexShininess;
    vec3 norm;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, fs_in.TexCoords).rgb;
        matTexAmbient = texture(material.ambientTex, fs_in.TexCoords).rgb;
        matTexSpecular = texture(material.specularTex, fs_in.TexCoords).rgb;
        matTexShininess = 32;//texture(material.shininessTex, fs_in.TexCoords).r;
        norm = normalize(texture(material.normalTex, fs_in.TexCoords).rgb);
        //the normal is now in tangent space
        //norm = normalize(norm * 2.0 - vec3(1.0, 1.0, 1.0));
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexAmbient = material.ambient;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
        norm = normalize(normal);
    }
    vec3 ambient = light.ambient * matTexAmbient;
    
    //diffuse 
    //the direction of light now is not calculated with the light position, but with the light direction
    //vec3 norm = normalize(normal);
    vec3 lightDir = normalize(-light.direction);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * matTexDiffuse);
    
    //specular with halfway direction by Blinn
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 32);

    vec3 specular = light.specular * (spec * matTexSpecular);  
        
    return (ambient + diffuse + specular);
}

vec3 calcPointLight(Material material, PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexAmbient;
    vec3 matTexSpecular;
    float matTexShininess;
    vec3 norm;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, fs_in.TexCoords).rgb;
        matTexAmbient = texture(material.ambientTex, fs_in.TexCoords).rgb;
        matTexSpecular = vec3(0.1, 0.1, 0.1);//texture(material.specularTex, fs_in.TexCoords).rgb;
        matTexShininess = 32;//texture(material.shininessTex, fs_in.TexCoords).r;
        norm = texture(material.normalTex, fs_in.TexCoords).rgb;
        norm = normalize(norm * 2.0 - 1.0);
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexAmbient = material.ambient;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
        norm = normalize(normal);
    }
    //normalizing vectors to obtain unit vectors since we only need direction
    //vec3 norm = normal;
    vec3 lightDir = normalize(fs_in.TangentLightPos - fragPos);

    //diffuse
    //we use max between the dot product and 0 to make sure the value of diff is not negative (if we got 0, we get a black object with no light)
    float diff = max(dot(lightDir, norm), 0.0);

    //specular with halfway direction by Blinn
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), matTexShininess);

    //attenuation coefficent
    //attenuation is used to obtain smoother edges simulating real lights
    float distance = length(fs_in.TangentLightPos - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
  			                light.quadratic * (distance * distance));
    
    vec3 ambient = light.ambient * matTexAmbient;
    vec3 diffuse = light.diffuse * diff * matTexDiffuse;
    vec3 specular = light.specular * spec * matTexSpecular;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}

/*vec3 calcSpotLight(Material material, SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexAmbient;
    vec3 matTexSpecular;
    float matTexShininess;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, fs_in.TexCoords).rgb;
        matTexAmbient = texture(material.ambientTex, fs_in.TexCoords).rgb;
        matTexSpecular = texture(material.specularTex, fs_in.TexCoords).rgb;
        matTexShininess = texture(material.shininessTex, fs_in.TexCoords).r;
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexAmbient = material.ambient;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
    }

    vec3 lightDir = normalize(light.position - fragPos);

    float theta = dot(lightDir, normalize(-light.direction));

    //spotFactor is used to modify, based on cutOff angle and spot exponent, intensity of the light
    float spotFactor = 1.0;

    //we check if the light is inside the light cone
    if(theta > light.cutOff)
    {
        spotFactor = pow(theta, light.spotExponent);
        //ambient
        vec3 ambient = light.ambient * matTexAmbient;

        //diffuse
        vec3 norm = normal;
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = spotFactor * light.diffuse * diff * matTexDiffuse;

        //specular with halfway direction by Blinn
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(norm, halfwayDir), 0.0), matTexShininess);
        vec3 specular = spotFactor * light.specular * spec * matTexSpecular; 
            
        return(ambient + diffuse + specular);
    }
    else
    {
        return matTexAmbient * light.ambient;
    }
}*/