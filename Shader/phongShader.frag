#version 450 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 idColor;

struct Material
{
    sampler2D diffuseTex;
    sampler2D specularTex;
    sampler2D normalTex;
    vec3 diffuse;
    vec3 specular;
    float shininess;

    bool hasDiffuseTexture;
    bool hasSpecularTexture;
    bool hasNormalTexture;
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

struct AreaLight
{
    int numberOfPointLights;
    PointLight pointLights[16];
};

in VS_OUT
{
    vec3 Normal;
    vec2 TexCoords;
    vec3 FragPos;
    mat3 TBN;
} fs_in;

flat in float vID;

uniform vec3 cameraPos;

uniform Material material;

#define MAX_LIGHTS 32
uniform int numberOfPointLights;
uniform PointLight pointLights[MAX_LIGHTS];

uniform int numberOfSpotLights;
uniform SpotLight spotLights[MAX_LIGHTS];

uniform int numberOfAreaLights;
uniform AreaLight areaLights[4];

uniform int numberOfDirectionalLights;
uniform DirectionalLight directionalLights[8];

vec3 calcDirectionalLight(vec3 diffuse, vec3 specular, float shininess, vec3 normal, DirectionalLight light, vec3 viewDir);
vec3 calcPointLight(vec3 diffuse, vec3 specular, float shininess, vec3 normal, PointLight light, vec3 viewDir);
vec3 calcSpotLight(vec3 diffuse, vec3 specular, float shininess, vec3 normal, SpotLight light, vec3 viewDir);

void main()
{
    vec3 viewDir = normalize(cameraPos - fs_in.FragPos);
    vec3 result = vec3(0, 0, 0);

    vec3 diffuse = material.hasDiffuseTexture ? texture(material.diffuseTex, fs_in.TexCoords).rgb : material.diffuse;
    vec3 specular = material.hasSpecularTexture ? texture(material.specularTex, fs_in.TexCoords).rgb : material.specular;
    float shininess = material.shininess;

    vec3 normal = vec3(0.0);
    if(material.hasNormalTexture)
    {
        vec3 tangentNormal = texture(material.normalTex, fs_in.TexCoords).xyz;
        tangentNormal = tangentNormal * 2.0 - 1.0;
        normal = normalize(fs_in.TBN * tangentNormal);
    }
    else
    {
        normal = normalize(fs_in.Normal);
    }

    for(int i = 0; i < numberOfPointLights; ++i)
        result += calcPointLight(diffuse, specular, shininess, normal, pointLights[i], viewDir);

    for(int i = 0; i < numberOfSpotLights; ++i)
        result += calcSpotLight(diffuse, specular, shininess, normal, spotLights[i], viewDir);

    for(int i = 0; i < numberOfDirectionalLights; ++i)
        result += calcDirectionalLight(diffuse, specular, shininess, normal, directionalLights[i], viewDir);

    for (int i = 0; i < numberOfAreaLights; i++)
        for (int j = 0; j < areaLights[i].numberOfPointLights; j++)
            result += calcPointLight(diffuse, specular, shininess, normal, areaLights[i].pointLights[j], viewDir);

    idColor = vec4(vID);
    FragColor = vec4(result, 1.0);
}

vec3 calcDirectionalLight(vec3 diffuse, vec3 specular, float shininess, vec3 normal, DirectionalLight light, vec3 viewDir)
{
    //light direction is calculated only by the choosed direction
    vec3 lightDir = normalize(-light.direction);
    //diffuse 
    //the direction of light now is not calculated with the light position, but with the light direction
    float diff = max(dot(normal, lightDir), 0.0);
    
    //specular with halfway direction by Blinn
    vec3 halfwayDir = normalize(lightDir + viewDir);
    //float spec = pow(max(dot(norm, halfwayDir), 0.0), matTexShininess);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    if(diff == 0.0)
        spec = 0.0;
    
    vec3 totalDiffuse = light.diffuse * diff * diffuse;
    vec3 totalSpecular = light.specular * spec * specular;
    vec3 totalAmbient = light.ambient * diffuse;
        
    return (totalAmbient + totalDiffuse + totalSpecular);
}

vec3 calcPointLight(vec3 diffuse, vec3 specular, float shininess, vec3 normal, PointLight light, vec3 viewDir)
{
    //normalizing vectors to obtain unit vectors since we only need direction
    vec3 lightDir = normalize(light.position - fs_in.FragPos);

    //diffuse
    //we use max between the dot product and 0 to make sure the value of diff is not negative (if we got 0, we get a black object with no light)
    float diff = max(dot(lightDir, normal), 0.0);

    //specular with halfway direction by Blinn
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    //revomes the specular contribuition if the diffuse contribution is 0
    if(diff == 0.0)
        spec = 0.0;

    //attenuation coefficent
    //attenuation is used to obtain smoother edges simulating real lights
    float distance = length(light.position - fs_in.FragPos);
    float attenuation = 1 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    vec3 totalAmbient = light.ambient * diffuse;
    vec3 totalDiffuse = light.diffuse * diff * diffuse;
    vec3 totalSpecular = light.specular * spec * specular;

    totalAmbient *= attenuation;
    totalDiffuse *= attenuation;
    totalSpecular *= attenuation;
    return (totalAmbient + totalDiffuse + totalSpecular);
}

vec3 calcSpotLight(vec3 diffuse, vec3 specular, float shininess, vec3 normal, SpotLight light, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fs_in.FragPos);

    float theta = dot(lightDir, normalize(-light.direction));

    //spotFactor is used to modify, based on cutOff angle and spot exponent, intensity of the light
    float spotFactor = 1.0;
    float distance = length(light.position - fs_in.FragPos);
    float attenuation = 1 / (distance * distance);

    //we check if the light is inside the light cone
    if(theta > light.cutOff)
    {
        spotFactor = pow(theta, light.spotExponent);
        //ambient
        vec3 totalAmbient = light.ambient * diffuse;

        //diffuse
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 totalDiffuse = spotFactor * light.diffuse * diff * diffuse;

        //specular with halfway direction by Blinn
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
        if(diff == 0.0)
            spec = 0.0;
        vec3 totalSpecular = spotFactor * light.specular * spec * specular;

        totalAmbient *= attenuation;
        totalSpecular *= attenuation;
        totalDiffuse *= attenuation;
            
        return(totalAmbient + totalDiffuse + totalSpecular);
    }
    else
    {
        return diffuse * light.ambient * attenuation;
    }
}