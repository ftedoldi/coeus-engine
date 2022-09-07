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
} fs_in;

flat in float vID;

uniform vec3 viewPos;

uniform Material material;
// uniform DirectionalLight dirLight;
// uniform PointLight pointLight;
// uniform SpotLight spotLight;

#define MAX_LIGHTS 32
uniform int numberOfPointLights;
uniform PointLight pointLights[MAX_LIGHTS];

uniform int numberOfSpotLights;
uniform SpotLight spotLights[MAX_LIGHTS];

uniform int numberOfAreaLights;
uniform AreaLight areaLights[4];

uniform int numberOfDirectionalLights;
uniform DirectionalLight directionalLights[8];

uniform bool hasTexture;
uniform bool hasNormalTexture;

vec3 getNormalFromMap()
{
    //calculate normals in tangent space, based on normalMap
    vec3 tangentNormal = texture(material.normalTex, fs_in.TexCoords).xyz * 2.0 - 1.0;

    //calculate partial derivaties
    vec3 Q1  = dFdx(fs_in.FragPos);
    vec3 Q2  = dFdy(fs_in.FragPos);
    vec2 st1 = dFdx(fs_in.TexCoords);
    vec2 st2 = dFdy(fs_in.TexCoords);

    //calculater TBN matrix
    vec3 N   = normalize(fs_in.Normal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

vec3 calcDirectionalLight(Material material, DirectionalLight light, vec3 viewDir);
vec3 calcPointLight(Material material, PointLight light, vec3 viewDir);
vec3 calcSpotLight(Material material, SpotLight light, vec3 viewDir);

void main()
{
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 result = vec3(0, 0, 0);

    // result += calcDirectionalLight(material, dirLight, viewDir);
    // result += calcPointLight(material, pointLight, viewDir);

    // PointLight pL;
    // pL.position = vec3(0, 0, 0);
    // pL.ambient = vec3(0.1, 0.1, 0.1);
    // pL.diffuse = vec3(0.34, 0.23, 0.67);
    // pL.specular = vec3(0, 0, 0);
    // pL.constant = 0.1;
    // pL.linear = 0.2;
    // pL.quadratic = 0;

    // result += calcPointLight(material, pL, viewDir);
    // result += calcSpotLight(material, spotLight, viewDir);

    for(int i = 0; i < numberOfPointLights; ++i)
        result += calcPointLight(material, pointLights[i], viewDir);

    for(int i = 0; i < numberOfSpotLights; ++i)
        result += calcSpotLight(material, spotLights[i], viewDir);

    for(int i = 0; i < numberOfDirectionalLights; ++i)
        result += calcDirectionalLight(material, directionalLights[i], viewDir);

    for (int i = 0; i < numberOfAreaLights; i++)
        for (int j = 0; j < areaLights[i].numberOfPointLights; j++)
            result += calcPointLight(material, areaLights[i].pointLights[j], viewDir);

    idColor = vec4(vID);
    FragColor = vec4(result, 1.0);
}

vec3 calcDirectionalLight(Material material, DirectionalLight light, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexSpecular;
    float matTexShininess;
    vec3 norm;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, fs_in.TexCoords).rgb;
        matTexSpecular = texture(material.specularTex, fs_in.TexCoords).rgb;
        matTexShininess = 32;
        if(hasNormalTexture)
            norm = getNormalFromMap();
        else
            norm = normalize(fs_in.Normal);
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
        norm = normalize(fs_in.Normal);
    }
    
    //light direction is calculated only by the choosed direction
    vec3 lightDir = normalize(-light.direction);
    //diffuse 
    //the direction of light now is not calculated with the light position, but with the light direction
    
    float diff = max(dot(norm, lightDir), 0.0);
    
    //specular with halfway direction by Blinn
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), matTexShininess);

    if(diff == 0.0)
        spec = 0.0;
    
    vec3 diffuse = light.diffuse * diff * matTexDiffuse;
    vec3 specular = light.specular * spec * matTexSpecular;
    vec3 ambient = light.ambient * matTexDiffuse;
        
    return (ambient + diffuse + specular);
}

vec3 calcPointLight(Material material, PointLight light, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexSpecular;
    float matTexShininess;
    vec3 norm;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, fs_in.TexCoords).rgb;
        matTexSpecular = texture(material.specularTex, fs_in.TexCoords).rgb;
        matTexShininess = 32;
        if(hasNormalTexture)
            norm = getNormalFromMap();
        else
            norm = normalize(fs_in.Normal);    
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
        norm = normalize(fs_in.Normal);
    }
    //normalizing vectors to obtain unit vectors since we only need direction
    vec3 lightDir = normalize(light.position - fs_in.FragPos);

    //diffuse
    //we use max between the dot product and 0 to make sure the value of diff is not negative (if we got 0, we get a black object with no light)
    float diff = max(dot(lightDir, norm), 0.0);

    //specular with halfway direction by Blinn
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), matTexShininess);

    //revomes the specular contribuition if the diffuse contribution is 0
    if(diff == 0.0)
        spec = 0.0;

    //attenuation coefficent
    //attenuation is used to obtain smoother edges simulating real lights
    float distance = length(light.position - fs_in.FragPos);
    float attenuation = 1 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    vec3 ambient = light.ambient * matTexDiffuse;
    vec3 diffuse = light.diffuse * diff * matTexDiffuse;
    vec3 specular = light.specular * spec * matTexSpecular;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}

vec3 calcSpotLight(Material material, SpotLight light, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexSpecular;
    float matTexShininess;
    vec3 norm;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, fs_in.TexCoords).rgb;
        matTexSpecular = texture(material.specularTex, fs_in.TexCoords).rgb;
        matTexShininess = 32;
        if(hasNormalTexture)
            norm = getNormalFromMap();
        else
            norm = normalize(fs_in.Normal);
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
        norm = normalize(fs_in.Normal);
    }

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
        vec3 ambient = light.ambient * matTexDiffuse;

        //diffuse
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = spotFactor * light.diffuse * diff * matTexDiffuse;

        //specular with halfway direction by Blinn
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(norm, halfwayDir), 0.0), matTexShininess);
        if(diff == 0.0)
            spec = 0.0;
        vec3 specular = spotFactor * light.specular * spec * matTexSpecular;

        ambient *= attenuation;
        specular *= attenuation;
        diffuse*= attenuation;
            
        return(ambient + diffuse + specular);
    }
    else
    {
        return matTexDiffuse * light.ambient * attenuation;
    }
}