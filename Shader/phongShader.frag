#version 330 core
out vec4 FragColor;
  
in vec2 TexCoords;

struct Material
{
    sampler2D diffuseTex;
    sampler2D specularTex;
    sampler2D shininessTex;
    sampler2D ambientTex;
    vec3 diffuse;
    vec3 specular;
    vec3 ambient;
    float shininess;
};

//struct MaterialTex
//{
//    sampler2D diffuse1;
//    sampler2D specular1;
//    sampler2D shininess;
//};

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


in vec3 FragPos;
in vec3 Normal; 

uniform vec3 viewPos;
//uniform sampler2D diffuse1;

uniform Material material;
//uniform MaterialTex materialTex;
uniform DirectionalLight dirLight;
uniform PointLight pointLight;
uniform SpotLight spotLight;

#define MAX_LIGHTS 128
uniform int numLights;
uniform PointLight pointLights[MAX_LIGHTS];

uniform bool hasTexture;

vec3 calcDirectionalLight(Material material, DirectionalLight ligth, vec3 normal, vec3 viewDir);
vec3 calcPointLight(Material material, PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 calcSpotLight(Material material, SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

void main()
{
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 result = vec3(0.0, 0.0, 0.0);

    //result += calcDirectionalLight(material, dirLight, Normal, viewDir);
    result += calcPointLight(material, pointLight, Normal, FragPos, viewDir);
    //result += calcSpotLight(material, spotLight, Normal, FragPos, viewDir);

    /*for(int i = 0; i < numLights; ++i)
    {
        result += calcPointLight(material, pointLights[i], Normal, FragPos, viewDir);
    }*/

    FragColor = vec4(result, 1.0);
}

vec3 calcDirectionalLight(Material material, DirectionalLight light, vec3 normal, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexAmbient;
    vec3 matTexSpecular;
    float matTexShininess;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, TexCoords).rgb;
        matTexAmbient = texture(material.ambientTex, TexCoords).rgb;

        matTexSpecular = texture(material.specularTex, TexCoords).rgb;
        matTexShininess = 20;//texture(material.shininessTex, TexCoords).rgb;
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexAmbient = material.ambient;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
    }
    vec3 ambient = light.ambient * matTexAmbient;
    
    //diffuse 
    //the direction of light now is not calculated with the light position, but with the light direction
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(-light.direction);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * matTexDiffuse);
    
    //specular
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), matTexShininess);
    vec3 specular = light.specular * (spec * matTexSpecular);  
        
    return (ambient + diffuse + specular);
}

vec3 calcPointLight(Material material, PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexAmbient;
    vec3 matTexSpecular;
    float matTexShininess;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, TexCoords).rgb;
        matTexAmbient = texture(material.ambientTex, TexCoords).rgb;

        matTexSpecular = texture(material.specularTex, TexCoords).rgb;
        matTexShininess = 20;//texture(material.shininessTex, TexCoords).rgb;
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexAmbient = material.ambient;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
    }
    //normalizing vectors to obtain unit vectors since we only need direction
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(light.position - fragPos);

    //diffuse
    //we use max between the dot product and 0 to make sure the value of diff is not negative (if we got 0, we get a black object with no light)
    float diff = max(dot(norm, lightDir), 0.0);

    //specular
    //negative lightDir because the reflect function expects the first vector to point from the light source towards the fragment's position
    //lightDir points to the other way: from the fragment to the light source
    vec3 reflectDir = reflect(-lightDir, norm);
    //the higher shininess is, the more the object will reflect, so we get smaller light concentrated
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), matTexShininess);

    //attenuation coefficent
    //attenuation is used to obtain smoother edges simulating real lights
    float distance = length(light.position - fragPos);
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

vec3 calcSpotLight(Material material, SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 matTexDiffuse;
    vec3 matTexAmbient;
    vec3 matTexSpecular;
    float matTexShininess;
    if(hasTexture)
    {
        matTexDiffuse = texture(material.diffuseTex, TexCoords).rgb;
        matTexAmbient = texture(material.ambientTex, TexCoords).rgb;

        matTexSpecular = texture(material.specularTex, TexCoords).rgb;
        matTexShininess = 20;//texture(material.shininessTex, TexCoords).rgb;
    }
    else
    {
        matTexDiffuse = material.diffuse;
        matTexAmbient = material.ambient;
        matTexSpecular = material.specular;
        matTexShininess = material.shininess;
    }

    vec3 lightDir = normalize(light.position - FragPos);

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
        vec3 norm = normalize(normal);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = spotFactor * light.diffuse * diff * matTexDiffuse;

        //specular
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), matTexShininess);
        vec3 specular = spotFactor * light.specular * spec * matTexSpecular; 
            
        return(ambient + diffuse + specular);
    }
    else
    {
        return matTexAmbient * light.ambient;
    }
}