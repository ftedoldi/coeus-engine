#version 410 core
out vec4 FragColor;
  
in vec2 TexCoords;

struct Material
{
    vec3 diffuse;
    vec3 specular;
    vec3 ambient;
    float shininess;
};

struct MaterialTex
{
    sampler2D diffuse1;
    sampler2D specular1;
    sampler2D shininess;
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


in vec3 FragPos;
in vec3 Normal; 

uniform vec3 viewPos;
//uniform sampler2D diffuse1;

uniform Material material;
uniform MaterialTex materialTex;
uniform DirectionalLight dirLight;
uniform PointLight pointLight;
uniform SpotLight spotLight;

uniform bool hasTexture;

vec3 calcDirectionalLight(Material material, DirectionalLight ligth, vec3 normal, vec3 viewDir);
vec3 calcPointLight(Material material, PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 calcSpotLight(Material material, SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

void main()
{
    if(hasTexture)
    {
        //TODO calculate light on textured objects

        //instead of using diffuse materials, we use the diffuse texture converted to rgb
        //ambient
        /*vec3 ambient = pointLight.ambient * texture(materialTex.diffuse1, TexCoords).rgb;
  	
        // diffuse 
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(pointLight.position - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = pointLight.diffuse * diff * texture(materialTex.diffuse1, TexCoords).rgb;
        vec3 viewDir = normalize(viewPos - FragPos);

        vec3 reflectDir = reflect(-lightDir, norm);
        //power of the shininess coefficient. The higher it is, the more the object will reflect, so we get smaller light concentrated
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), float(texture(materialTex.shininess, TexCoords).rgb));
        vec3 specular = pointLight.specular * (spec * texture(materialTex.specular1, TexCoords).rgb); 
            
        vec3 result = ambient + diffuse + specular;*/
        FragColor = vec4(texture(materialTex.diffuse1, TexCoords).rgb, 1.0);
    }
    else
    {
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 result = vec3(0.0, 0.0, 0.0);

        //result += calcDirectionalLight(material, dirLight, Normal, viewDir);
        //result += calcPointLight(material, pointLight, Normal, FragPos, viewDir);
        result += calcSpotLight(material, spotLight, Normal, FragPos, viewDir);

        FragColor = vec4(result, 1.0);
    }
}

vec3 calcDirectionalLight(Material material, DirectionalLight light, vec3 normal, vec3 viewDir)
{
    vec3 ambient = light.ambient * material.ambient;
    
    //diffuse 
    //the direction of light now is not calculated with the light position, but with the light direction
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(-light.direction);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * material.diffuse);
    
    //specular
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);  
        
    return (ambient + diffuse + specular);
}

vec3 calcPointLight(Material material, PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
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
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

    //attenuation coefficent
    //attenuation is used to obtain smoother edges simulating real lights
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
  			                light.quadratic * (distance * distance));
    
    vec3 ambient = light.ambient * material.diffuse;
    vec3 diffuse = light.diffuse * diff * material.diffuse;
    vec3 specular = light.specular * spec * material.specular;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}

vec3 calcSpotLight(Material material, SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - FragPos);

    float theta = dot(lightDir, normalize(-light.direction));

    //spotFactor is used to modify, based on cutOff angle and spot exponent, intensity of the light
    float spotFactor = 1.0;

    //we check if the light is inside the light cone
    if(theta > light.cutOff)
    {
        spotFactor = pow(theta, light.spotExponent);
        //ambient
        vec3 ambient = light.ambient * material.ambient;

        //diffuse
        vec3 norm = normalize(normal);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = spotFactor * light.diffuse * diff * material.diffuse;

        //specular
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
        vec3 specular = spotFactor * light.specular * spec * material.specular; 
            
        return(ambient + diffuse + specular);
    }
    else
    {
        return material.ambient * light.ambient;
    }
}