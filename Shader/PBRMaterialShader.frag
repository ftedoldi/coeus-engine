#version 450 core

layout(location=0) out vec4 FragColor;
layout(location=1) out vec4 idColor;

in vec3 Normal;

// material parameters
struct PBRmaterial
{
    vec4 albedoColor;
    float metallicColor;
    float roughnessColor;
};

uniform PBRmaterial material;

//IBL
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

uniform vec3 cameraPos;

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

struct DirectionalLight
{
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
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

#define MAX_LIGHTS 32
uniform int numberOfPointLights;
uniform PointLight pointLights[MAX_LIGHTS];

uniform int numberOfSpotLights;
uniform SpotLight spotLights[MAX_LIGHTS];

uniform int numberOfDirectionalLights;
uniform DirectionalLight directionalLights[8];

in VS_OUT
{
    vec3 Normal;
    vec2 TexCoords;
    vec3 FragPos;
    mat3 TBN;
} fs_in;

flat in float vID;

const float PI = 3.14159265359;

vec3 CalculateDirectionalLight(vec3 albedo, vec3 N, float metallic, float roughness, vec3 V, vec3 F0);
vec3 CalculatePointLight(vec3 albedo, vec3 N, float metallic, float roughness, vec3 V, vec3 F0);
vec3 CalculateSpotLight(vec3 albedo, vec3 N, float metallic, float roughness, vec3 V, vec3 F0);

//Trowbridge-Reitz GGX
float DistributionGGX(vec3 normal, vec3 halfway, float rough)
{
    float a = rough * rough;
    float a2 = a * a;
    float NdotH = max(dot(normal, halfway), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

//Geometry function
//We use a Schlick-GGX function
float GeometrySchlickGGX(float cosTheta, float roughness)
{
    float r = (roughness + 1.0);
    float k = (roughness * roughness) / 8.0;

    float numerator   = cosTheta;
    float denominator = cosTheta * (1.0 - k) + k;

    return numerator / denominator;
}

//To effectively approximate the geometry we use Schlick-GGX considering view direction and light direction
//and multiply them
float GeometrySmith(vec3 normal, vec3 viewDir, vec3 lightDir, float rough)
{
    float NoV = max(dot(normal, viewDir), 0.0);
    float NoL = max(dot(normal, lightDir), 0.0);
    float ggx1 = GeometrySchlickGGX(NoV, rough);
    float ggx2 = GeometrySchlickGGX(NoL, rough);
	
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    //return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return max(F0 + (1.0 - F0) * pow(2, (-5.55473 * cosTheta - 6.98316) * cosTheta), 0.0);
}

//fresnelSchlick considering roughness parameter
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}  

void main()
{
    vec4 completeAlbedo = material.albedoColor;
    vec3 albedo = completeAlbedo.rgb;
    float albedoAlpha = completeAlbedo.w;
    float roughness = material.roughnessColor;
    roughness = max(roughness, 0.04);
    float metallic = material.metallicColor;
    
    vec3 N = normalize(fs_in.Normal);
    //gltf format
    //float metallic = texture(metallicMap, fs_in.TexCoords).b;
    //float roughness = texture(roughnessMap, fs_in.TexCoords).g;

    vec3 V = normalize(cameraPos - fs_in.FragPos);

    vec3 R = reflect(-V, N);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);
    Lo += CalculateDirectionalLight(albedo, N, metallic, roughness, V, F0);
    Lo += CalculatePointLight(albedo, N, metallic, roughness, V, F0);
    Lo += CalculateSpotLight(albedo, N, metallic, roughness, V, F0);
    
    float ao = 1.0;
    // ambient lighting (we now use IBL as the ambient term)

    //we use fresnel equation to determine the surface indirect reflactance (specular) ratio
    //from which we derive the refractive (diffuse) ratio
    //since ambient light comes from all directions whitin the hemisphere oriented around the normal N
    //there is no halfway vector to determine the Fresnel response.
    //To still simulate it we can calculate the Fresnel from the angle between the normal and view vector.
    //But, since the Fresnel equation is influenced by the roughness and we dont consider it alleviating the
    //issue it cause, by taking it in consideration on the Fresnel equation.
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

    //retrieving the irradiance influenting the fragment
    vec3 irradiance = texture(irradianceMap, N).rgb;
    
    vec3 kD = mix(vec3(1.0) - F, vec3(0.0), metallic);
    
    //we finally calculate the diffuse component by multiplyinng the irradiance by the albedo
    vec3 diffuse      = kD * irradiance * albedo;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R,  roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F0 * brdf.x + brdf.y);

    vec3 ambient = (specular + diffuse) * ao;
    
    vec3 color = ambient + Lo;

    // we do HDR Reinhard tonemapping and gamma correction in post processing
    idColor = vec4(vID);
    FragColor = vec4(color, 1.0);
}

vec3 CalculateDirectionalLight(vec3 albedo, vec3 N, float metallic, float roughness, vec3 V, vec3 F0)
{
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < numberOfDirectionalLights; ++i)
    {
        //calculate radiance
        vec3 L = normalize(-directionalLights[i].direction);
        vec3 H = normalize(V + L);
        vec3 radiance = /*directionalLights[i].ambient **/ directionalLights[i].diffuse;

        //Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.1) * max(dot(N, L), 0.0) + 0.001;
        vec3 specular = numerator / denominator;

        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;	  

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }

    return Lo;
}

vec3 CalculatePointLight(vec3 albedo, vec3 N, float metallic, float roughness, vec3 V, vec3 F0)
{
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < numberOfPointLights; ++i)
    {
        //calculate radiance
        vec3 L = normalize(pointLights[i].position - fs_in.FragPos);
        vec3 H = normalize(V + L);

        float diff = max(dot(L, N), 0.0);

        float distance = length(pointLights[i].position - fs_in.FragPos);
        float attenuation = 1.0 / (pointLights[i].constant + pointLights[i].linear * distance + 
                                pointLights[i].quadratic * (distance * distance));
        vec3 radiance = pointLights[i].diffuse * diff * attenuation;

        //Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.1) * max(dot(N, L), 0.0) + 0.001;
        vec3 specular = numerator / denominator;

        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;	  

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }

    return Lo;
}

vec3 CalculateSpotLight(vec3 albedo, vec3 N, float metallic, float roughness, vec3 V, vec3 F0)
{
    vec3 Lo = vec3(0.0);
    vec3 radiance = vec3(0.0);
    for(int i = 0; i < numberOfSpotLights; ++i)
    {
        //calculate radiance
        vec3 L = normalize(spotLights[i].position - fs_in.FragPos);
        vec3 H = normalize(V + L);

        float theta = dot(L, normalize(-spotLights[i].direction));

        //spotFactor is used to modify, based on cutOff angle and spot exponent, intensity of the light
        float spotFactor = 1.0;
        float distance = length(spotLights[i].position - fs_in.FragPos);
        float attenuation = 1 / (distance * distance);

        //we check if the light is inside the light cone
        if(theta > spotLights[i].cutOff)
        {
            spotFactor = pow(theta, spotLights[i].spotExponent);

            //diffuse
            float diff = max(dot(N, L), 0.0);
            vec3 diffuse = spotFactor * spotLights[i].diffuse * diff * albedo;

            radiance = diffuse /** spotLights[i].ambient*/ * attenuation;
        }
        else
        {
            radiance = albedo /** spotLights[i].ambient*/ * attenuation;
        }

        //Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.1) * max(dot(N, L), 0.0) + 0.001;
        vec3 specular = numerator / denominator;

        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;	  

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }

    return Lo;
}

