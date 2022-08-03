#version 420 core

out vec4 FragColor;
in vec3 Normal;

// material parameters
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

//IBL
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

uniform vec3 viewPos;

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

uniform PointLight pointLight;

in VS_OUT
{
    vec3 Normal;
    vec2 TexCoords;
    vec3 FragPos;
    vec3 LightPos;
} fs_in;

const float PI = 3.14159265359;
const float nMaxLights = 1;

//calculate normals in tangent space and transform them in world space
vec3 getNormalFromMap()
{
    //calculate normals in tangent space, based on normalMap
    vec3 tangentNormal = texture(normalMap, fs_in.TexCoords).xyz * 2.0 - 1.0;

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
float GeometrySchlickGGX(float NdotV, float rough)
{
    float r = (rough + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
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
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

//fresnelSchlick considering roughness parameter
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}  

void main()
{
    vec3 albedo = texture(albedoMap, fs_in.TexCoords).rgb;
    //gltf format
    //float metallic = texture(metallicMap, fs_in.TexCoords).b;
    //float roughness = texture(roughnessMap, fs_in.TexCoords).g;

    //obj file format
    float roughness = texture(roughnessMap, fs_in.TexCoords).r;
    float metallic = texture(metallicMap, fs_in.TexCoords).r;
    //float ao = texture(aoMap, fs_in.TexCoords).r;
    //float roughness = 0;
    //float metallic = 0;

    vec3 N = getNormalFromMap();

    vec3 V = normalize(viewPos - fs_in.FragPos);

    vec3 R = reflect(-V, N);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);
    
    for(int i = 0; i < nMaxLights; ++i)
    {
        //calculate radiance
        vec3 L = normalize(pointLight.position - fs_in.FragPos);
        vec3 H = normalize(V + L);
        float distance = length(pointLight.position - fs_in.FragPos);
        float attenuation = 1.0 / (pointLight.constant + pointLight.linear * distance + 
                                pointLight.quadratic * (distance * distance));
        vec3 radiance = pointLight.diffuse * attenuation;

        //Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // prevent division by 0
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
    
    vec3 kS = F;
    //obtaining the refractive ratio by substracting 1 to the reflactance ratio
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;	  
    
    //retrieving the irradiance influenting the fragment
    vec3 irradiance = texture(irradianceMap, N).rgb;
    //we finally calculate the diffuse component by multiplyinng the irradiance by the albedo
    vec3 diffuse      = irradiance * albedo;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R,  roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;
    
    vec3 color = ambient + Lo;

    // we do HDR Reinhard tonemapping and gamma correction in post processing

    FragColor = vec4(color, 1.0);
}

