#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;

/*struct PointLight
{
    vec3 position;
	vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};*/

uniform vec3 position;
uniform vec3 scale;
uniform vec4 rotation;
uniform mat4 projection;
//uniform PointLight pointLight;
uniform vec3 pointLightPos;

out VS_OUT
{
	vec2 TexCoords;
	vec3 Normal;
	vec3 FragPos;
	vec3 TangentLightPos;
	vec3 TangentFragPos;
} vs_out;

vec4 calcViewPosition(vec3 prova, vec4 conj);

void main()
{
	//TODO make comments
	//Calculate texture coordinates
	vs_out.TexCoords = aTexCoords;
	
	vec4 conj = vec4(-rotation.x, -rotation.y, -rotation.z, rotation.w);

	//Calculate view position
	vec3 scaledVector = vec3(aPos.x * scale.x, aPos.y * scale.y, aPos.z * scale.z);
	vec4 rotatedVector = calcViewPosition(scaledVector, conj);
	vs_out.FragPos = rotatedVector.xyz + position;
	
	//Calculate view position normals
	vec4 rotatedNormal = calcViewPosition(aNormal.xyz, conj);
	vec3 N = normalize(rotatedNormal.xyz);

	vec4 rotatedTang = calcViewPosition(aTangent.xyz, conj);
	vec3 T = normalize(rotatedTang.xyz);

	T = normalize(T - dot(T, N) * N);
	vec3 B = cross(N, T);

	if(dot(cross(N, T), B) < 0.0)
		T = T * -1.0;

	mat3 TBN = transpose(mat3(T, B, N));
	vs_out.TangentLightPos = TBN * pointLightPos;
	vs_out.TangentFragPos = TBN * vs_out.FragPos;
	vs_out.Normal = TBN * rotatedNormal.xyz;

	vec3 viewPos = rotatedVector.xyz + position; // 1 scaling 2 rotation 3 sum translation

	gl_Position = projection * vec4(viewPos, 1.0);
}

vec4 calcViewPosition(vec3 prova, vec4 conj)
{
	vec4 scaledQuaternion = vec4(prova, 0.0);
	vec4 inverseRotation = vec4((conj.w * scaledQuaternion.xyz + conj.xyz * scaledQuaternion.w + cross(conj.xyz, scaledQuaternion.xyz)).xyz,
		conj.w * scaledQuaternion.w - dot(conj.xyz, scaledQuaternion.xyz));
	vec4 rotatedVector = vec4((inverseRotation.w * rotation.xyz + inverseRotation.xyz * rotation.w + cross(inverseRotation.xyz, rotation.xyz)).xyz,
		inverseRotation.w * rotation.w - dot(inverseRotation.xyz, rotation.xyz));

	return rotatedVector;
}