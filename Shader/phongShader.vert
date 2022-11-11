#version 450 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;

uniform vec3 position;
uniform vec3 scale;
uniform vec4 rotation;
uniform mat4 projection;

uniform vec3 WorldPosition;
uniform vec3 WorldScale;
uniform vec4 WorldRotation;

uniform float ID; 

out VS_OUT
{
	vec3 Normal;
	vec2 TexCoords;
	vec3 FragPos;
	mat3 TBN;
} vs_out;

flat out float vID;

vec4 calcViewPosition(vec3 vector, vec4 conj);
vec4 calcWorldPosition(vec3 vector, vec4 worldConj);

void main()
{
	//Calculate texture coordinates
	vs_out.TexCoords = aTexCoords;
	vID = ID;

	//Calculate conjugate with world rotation, to apply later
	vec4 worldConj = vec4(-WorldRotation.x, -WorldRotation.y, -WorldRotation.z, WorldRotation.w);

	//Calculate world object position
	vec3 scaledWorldVector = vec3(aPos.x * WorldScale.x, aPos.y * WorldScale.y, aPos.z * WorldScale.z);
	vec4 rotatedWorldVector = calcWorldPosition(scaledWorldVector, worldConj);
	vs_out.FragPos = rotatedWorldVector.xyz + WorldPosition;

	//Calculate world object normals
	vec4 rotatedNormal = calcWorldPosition(aNormal.xyz, worldConj);
	vs_out.Normal = rotatedNormal.xyz;

	//Calculate world object tangents
	vec4 rotatedTangent = calcWorldPosition(aTangent.xyz, worldConj);

	vec3 T = normalize(rotatedTangent.xyz);
	vec3 N = normalize(rotatedNormal.xyz);
	T = normalize(T - dot(T, N) * N);
	vec3 B = cross(N, T);
	vs_out.TBN = mat3(T, B, N);

	//Calculate view position
	vec4 conj = vec4(-rotation.x, -rotation.y, -rotation.z, rotation.w);

	vec3 scaledViewVector = vec3(aPos.x * scale.x, aPos.y * scale.y, aPos.z * scale.z);
	vec4 rotatedViewVector = calcViewPosition(scaledViewVector, conj);
	vec3 viewPos = rotatedViewVector.xyz + position; // 1 scaling 2 rotation 3 sum translation

	gl_Position = projection * vec4(viewPos, 1.0);
}

vec4 calcViewPosition(vec3 vector, vec4 conj)
{
	vec4 scaledQuaternion = vec4(vector, 0.0);
	//first hamilton product between quaternion and point quaternion
	//second hamilton product between result and conj
	vec4 firstRotation = vec4(rotation.w * scaledQuaternion.x + rotation.x * scaledQuaternion.w + rotation.y * scaledQuaternion.z - rotation.z * scaledQuaternion.y,
							  rotation.w * scaledQuaternion.y + rotation.y * scaledQuaternion.w + rotation.z * scaledQuaternion.x - rotation.x * scaledQuaternion.z,
							  rotation.w * scaledQuaternion.z + rotation.z * scaledQuaternion.w + rotation.x * scaledQuaternion.y - rotation.y * scaledQuaternion.x,
							  rotation.w * scaledQuaternion.w - rotation.x * scaledQuaternion.x - rotation.y * scaledQuaternion.y - rotation.z * scaledQuaternion.z);
	
	vec4 inverseRotation = vec4(firstRotation.w * conj.x + firstRotation.x * conj.w + firstRotation.y * conj.z - firstRotation.z * conj.y,
							  firstRotation.w * conj.y + firstRotation.y * conj.w + firstRotation.z * conj.x - firstRotation.x * conj.z,
							  firstRotation.w * conj.z + firstRotation.z * conj.w + firstRotation.x * conj.y - firstRotation.y * conj.x,
							  firstRotation.w * conj.w - firstRotation.x * conj.x - firstRotation.y * conj.y - firstRotation.z * conj.z);

	return inverseRotation;
}

vec4 calcWorldPosition(vec3 vector, vec4 worldConj)
{
	vec4 scaledQuaternion = vec4(vector, 0.0);

	vec4 firstRotation = vec4(WorldRotation.w * scaledQuaternion.x + WorldRotation.x * scaledQuaternion.w + WorldRotation.y * scaledQuaternion.z - WorldRotation.z * scaledQuaternion.y,
							  WorldRotation.w * scaledQuaternion.y + WorldRotation.y * scaledQuaternion.w + WorldRotation.z * scaledQuaternion.x - WorldRotation.x * scaledQuaternion.z,
							  WorldRotation.w * scaledQuaternion.z + WorldRotation.z * scaledQuaternion.w + WorldRotation.x * scaledQuaternion.y - WorldRotation.y * scaledQuaternion.x,
							  WorldRotation.w * scaledQuaternion.w - WorldRotation.x * scaledQuaternion.x - WorldRotation.y * scaledQuaternion.y - WorldRotation.z * scaledQuaternion.z);
	
	vec4 inverseRotation = vec4(firstRotation.w * worldConj.x + firstRotation.x * worldConj.w + firstRotation.y * worldConj.z - firstRotation.z * worldConj.y,
							  firstRotation.w * worldConj.y + firstRotation.y * worldConj.w + firstRotation.z * worldConj.x - firstRotation.x * worldConj.z,
							  firstRotation.w * worldConj.z + firstRotation.z * worldConj.w + firstRotation.x * worldConj.y - firstRotation.y * worldConj.x,
							  firstRotation.w * worldConj.w - firstRotation.x * worldConj.x - firstRotation.y * worldConj.y - firstRotation.z * worldConj.z);

	return inverseRotation;
}