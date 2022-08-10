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
} vs_out;

flat out float vID;

vec4 calcViewPosition(vec3 vector, vec4 conj);
vec4 calcWorldPosition(vec3 vector, vec4 worldConj);

void main()
{
	//TODO make comments
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
	vs_out.Normal = normalize(rotatedNormal.xyz);

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
	vec4 inverseRotation = vec4((conj.w * scaledQuaternion.xyz + conj.xyz * scaledQuaternion.w + cross(conj.xyz, scaledQuaternion.xyz)).xyz,
		conj.w * scaledQuaternion.w - dot(conj.xyz, scaledQuaternion.xyz));
	vec4 rotatedVector = vec4((inverseRotation.w * rotation.xyz + inverseRotation.xyz * rotation.w + cross(inverseRotation.xyz, rotation.xyz)).xyz,
		inverseRotation.w * rotation.w - dot(inverseRotation.xyz, rotation.xyz));

	return rotatedVector;
}

vec4 calcWorldPosition(vec3 vector, vec4 worldConj)
{
	vec4 scaledQuaternion = vec4(vector, 0.0);
	vec4 inverseRotation = vec4((worldConj.w * scaledQuaternion.xyz + worldConj.xyz * scaledQuaternion.w + cross(worldConj.xyz, scaledQuaternion.xyz)).xyz,
		worldConj.w * scaledQuaternion.w - dot(worldConj.xyz, scaledQuaternion.xyz));
	vec4 rotatedVector = vec4((inverseRotation.w * WorldRotation.xyz + inverseRotation.xyz * WorldRotation.w + cross(inverseRotation.xyz, WorldRotation.xyz)).xyz,
		inverseRotation.w * WorldRotation.w - dot(inverseRotation.xyz, WorldRotation.xyz));

	return rotatedVector;
}