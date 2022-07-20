#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

uniform vec3 position;
uniform vec3 scale;
uniform vec4 rotation;
uniform mat4 projection;

out VS_OUT
{
	vec2 TexCoords;
	vec3 Normal;
	vec3 FragPos;
} vs_out;

void main()
{
	vec4 conj = vec4(-rotation.x, -rotation.y, -rotation.z, rotation.w);

	//Calculate normal rotation
	vec4 normalQuaternion = vec4(aNormal.xyz, 0.0);
	vec4 inverseRotationNormal = vec4((conj.w * normalQuaternion.xyz + conj.xyz * normalQuaternion.w + cross(conj.xyz, normalQuaternion.xyz)).xyz,
		conj.w * normalQuaternion.w - dot(conj.xyz, normalQuaternion.xyz));
	vec4 rotatedNormal = vec4((inverseRotationNormal.w * rotation.xyz + inverseRotationNormal.xyz * rotation.w + cross(inverseRotationNormal.xyz, rotation.xyz)).xyz,
		inverseRotationNormal.w * rotation.w - dot(inverseRotationNormal.xyz, rotation.xyz));
	
	vs_out.Normal = rotatedNormal.xyz;

	vs_out.TexCoords = aTexCoords;
	
	//Calculate view position
	vec3 scaledVector = vec3(aPos.x * scale.x, aPos.y * scale.y, aPos.z * scale.z);
	vec4 scaledQuaternion = vec4(scaledVector, 0.0);
	vec4 inverseRotation = vec4((conj.w * scaledQuaternion.xyz + conj.xyz * scaledQuaternion.w + cross(conj.xyz, scaledQuaternion.xyz)).xyz,
	 conj.w * scaledQuaternion.w - dot(conj.xyz, scaledQuaternion.xyz));
	vec4 rotatedVector = vec4((inverseRotation.w * rotation.xyz + inverseRotation.xyz * rotation.w + cross(inverseRotation.xyz, rotation.xyz)).xyz,
	 inverseRotation.w * rotation.w - dot(inverseRotation.xyz, rotation.xyz));

	vs_out.FragPos = rotatedVector.xyz + position;

	vec3 viewPos = rotatedVector.xyz + position; // 1 scaling 2 rotation 3 sum translation

	gl_Position = projection * vec4(viewPos, 1.0);
}