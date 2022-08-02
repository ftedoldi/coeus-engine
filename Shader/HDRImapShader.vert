#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 localPos;

uniform vec4 rotation;
uniform mat4 projection;

void main()
{
	//TODO make comments
    localPos = aPos;
	vec4 conj = vec4(-rotation.x, -rotation.y, -rotation.z, rotation.w);
	vec4 posQuaternion = vec4(aPos, 0.0);
	vec4 inverseRotation = vec4((conj.w * posQuaternion.xyz + conj.xyz * posQuaternion.w + cross(conj.xyz, posQuaternion.xyz)).xyz,
		conj.w * posQuaternion.w - dot(conj.xyz, posQuaternion.xyz));
	vec4 rotatedVector = vec4((inverseRotation.w * rotation.xyz + inverseRotation.xyz * rotation.w + cross(inverseRotation.xyz, rotation.xyz)).xyz,
		inverseRotation.w * rotation.w - dot(inverseRotation.xyz, rotation.xyz));

	vec3 viewPos = rotatedVector.xyz;

    gl_Position = projection * vec4(viewPos, 1.0);
}  