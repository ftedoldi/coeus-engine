#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform vec4 rotation;
uniform vec3 position;
uniform mat4 projection;

void main()
{
	//TODO make comments
	vec4 conj = vec4(-rotation.x, -rotation.y, -rotation.z, rotation.w);
	vec4 posQuaternion = vec4(aPos, 0.0);
	vec4 inverseRotation = vec4((conj.w * posQuaternion.xyz + conj.xyz * posQuaternion.w + cross(conj.xyz, posQuaternion.xyz)).xyz,
		conj.w * posQuaternion.w - dot(conj.xyz, posQuaternion.xyz));
	vec4 rotatedVector = vec4((inverseRotation.w * rotation.xyz + inverseRotation.xyz * rotation.w + cross(inverseRotation.xyz, rotation.xyz)).xyz,
		inverseRotation.w * rotation.w - dot(inverseRotation.xyz, rotation.xyz));

	vec3 viewPos = rotatedVector.xyz;
    
    TexCoords = aPos;
    vec4 pos = projection * vec4(viewPos, 1.0);
	//we do pos.xyww because we want to make sure that the depth value of the rendered cube
	//will always be 1.0, the maximum depth value, so everything gets rendered before the cubemap
    gl_Position = pos.xyww;
}  