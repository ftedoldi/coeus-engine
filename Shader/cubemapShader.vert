#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform vec4 rotation;
uniform vec3 position;
uniform mat4 projection;

void main()
{
	TexCoords = aPos;
	vec4 conj = vec4(-rotation.x, -rotation.y, -rotation.z, rotation.w);
	vec4 posQuaternion = vec4(TexCoords, 0.0);

	vec4 firstRotation = vec4(rotation.w * posQuaternion.x + rotation.x * posQuaternion.w + rotation.y * posQuaternion.z - rotation.z * posQuaternion.y,
							  rotation.w * posQuaternion.y + rotation.y * posQuaternion.w + rotation.z * posQuaternion.x - rotation.x * posQuaternion.z,
							  rotation.w * posQuaternion.z + rotation.z * posQuaternion.w + rotation.x * posQuaternion.y - rotation.y * posQuaternion.x,
							  rotation.w * posQuaternion.w - rotation.x * posQuaternion.x - rotation.y * posQuaternion.y - rotation.z * posQuaternion.z);
	
	vec4 inverseRotation = vec4(firstRotation.w * conj.x + firstRotation.x * conj.w + firstRotation.y * conj.z - firstRotation.z * conj.y,
								firstRotation.w * conj.y + firstRotation.y * conj.w + firstRotation.z * conj.x - firstRotation.x * conj.z,
								firstRotation.w * conj.z + firstRotation.z * conj.w + firstRotation.x * conj.y - firstRotation.y * conj.x,
								firstRotation.w * conj.w - firstRotation.x * conj.x - firstRotation.y * conj.y - firstRotation.z * conj.z);

	vec3 viewPos = inverseRotation.xyz;

    vec4 pos = projection * vec4(viewPos, 1.0);
	//we do pos.xyww because we want to make sure that the depth value of the rendered cube
	//will always be 1.0, the maximum depth value, so everything gets rendered before the cubemap
    gl_Position = pos.xyww;
}  