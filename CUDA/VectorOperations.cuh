#ifndef __VECTOROPERATIONS_H__
#define __VECTOROPERATIONS_H__

#ifdef __cplusplus
extern "C" {
#endif

int __declspec(dllexport) AddVectors3(float vector1[3], float vector2[3], float result[3]);
int __declspec(dllexport) Matrix3ToMatrix4(float m3[9], float m4[16]);

#ifdef __cplusplus
}
#endif

#endif // __VECTOROPERATIONS_H__