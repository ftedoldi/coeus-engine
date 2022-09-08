#ifndef __FIRSTCLASS_H__
#define __FIRSTCLASS_H__

#ifdef __cplusplus
extern "C" {
#endif

int __declspec(dllexport) AddVectors();
int __declspec(dllexport) AddVector3(float vector1[3], float vector2[3], float result[3]);

#ifdef __cplusplus
}
#endif

#endif // __FIRSTCLASS_H__