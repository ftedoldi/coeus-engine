#ifndef __POSTPROCESS_H__
#define __POSTPROCESS_H__

#ifdef __cplusplus
extern "C" {
#endif

void __declspec(dllexport) PostprocessCUDA( cudaGraphicsResource_t& dst, cudaGraphicsResource_t& src, unsigned int width, unsigned int height, float* filter, float scale, float offset );

#ifdef __cplusplus
}
#endif

#endif // __POSTPROCESS_H__