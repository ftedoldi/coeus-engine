#ifndef __CUDAINTEROPERATIONMANAGER_H__
#define __CUDAINTEROPERATIONMANAGER_H__

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <vector>

// float g_Unfiltered[] = {
//     0, 0, 0, 0, 0,
//     0, 0, 0, 0, 0,
//     0, 0, 1, 0, 0,
//     0, 0, 0, 0, 0, 
//     0, 0, 0, 0, 0,
//     1, 0
// };

// float g_BlurFilter[] = { 
//     1, 1, 1, 1, 1,
//     1, 2, 2, 2, 1, 
//     1, 2, 3, 2, 1,
//     1, 2, 2, 2, 2,
//     1, 1, 1, 1, 1,
//     35, 0
// };

// float g_SharpeningFilter[] = {
//     0,  0,  0,  0,  0,
//     0,  0, -1,  0,  0, 
//     0, -1,  5, -1,  0,
//     0,  0, -1,  0,  0, 
//     0,  0,  0,  0,  0,
//     1, 0
// };

// float g_EmbossFilter[] = { 
//     0, 0, 0,  0, 0, 
//     0, 0, 0,  0, 0,
//     0, 0, 1,  0, 0,
//     0, 0, 0, -1, 0,
//     0, 0, 0,  0, 0,
//     1, 128
// };

// float g_InvertFilter[] = {
//     0, 0,  0, 0, 0,
//     0, 0,  0, 0, 0, 
//     0, 0, -1, 0, 0,
//     0, 0,  0, 0, 0,
//     0, 0,  0, 0, 0,
//     1, 255
// };

// float g_EdgeFilter[] = {
//     0,  0,  0,  0,  0,
//     0, -1, -1, -1,  0, 
//     0, -1,  8, -1,  0,
//     0, -1, -1, -1,  0, 
//     0,  0,  0,  0,  0,
//     1, 0
// };

namespace CUDA
{
    namespace Interop
    {
        // TODO: Implement more filter matrices
        // FIXME: GPU crash due to too many memory leaks

        enum ScreenSpaceFilters
        {
            NONE,
            BLUR,
            SHARPEN,
            EMBOSS,
            INVERT,
            LAPLACIAN,
            SOBEL_H,
            SOBEL_V,
            PREWITT_H,
            PREWITT_V,
            DITHER
        };

        struct FilterMatrix
        {
            float unfilter[27]  = {
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0,
                1, 0
            };

            float blur[27] = {
                1, 1, 1, 1, 1,
                1, 2, 2, 2, 1,
                1, 2, 3, 2, 1,
                1, 2, 2, 2, 2, 
                1, 1, 1, 1, 1,
                35, 0
            };

            float sharpen[27] = {
                0,  0,  0,  0,  0,
                0,  0, -1,  0,  0,
                0, -1,  5, -1,  0,
                0,  0, -1,  0,  0,
                0,  0,  0,  0,  0,
                1, 0
            };

            float emboss[27] = {
                0, 0, 0,  0, 0,
                0, 0, 0,  0, 0,
                0, 0, 1,  0, 0,
                0, 0, 0, -1, 0, 
                0, 0, 0,  0, 0,
                1, 0.5
            };

            float invert[27] = {
                0, 0,  0, 0, 0,
                0, 0,  0, 0, 0,
                0, 0, -1, 0, 0,
                0, 0,  0, 0, 0, 
                0, 0,  0, 0, 0,
                1, 1
            };

            float laplacian[27] = {
                0,  0,  0,  0,  0,
                0, -1, -1, -1,  0, 
                0, -1,  8, -1,  0,
                0, -1, -1, -1,  0,  
                0,  0,  0,  0,  0,
                1, 0
            };
            
            float prewittH[27] = {
                0,  0,  0,  0,  0,
                0, -1, 0, 1,  0, 
                0, -1, 0, 1,  0,
                0, -1, 0, 1,  0,  
                0,  0,  0,  0,  0,
                1, 0
            };
            
            float prewittV[27] = {
                0,  0,  0,  0,  0,
                0, -1, -1, -1,  0, 
                0, 0, 0, 0,  0,
                0, 1, 1, 1,  0,  
                0,  0,  0,  0,  0,
                1, 0
            };
            
            float sobelH[27] = {
                0,  0,  0,  0,  0,
                0, -1, 0, 1,  0, 
                0, -2, 0, 2, 0,
                0, -1, 0, 1, 0,  
                0,  0,  0,  0,  0,
                1, 0
            };
            
            float sobelV[27] = {
                0,  0,  0,  0,  0,
                0, -1, -2, -1,  0, 
                0,  0,  0,  0, 0,
                0,  1,  2,  1, 0,  
                0,  0,  0,  0,  0,
                1, 0
            };

            float dither[27] = {
                0.74, 0.27, 0.40, 0.60, 0,
                0.80, 0.00, 0.13, 0.94, 0,
                0.47, 0.54, 0.67, 0.34, 0,
                0.20, 1.00, 0.87, 0.07, 0,
                0, 0, 0, 0, 0,
                1, 0};
        };
        
        class CUDAInteroperationManager
        {
            private:
                FilterMatrix filters;

            public:
                cudaGraphicsResource_t cudaResources[2];

                CUDAInteroperationManager();

                void applyFilterOverTexture(const int& width, const int& height, const ScreenSpaceFilters& filterType);

                void createCUDATextureResource( cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags );
                void deleteCUDATextureResource( cudaGraphicsResource_t& cudaResource );

                void createCUDABufferResource( cudaGraphicsResource_t& cudaResource, GLuint buffer, cudaGraphicsMapFlags mapFlags );
                void deleteCUDABufferResource( cudaGraphicsResource_t& cudaResource );
        };

    } // namespace Interop
    
} // namespace CUDA


#endif // __CUDAINTEROPERATIONMANAGER_H__