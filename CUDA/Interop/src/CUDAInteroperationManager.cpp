#include "../CUDAInteroperationManager.hpp"

#include <Window.hpp>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#include <Postprocess.cuh>

namespace CUDA::Interop
{
    
    CUDAInteroperationManager::CUDAInteroperationManager()
    {
        cudaGLSetGLDevice(0);
        cudaResources[0] = 0;
        cudaResources[1] = 0;
    }

    void CUDAInteroperationManager::applyFilterOverTexture(const int& width, const int& height, const ScreenSpaceFilters& filterType)
    {
        float* currentFilter;
        if (filterType == ScreenSpaceFilters::NONE)
            currentFilter = this->filters.unfilter;
        else if (filterType == ScreenSpaceFilters::BLUR)
            currentFilter = this->filters.blur;
        else if (filterType == ScreenSpaceFilters::LAPLACIAN)
            currentFilter = this->filters.laplacian;
        else if (filterType == ScreenSpaceFilters::EMBOSS)
            currentFilter = this->filters.emboss;
        else if (filterType == ScreenSpaceFilters::INVERT)
            currentFilter = this->filters.invert;
        else if (filterType == ScreenSpaceFilters::PREWITT_H)
            currentFilter = this->filters.prewittH;
        else if (filterType == ScreenSpaceFilters::PREWITT_V)
            currentFilter = this->filters.prewittV;
        else if (filterType == ScreenSpaceFilters::SOBEL_H)
            currentFilter = this->filters.sobelH;
        else if (filterType == ScreenSpaceFilters::SOBEL_V)
            currentFilter = this->filters.sobelV;
        else if (filterType == ScreenSpaceFilters::DITHER)
            currentFilter = this->filters.dither;
        else
            currentFilter = this->filters.sharpen;
        
        PostprocessCUDA( 
                        cudaResources[1], 
                        cudaResources[0], 
                        width, 
                        height, 
                        currentFilter, 
                        currentFilter[25], 
                        currentFilter[26] 
                    );
    }

    void CUDAInteroperationManager::createCUDATextureResource(cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags)
    {
        // Map the GL texture resource with the CUDA resource
        checkCudaErrors(cudaGraphicsGLRegisterImage( &cudaResource, GLtexture, GL_TEXTURE_2D, mapFlags ));
    }

    void CUDAInteroperationManager::deleteCUDATextureResource(cudaGraphicsResource_t& cudaResource)
    {
        if ( cudaResource != 0 )
        {
            checkCudaErrors(cudaGraphicsUnregisterResource( cudaResource ));
            cudaResource = 0;
        }
    }

    void CUDAInteroperationManager::createCUDABufferResource( cudaGraphicsResource_t& cudaResource, GLuint buffer, cudaGraphicsMapFlags mapFlags )
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaResource, buffer, mapFlags));
    }

    void CUDAInteroperationManager::deleteCUDABufferResource(cudaGraphicsResource_t& cudaResource)
    {
        if ( cudaResource != 0 )
        {
            checkCudaErrors(cudaGraphicsUnregisterResource( cudaResource ));
            cudaResource = 0;
        }
    }

} // namespace CUDA::Interop
