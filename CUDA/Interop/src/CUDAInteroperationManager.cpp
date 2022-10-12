#include "../CUDAInteroperationManager.hpp"

#include <Window.hpp>

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
        else if (filterType == ScreenSpaceFilters::EDGE)
            currentFilter = this->filters.edge;
        else if (filterType == ScreenSpaceFilters::EMBOSS)
            currentFilter = this->filters.emboss;
        else if (filterType == ScreenSpaceFilters::INVERT)
            currentFilter = this->filters.invert;
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
        cudaGraphicsGLRegisterImage( &cudaResource, GLtexture, GL_TEXTURE_2D, mapFlags );
    }

    void CUDAInteroperationManager::deleteCUDATextureResource(cudaGraphicsResource_t& cudaResource)
    {
        if ( cudaResource != 0 )
        {
            cudaGraphicsUnregisterResource( cudaResource );
            cudaResource = 0;
        }
    }

} // namespace CUDA::Interop
