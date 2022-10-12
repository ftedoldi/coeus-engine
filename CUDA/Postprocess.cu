#include <cuda_runtime_api.h>

#include <stdio.h>

#include "Postprocess.cuh"

#define USE_SHARED_MEM 0

#define FILTER_SIZE (5*5) // 5x5 kernel filter
#define BLOCK_SIZE 16     // block size

__constant__ float kernelFilter_device[FILTER_SIZE];
__constant__ int indexOffsetsU_device[25];
__constant__ int indexOffsetsV_device[25];
__constant__ float invScale_device;
__constant__ float offset_device;

texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;

template< typename R, typename T >
__device__ R Clamp( T value, T min, T max )
{
    if ( value < min )
    {
        return (R)min;
    }
    else if ( value > max )
    {
        return (R)max;
    }
    else
    {
        return (R)value;
    }
}

__global__ void PostprocessKernel( float4* dst, unsigned int imgWidth, unsigned int imgHeight )
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bw = blockDim.x;
    unsigned int bh = blockDim.y;
    // Non-normalized U, V coordinates of input texture for current thread.
    unsigned int u = ( bw * blockIdx.x ) + tx;
    unsigned int v = ( bh * blockIdx.y ) + ty;

    // Early-out if we are beyond the texture coordinates for our texture.
    if ( u > imgWidth || v > imgHeight ) return;

#if USE_SHARED_MEM
    __shared__ uchar4 sTex[BLOCK_SIZE+4][BLOCK_SIZE+4]; // 20 * 20 * 4 Bytes = 1,600 Bytes ~= 1.5 KB
    // U, V, coordinates relative to the shared memory block
    unsigned int sU = tx + 2;
    unsigned int sV = ty + 2;

    // Load the current (center) pixel into shared memory
    sTex[sU][sV] = tex2D( texRef, u, v );
    if ( tx < 2 )
    {
        // Left-edge
        sTex[tx][sV] = tex2D( texRef, u - 2, v );
        // Right-edge
        sTex[BLOCK_SIZE+sU][sV] = tex2D( texRef, u + BLOCK_SIZE, v );
    }
    if ( ty < 2 )
    {
        // Top-edge
        sTex[sU][ty] = tex2D( texRef, u, v - 2 );
        // Bottom-edge
        sTex[sU][BLOCK_SIZE+sV] = tex2D( texRef, u, v + BLOCK_SIZE );
    }
    if ( tx < 2 && ty < 2 ) // Corners
    {
        // Top-left 
        sTex[tx][ty] = tex2D(texRef, u - 2, v - 2 );
        // Top-right
        sTex[BLOCK_SIZE + sU][ty] = tex2D( texRef, u + BLOCK_SIZE, v - 2 );

        // Bottom-left
        sTex[tx][BLOCK_SIZE + sV] = tex2D( texRef, u - 2, v + BLOCK_SIZE );
        // Bottom-right
        sTex[BLOCK_SIZE + sU][BLOCK_SIZE + sV] = tex2D( texRef, u + BLOCK_SIZE, v + BLOCK_SIZE );
    }
    __syncthreads();
#endif

    unsigned int index = ( v * imgWidth ) + u;
    
    float4 tempColor = make_float4(0, 0, 0, 1);
    for ( int i = 0; i < FILTER_SIZE; ++i )
    {
#if USE_SHARED_MEM
        uchar4 color = sTex[sU + indexOffsetsU_device[i]][sV + indexOffsetsV_device[i]]; 
#else
        float4 color = tex2D( texRef, u + indexOffsetsU_device[i], v + indexOffsetsV_device[i] );
#endif
        tempColor.x += color.x * kernelFilter_device[i];
        tempColor.y += color.y * kernelFilter_device[i];
        tempColor.z += color.z * kernelFilter_device[i];
    }

    dst[index] = make_float4(Clamp<float>(tempColor.x * invScale_device + offset_device, 0.0f, 1.0f), Clamp<float>(tempColor.y * invScale_device + offset_device, 0.0f, 1.0f), Clamp<float>(tempColor.z * invScale_device + offset_device, 0.0f, 1.0f), 1 );
}

float4* g_dstBuffer = NULL;
size_t g_BufferSize = 0; 

void PostprocessCUDA( cudaGraphicsResource_t& dst, cudaGraphicsResource_t& src, unsigned int width, unsigned int height, float* filter, float scale, float offset )
{
    // Avoid divide by zero error:
    float invScale = ( scale == 0.0f ) ? 1.0f : 1.0f / scale;

    int indexOffsetsU[] = {
        -2, -1, 0, 1, 2,
        -2, -1, 0, 1, 2,
        -2, -1, 0, 1, 2,
        -2, -1, 0, 1, 2,
        -2, -1, 0, 1, 2,
    };
    int indexOffsetsV[] = {
        -2, -2, -2, -2, -2,
        -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,
         2,  2,  2,  2,  2,
    };
        
    // Copy the scale and offset to the device for use by the kernel.
    cudaMemcpyToSymbol(invScale_device, &invScale, sizeof(float), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol(offset_device, &offset, sizeof(float), 0, cudaMemcpyHostToDevice );
    
    // Copy the data in the filter to the constant device variable.
    cudaMemcpyToSymbol(kernelFilter_device, filter, FILTER_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice );

    // Copy the index offset arrays to constant memory 
    cudaMemcpyToSymbol(indexOffsetsU_device, indexOffsetsU, 25 * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(indexOffsetsV_device, indexOffsetsV, 25 * sizeof(int), 0, cudaMemcpyHostToDevice);

    cudaGraphicsResource_t resources[2] = { src, dst };

    // Map the resources so they can be used in the kernel.
    cudaGraphicsMapResources( 2, resources );

    cudaArray* srcArray;
    cudaArray* dstArray;

    // Get a device pointer to the OpenGL buffers
    cudaGraphicsSubResourceGetMappedArray( &srcArray, src, 0, 0 );
    cudaGraphicsSubResourceGetMappedArray( &dstArray, dst, 0, 0 );

    // Map the source texture to a texture reference.
    cudaBindTextureToArray( texRef, srcArray );

    // Destination buffer to store the result of the postprocess effect.
    size_t bufferSize = width * height * sizeof(float4);
    if ( g_BufferSize != bufferSize )
    {
        if ( g_dstBuffer != NULL )
        {
            cudaFree( g_dstBuffer );
        }
        // Only re-allocate the global memory buffer if the screen size changes, 
        // or it has never been allocated before (g_BufferSize is still 0)
        g_BufferSize = bufferSize;
        cudaMalloc( &g_dstBuffer, g_BufferSize );
    }

    // Compute the grid size
    size_t blocksW = (size_t)ceilf( width / (float)BLOCK_SIZE );
    size_t blocksH = (size_t)ceilf( height / (float)BLOCK_SIZE );
    dim3 gridDim( blocksW, blocksH, 1 );
    dim3 blockDim( BLOCK_SIZE, BLOCK_SIZE, 1 );

    PostprocessKernel<<< gridDim, blockDim >>>( g_dstBuffer, width, height );

    // Copy the destination back to the source array
    cudaMemcpyToArray( dstArray, 0, 0, g_dstBuffer, bufferSize, cudaMemcpyDeviceToDevice );

    // Unbind the texture reference
    cudaUnbindTexture( texRef );

    // Unmap the resources again so the texture can be rendered in OpenGL
    cudaGraphicsUnmapResources( 2, resources );

    cudaFree( srcArray );
    cudaFree( dstArray );
}