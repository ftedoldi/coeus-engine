#include "VertexMove.cuh"
#include "../Includes/CUDA/cuda_runtime.h"
#include "../Includes/CUDA/device_launch_parameters.h"

#include <stdio.h>

__global__ void kernelMoveVertices(float3* dptr, size_t numVertices, double dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < numVertices)
        dptr[tid].x += dt;
    
}

void MoveVertices(cudaGraphicsResource_t& vbo, size_t numVertices, double dt)
{
    dim3 block(16, 1, 1);
    dim3 grid((numVertices + block.x - 1) / block.x, 1, 1);
    float3* dptr;
    size_t vs_dst;

    // Map the resources so they can be used in the kernel.
    cudaGraphicsMapResources(1, &vbo);
    
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &vs_dst, vbo);

    //Kernel call
    kernelMoveVertices<<<block, grid>>>(dptr, numVertices, dt);

    cudaGraphicsUnmapResources(1, &vbo);
}