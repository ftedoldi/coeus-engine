#include "VectorOperations.cuh"

#include "../Includes/CUDA/cuda_runtime.h"
#include "../Includes/CUDA/device_launch_parameters.h"

#include <stdio.h>

__global__ void Vector3AddKernel(float* a, float* b, float* c, int sizeOfArrays)
{
    int i = threadIdx.x;

    if (i < sizeOfArrays)
        c[i] = a[i] + b[i];
}

__global__ void Matrix3ToMatrix4Kernel(float* m3, float* m4)
{
    int x = threadIdx.x;
    int y = threadIdx.y;

    int flatIndex = x + (4 * y);

    if ( x < 3 && y < 3 )
        m4[flatIndex] = m3[x + 3 * y];
    else
        m4[flatIndex] = 0;
}

int AddVectors3(float vector1[3], float vector2[3], float result[3])
{
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 3 * sizeof(float));
    cudaMalloc(&d_b, 3 * sizeof(float));
    cudaMalloc(&d_c, 3 * sizeof(float));

    cudaMemcpy(d_a, vector1, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, vector2, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, result, 3 * sizeof(float), cudaMemcpyHostToDevice);

    Vector3AddKernel<<<1, 3>>>(d_a, d_b, d_c, 3);

    cudaMemcpy(result, d_c, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 1;
}

int Matrix3ToMatrix4(float m3[9], float m4[16])
{
    float *d_m3, *d_m4;

    cudaMalloc(&d_m3, 3 * 3 * sizeof(float));
    cudaMalloc(&d_m4, 4 * 4 * sizeof(float));

    cudaMemcpy(d_m3, m3, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m4, m4, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid (1);
    dim3 block (4, 4);

    Matrix3ToMatrix4Kernel<<<grid, block>>>(d_m3, d_m4);

    cudaMemcpy(m4, d_m4, 4 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    cudaFree(d_m3);
    cudaFree(d_m4);

    m4[15] = 1;

    return 1;
}