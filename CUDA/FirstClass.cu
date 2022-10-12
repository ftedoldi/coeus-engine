#include "FirstClass.cuh"

#include "../Includes/CUDA/cuda_runtime.h"
#include "../Includes/CUDA/device_launch_parameters.h"

#include <stdio.h>

#define SIZE 1024

__global__ void VectorAdd(int* a, int* b, int* c, int sizeOfArrays)
{
    int i = threadIdx.x;

    if (i < sizeOfArrays)
        c[i] = a[i] + b[i];
}

__global__ void Vector3Add(float* a, float* b, float* c, int sizeOfArrays)
{
    int i = threadIdx.x;

    if (i < sizeOfArrays)
        c[i] = a[i] + b[i];
}

__host__ int AddVectors()
{
    int *a, *b, *c;
    a = (int*)malloc(SIZE * sizeof(int));
    b = (int*)malloc(SIZE * sizeof(int));
    c = (int*)malloc(SIZE * sizeof(int));

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, SIZE * sizeof(int));
    cudaMalloc(&d_b, SIZE * sizeof(int));
    cudaMalloc(&d_c, SIZE * sizeof(int));

    /// Initialization of array a & b & c
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }

    cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    VectorAdd<<<1, SIZE>>>(d_a, d_b, d_c, SIZE);

    cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < SIZE; i++)
        printf("value of c[%d] is %d \n", i, c[i]);

    free(a);
    free(b);
    free(c);

    return 0;
}

int AddVector3(float vector1[3], float vector2[3], float result[3])
{
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 3 * sizeof(float));
    cudaMalloc(&d_b, 3 * sizeof(float));
    cudaMalloc(&d_c, 3 * sizeof(float));

    cudaMemcpy(d_a, vector1, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, vector2, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, result, 3 * sizeof(int), cudaMemcpyHostToDevice);

    Vector3Add<<<1, 3>>>(d_a, d_b, d_c, 3);

    cudaMemcpy(result, d_c, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 1;
}
