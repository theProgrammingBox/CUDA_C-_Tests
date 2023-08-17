#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>

void PrintTensorf32(size_t width, size_t height, float* arr, const char* label = "Tensor", size_t majorStride = 0, size_t tensorSize = 0, size_t batchCount = 1)
{
	if (majorStride == 0)
		majorStride = width;
	printf("%s:\n", label);
	for (int b = batchCount; b--;)
	{
		for (size_t i = 0; i < height; i++)
		{
			for (size_t j = 0; j < width; j++)
				printf("%6.3f ", arr[i * majorStride + j]);
			printf("\n");
		}
		printf("\n");
		arr += tensorSize;
	}
}

__global__ void CurandNormalizef32(float* output, size_t size, float min, float range)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = float(*(size_t*)(output + index) * range + min);
}

void CurandGenerateUniformf32(curandGenerator_t generator, float* output, size_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (unsigned int*)output, size);
	CurandNormalizef32 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 2.3283064365387e-10f);
}