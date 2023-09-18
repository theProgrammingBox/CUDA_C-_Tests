#pragma once
#include <stdio.h>	// printf
#include <stdlib.h>	// malloc
#include <time.h>	// time
#include <stdint.h>	// uint32_t
#include <math.h>	// ceil
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

void FailIf(bool condition, const char* message)
{
	if (condition)
	{
		fprintf(stderr, "%s", message);
		exit(0);
	}
}

float InvSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

void PrintHostTensorf32(size_t height, size_t width, float* arr, const char* label = "Tensor", size_t majorStride = 0, size_t tensorSize = 0, size_t batchCount = 1)
{
	if (majorStride == 0)
		majorStride = width;
	printf("%s:\n", label);
	for (size_t b = batchCount; b--;)
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

void PrintDeviceTensorf32(size_t height, size_t width, float* arr, const char* label = "Tensor", size_t majorStride = 0, size_t tensorSize = 0, size_t batchCount = 1)
{
	float* hostArr = (float*)malloc(height * width * sizeof(float));
	cudaMemcpy(hostArr, arr, height * width * sizeof(float), cudaMemcpyDeviceToHost);
	if (majorStride == 0)
		majorStride = width;
	printf("%s:\n", label);
	for (size_t b = batchCount; b--;)
	{
		for (size_t i = 0; i < height; i++)
		{
			for (size_t j = 0; j < width; j++)
				printf("%6.3f ", hostArr[i * majorStride + j]);
			printf("\n");
		}
		printf("\n");
		hostArr += tensorSize;
	}
	free(hostArr);
}

__global__ void gpuRandFunc(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		uint32_t index = idx;

		index ^= seed1;
		index *= 0xBAC57D37;
		index ^= index >> 16;
		index ^= seed2;
		index *= 0x24F66AC9;
		index ^= index >> 16;

		arr[idx] = int32_t(index) * 0.000000000465661287524579f;
	}
}

__global__ void gpuAddFunc(float* arr, float* output, uint32_t width, uint32_t height)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width)
	{
		for (uint32_t i = 0; i < height; i++)
			output[i * width + idx] += arr[idx];
	}
}

void gpuAdd(float* arr, float* output, uint32_t width, uint32_t height)
{
	gpuAddFunc << <ceil(0.0009765625f * width), 1024 >> > (arr, output, width, height);
}

__global__ void gpuBinaryFunc(float* arr, uint32_t width, uint32_t height, uint32_t majorStride)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width* height)
	{
		uint32_t wx = idx % width;
		uint32_t hx = idx / width;
		uint32_t index = hx * majorStride + wx;
		arr[index] = arr[index] > 0.0f ? 1.0f : 0.0f;
	}
}

void gpuBinary(float* arr, uint32_t width, uint32_t height, uint32_t majorStride)
{
	gpuBinaryFunc << <ceil(0.0009765625f * width * height), 1024 >> > (arr, width, height, majorStride);
}

struct GpuRand
{
	uint32_t seed1, seed2;

	void Lehmer32(uint32_t& x)
	{
		x *= 0xBAC57D37;
		x ^= x >> 16;
		x *= 0x24F66AC9;
		x ^= x >> 16;
	}

	GpuRand()
	{
		uint32_t seed1 = time(NULL) ^ 0xE621B963;
		Lehmer32(seed1);
		Lehmer32(seed1);
		uint32_t seed2 = seed1 ^ 0x6053653F ^ (time(NULL) >> 32);
		Lehmer32(seed2);

		printf("Seed1: %u\n", seed1);
		printf("Seed2: %u\n\n", seed2);
	}

	void Rand(float* arr, uint32_t size)
	{
		Lehmer32(seed1);
		Lehmer32(seed2);
		gpuRandFunc << <ceil(0.0009765625f * size), 1024 >> > (arr, size, seed1, seed2);
	}
};