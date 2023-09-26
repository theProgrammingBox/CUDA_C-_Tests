#pragma once
#include <stdio.h>	// printf
#include <stdlib.h>	// malloc
#include <time.h>	// time
#include <stdint.h>	// uint32_t
#include <math.h>	// ceil
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

void FailIf(bool condition, const char* message) {
	if (condition) {
		fprintf(stderr, "%s", message);
		exit(0);
	}
}

float InvSqrt(float number) {
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

void PrintHostTensorf32(size_t height, size_t width, float* arr, const char* label = "Tensor", size_t majorStride = 0, size_t tensorSize = 0, size_t batchCount = 1) {
	if (majorStride == 0)
		majorStride = width;
	printf("%s:\n", label);
	for (size_t b = batchCount; b--;) {
		for (size_t i = 0; i < height; i++) {
			for (size_t j = 0; j < width; j++)
				printf("%6.3f ", arr[i * majorStride + j]);
			printf("\n");
		}
		printf("\n");
		arr += tensorSize;
	}
}

void PrintDeviceTensorf32(
	bool transposed,
	size_t height, size_t width,
	float* arr, const char* label = "Tensor",
	size_t majorStride = 0, size_t tensorSize = 0,
	size_t batchCount = 1)
{
	float* hostArr = (float*)malloc(height * width * sizeof(float) * batchCount);
	cudaMemcpy(hostArr, arr, height * width * sizeof(float) * batchCount, cudaMemcpyDeviceToHost);

	if (majorStride == 0) {
		majorStride = width;
	}

	printf("%s:\n", label);

	for (size_t b = 0; b < batchCount; b++) {
		for (size_t i = 0; i < (transposed ? width : height); i++) {
			for (size_t j = 0; j < (transposed ? height : width); j++) {
				size_t row = transposed ? j : i;
				size_t col = transposed ? i : j;
				printf("%6.3f ", hostArr[b * tensorSize + row * majorStride + col]);
			}
			printf("\n");
		}
		printf("\n");
	}

	free(hostArr);
}

__global__ void GpuReluForward(float* arr, uint32_t height, uint32_t width, uint32_t majorStride) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width * height) {
		uint32_t wx = idx % width;
		uint32_t hx = idx / width;
		uint32_t index = hx * majorStride + wx;
		arr[index] = arr[index] > 0.0f ? arr[index] : 0.0f;
	}
}

void ReluForward(float* arr, uint32_t height, uint32_t width, uint32_t majorStride) {
	GpuReluForward << <ceil(0.0009765625f * width * height), 1024 >> > (arr, height, width, majorStride);
}

__global__ void GpuReluBackward(float* arr, float* output, uint32_t height, uint32_t width, uint32_t majorStride) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width * height) {
		uint32_t wx = idx % width;
		uint32_t hx = idx / width;
		uint32_t index = hx * majorStride + wx;
		output[index] = arr[index] > 0.0f ? output[index] : 0.0f;
	}
}

void ReluBackward(float* arr, float* output, uint32_t height, uint32_t width, uint32_t majorStride) {
	GpuReluBackward << <ceil(0.0009765625f * width * height), 1024 >> > (arr, output, height, width, majorStride);
}

__global__ void GPUBatchAddForward(float* arr, float* output, uint32_t height, uint32_t width) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width) {
		//float alpha = 1.0f / sqrtf(width);
		float alpha = 1.0f;
		for (uint32_t i = 0; i < height; i++)
			output[i * width + idx] += alpha * arr[idx];
	}
}

void BatchAddForward(float* arr, float* output, uint32_t height, uint32_t width) {
	GPUBatchAddForward << <ceil(0.0009765625f * width), 1024 >> > (arr, output, height, width);
}

__global__ void GPUBatchAddBackward(float* arr, float* output, uint32_t height, uint32_t width) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width) {
		//float alpha = 1.0f / sqrtf(height);
		float alpha = 1.0f;
		float sum = 0.0f;
		for (uint32_t i = 0; i < height; i++)
			sum += output[i * width + idx];
		arr[idx] = sum * alpha;
	}
}

void BatchAddBackward(float* arr, float* output, uint32_t height, uint32_t width) {
	GPUBatchAddBackward << <ceil(0.0009765625f * width), 1024 >> > (arr, output, height, width);
}

__global__ void GPUAdamUpdate(float* gradMean, float* gradVar, float* grad, float* param, float meanBeta, float varBeta, float epsilon, float meanCor, float varCor, float learningRate, size_t size) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float gradient = grad[idx];
		float mean = meanBeta * gradMean[idx] + (1.0f - meanBeta) * gradient;
		float var = varBeta * gradVar[idx] + (1.0f - varBeta) * gradient * gradient;
		float meanCorr = mean / (1.0f - meanCor);
		float varCorr = var / (1.0f - varCor);
		gradMean[idx] = mean;
		gradVar[idx] = var;
		param[idx] += learningRate * meanCorr / (sqrtf(varCorr) + epsilon);
	}
}

void AdamUpdate(float* gradMean, float* gradVar, float* grad, float* param, float meanBeta, float varBeta, float epsilon, float meanCor, float varCor, float learningRate, size_t size) {
	GPUAdamUpdate << <ceil(0.0009765625f * size), 1024 >> > (gradMean, gradVar, grad, param, meanBeta, varBeta, epsilon, meanCor, varCor, learningRate, size);
}

__global__ void gpuRandFunc(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		uint32_t Hash = idx;

		Hash ^= seed1;
		Hash *= 0xBAC57D37;
		Hash ^= seed2;
		Hash *= 0x24F66AC9;

		arr[idx] = int32_t(Hash) * 0.0000000004656612875245796f;
	}
}

struct GpuRand {
	uint32_t seed1, seed2;

	GpuRand() {
		seed1 = time(NULL) ^ 0xE621B963;
		seed2 = 0x6053653F ^ (time(NULL) >> 32);

		printf("Seed1: %u\n", seed1);
		printf("Seed2: %u\n\n", seed2);
	}

	void Rand(float* arr, uint32_t size) {
		seed1 ^= seed2;
		seed1 *= 0xBAC57D37;
		seed2 ^= seed1;
		seed2 *= 0x24F66AC9;

		gpuRandFunc << <ceil(0.0009765625f * size), 1024 >> > (arr, size, seed1, seed2);
	}
};