#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>

void PrintMatrixF16(__half* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", __half2float(arr[i * cols + j]));
		printf("\n");
	}
	printf("\n");
}

__global__ void CurandNormalizeF16(__half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * range + min);
}

void CurandGenerateUniformF16(curandGenerator_t generator, __half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CurandNormalizeF16 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 0.0000152590218967f);
}

int main()
{
	printf("cuDNN version: %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
	printf("cuBLAS version: %d.%d.%d\n", CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH);
	printf("cuRAND version: %d.%d.%d\n", CURAND_VERSION / 1000, (CURAND_VERSION % 1000) / 100, CURAND_VERSION % 100);
	printf("\n");

	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	const __half alphaf16 = __float2half(1.0f);
	const __half betaf16 = __float2half(0.0f);
	
	const uint32_t N = 8;

	__half* gpuArr1;
	__half* gpuArr2;
	
	cudaMalloc(&gpuArr1, N * sizeof(__half));
	cudaMalloc(&gpuArr2, N * sizeof(__half));
	
	__half* cpuArr1 = new __half[N];
	__half* cpuArr2 = new __half[N];

	CurandGenerateUniformF16(curandGenerator, gpuArr1, N);
	CurandGenerateUniformF16(curandGenerator, gpuArr2, N);
	//cudaMemset(gpuArr2, 0, N * sizeof(__half));
	
	cudaMemcpy(cpuArr1, gpuArr1, N * sizeof(__half), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuArr2, gpuArr2, N * sizeof(__half), cudaMemcpyDeviceToHost);
	
	PrintMatrixF16(cpuArr1, 1, N, "Random");
	PrintMatrixF16(cpuArr2, 1, N, "Zero");

	/*
	use:
	cublasStatus_t cublasAxpyEx(cublasHandle_t handle,
		int n,
		const void* alpha,
		cudaDataType alphaType,
		const void* x,
		cudaDataType xType,
		int incx,
		void* y,
		cudaDataType yType,
		int incy,
		cudaDataType executiontype);
	*/

	cublasAxpyEx(cublasHandle, N, &alpha, CUDA_R_32F, gpuArr1, CUDA_R_16F, 1, gpuArr2, CUDA_R_16F, 1, CUDA_R_32F);
	cudaMemcpy(cpuArr2, gpuArr2, N * sizeof(__half), cudaMemcpyDeviceToHost);
	PrintMatrixF16(cpuArr2, 1, N, "Axpy");

	return 0;
}