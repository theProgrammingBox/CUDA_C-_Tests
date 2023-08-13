#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>

struct GpuMemoryManager
{
	struct MemFrag
	{
		size_t size;
		void* address;
	};

	std::vector<MemFrag*> MemFrags;

	~GpuMemoryManager()
	{
		for (MemFrag* frag : MemFrags)
		{
			cudaFree(frag->address);
			delete frag;
		}
	}

	void MapGpuMem()
	{
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);

		MemFrag* frag;
		size_t low, high, guess;
		cudaError_t err;
		do
		{
			frag = new MemFrag;
			low = 1, high = freeMem;
			do
			{
				guess = (low + high) >> 1;
				err = cudaMalloc((void**)&frag->address, guess);
				err == cudaSuccess ? low = guess + 1 : high = guess - 1;
				cudaFree(frag->address);
			} while (low <= high);
			low--;

			if (low > 0)
			{
				frag->size = low;
				cudaMalloc((void**)&frag->address, low);
				freeMem -= low;
				MemFrags.emplace_back(frag);
			}
		} while (low > 0);
		delete frag;
	}

	void PrintGpuMem()
	{
		for (MemFrag* frag : MemFrags)
			printf("Allocated %zu bytes at %p\n", frag->size, frag->address);
	}
};

int main()
{
	cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
	cublasStatus = cublasCreate(&cublasHandle);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		printf("cublasCreate failed with error code %d\n", cublasStatus);
		return EXIT_FAILURE;
	}

	curandStatus_t curandStatus;
	curandGenerator_t curandGenerator;
	curandStatus = curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	if (curandStatus != CURAND_STATUS_SUCCESS) {
		printf("curandCreateGenerator failed with error code %d\n", curandStatus);
		return EXIT_FAILURE;
	}

	curandStatus = curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);
	if (curandStatus != CURAND_STATUS_SUCCESS) {
		printf("curandSetPseudoRandomGeneratorSeed failed with error code %d\n", curandStatus);
		return EXIT_FAILURE;
	}

	GpuMemoryManager manager;

	manager.MapGpuMem();
	manager.PrintGpuMem();

	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("Free memory: %zu\n", freeMem);

	cudaMemGetInfo(&freeMem, &totalMem);
	printf("Free memory: %zu\n", freeMem);

	cublasDestroy(cublasHandle);
	curandDestroyGenerator(curandGenerator);

	return 0;
}
