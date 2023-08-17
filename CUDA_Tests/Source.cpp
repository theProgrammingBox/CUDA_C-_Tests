#include "GpuMemoryManager.cuh"

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

	float* d_A, * d_B, * d_C;
	const size_t m = 1 << 3;
	const size_t n = 1 << 3;
	const size_t k = 1 << 3;

	const size_t ASize = m * n;
	const size_t BSize = n * k;
	const size_t CSize = m * k;

	d_A = manager.MemFrags[0]->address;
	d_B = manager.MemFrags[0]->address + ASize;
	d_C = manager.MemFrags[0]->address + ASize + BSize;

	CurandGenerateUniformf32(curandGenerator, d_A, ASize);
	CurandGenerateUniformf32(curandGenerator, d_B, BSize);

	float* h_A = (float*)malloc(ASize * sizeof(float));
	float* h_B = (float*)malloc(BSize * sizeof(float));
	float* h_C = (float*)malloc(CSize * sizeof(float));

	cudaMemcpy(h_A, d_A, ASize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, BSize * sizeof(float), cudaMemcpyDeviceToHost);

	PrintTensorf32(n, m, h_A);
	PrintTensorf32(k, n, h_B);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasStatus = cublasSgemm(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		k, m, n,
		&alpha,
		d_B, k,
		d_A, n,
		&beta,
		d_C, k);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		printf("cublasSgemm failed with error code %d\n", cublasStatus);
		return EXIT_FAILURE;
	}

	cudaMemcpy(h_C, d_C, CSize * sizeof(float), cudaMemcpyDeviceToHost);

	PrintTensorf32(k, m, h_C);

	cublasStatus = cublasSgemmStridedBatched(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		k, m, n,
		&alpha,
		d_B, k, BSize,
		d_A, n, ASize,
		&beta,
		d_C, k, CSize,
		1);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		printf("cublasSgemmStridedBatched failed with error code %d\n", cublasStatus);
		return EXIT_FAILURE;
	}

	cudaMemcpy(h_C, d_C, CSize * sizeof(float), cudaMemcpyDeviceToHost);

	PrintTensorf32(k, m, h_C);

	printf("Freeing and reallocating memory\n");
	cudaError_t err;
	for (GpuMemoryManager::MemFrag* frag : manager.MemFrags)
	{
		printf("Freeing %zu bytes at %p\n", frag->size, frag->address);
		cudaFree(frag->address);

		err = cudaMalloc((void**)&frag->address, frag->size);
		if (err != cudaSuccess)
		{
			printf("cudaMalloc failed with error code %d\n", err);
			return EXIT_FAILURE;
		}
		printf("Allocated %zu bytes at %p\n", frag->size, frag->address);
	}

	cudaMemGetInfo(&freeMem, &totalMem);
	printf("Free memory: %zu\n", freeMem);

	cublasDestroy(cublasHandle);
	curandDestroyGenerator(curandGenerator);

	return 0;
}
