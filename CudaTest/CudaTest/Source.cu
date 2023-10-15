#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublasLt.h>

inline void checkCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

inline void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS API failed with status %d\n", status);
        exit(-1);
    }
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

__global__ void yess()
{

}

struct GpuRand {
    uint32_t seed1, seed2;

    GpuRand() {
        seed1 = 0xE621B963;
        seed2 = 0x6053653F;

        printf("Seed1: %u\n", seed1);
        printf("Seed2: %u\n\n", seed2);
    }

    void Rand(float* arr, uint32_t size) {
        seed1 ^= seed2;
        seed1 *= 0xBAC57D37;
        seed2 ^= seed1;
        seed2 *= 0x24F66AC9;

        gpuRandFunc <<<ceil(0.0009765625f * size), 1024>>> (arr, size, seed1, seed2);
    }
};

int main()
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    float alpha = 1.0;
    float beta = 0.0;

    size_t m = 1024;
    size_t n = 1024;
    size_t k = 1024;
    size_t N = 32;

    size_t lda = 1024;
    size_t ldb = 1024;
    size_t ldc = 1024;

    float* Adev, * Bdev, * Cdev, * biasDev;

    size_t workspaceSize = 1024 * 1024 * 4;
    void* workspace = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    
    cublasLtHandle_t ltHandle;

    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDev), m * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));

    GpuRand rand;
    rand.Rand(Adev, m * k * N);
    rand.Rand(Bdev, n * k * N);
    rand.Rand(biasDev, m * N);



    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
    
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, ldc));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);

    checkCublasStatus(cublasLtMatmul(ltHandle,
        operationDesc,
        &alpha,
        Adev,
        Adesc,
        Bdev,
        Bdesc,
        &beta,
        biasDev,
        Cdesc,
        Cdev,
        Ddesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0));

    return 0;
}