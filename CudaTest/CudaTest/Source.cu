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

int main()
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t aDesc = NULL, BDesc = NULL, biasDesc = NULL, outputDesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    cublasOperation_t inputTrans = CUBLAS_OP_T;
    cublasOperation_t weightTrans = CUBLAS_OP_N;

    float alpha = 1.0;
    float beta = 1.0;

    size_t inputWidth = 32;
    size_t inputHeight = 16;
    size_t outputWidth = 8;
    size_t batches = 1;

    size_t input1DStride = inputWidth;
    size_t weight1DStride = outputWidth;
    size_t output1DStride = outputWidth;

    size_t input2DStride = inputWidth * inputHeight;
    size_t weight2DStride = outputWidth * inputWidth;
    size_t output2DStride = outputWidth * inputHeight;

    float* aDev, * bDev, * outputDev, * biasDev;

    size_t workspaceSize = 1024 * 1024 * 4;
    void* workspace = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    
    cublasLtHandle_t ltHandle;

    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&bDev), inputWidth * inputHeight * batches * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&aDev), outputWidth * inputWidth * batches * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&outputDev), outputWidth * inputHeight * batches * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDev), outputWidth * inputHeight * batches * sizeof(float)));
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));

    GpuRand rand;
    rand.Rand(bDev, inputWidth * inputHeight * batches);
    rand.Rand(aDev, outputWidth * inputWidth * batches);
    rand.Rand(biasDev, outputWidth * inputHeight * batches);

    PrintDeviceTensorf32(false, inputHeight, inputWidth, bDev, "Input", input1DStride, input2DStride, batches);
    PrintDeviceTensorf32(false, inputWidth, outputWidth, aDev, "Weight", weight1DStride, weight2DStride, batches);
    PrintDeviceTensorf32(false, inputHeight, outputWidth, biasDev, "Bias", output1DStride, output2DStride, batches);

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &inputTrans, sizeof(inputTrans)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &weightTrans, sizeof(inputTrans)));
    
    checkCublasStatus(cublasLtMatrixLayoutCreate(&BDesc, CUDA_R_32F, inputTrans == CUBLAS_OP_N ? inputHeight : inputWidth, inputTrans == CUBLAS_OP_N ? inputWidth : inputHeight, input1DStride));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &input2DStride, sizeof(input2DStride)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_32F, weightTrans == CUBLAS_OP_N ? inputHeight : outputWidth, weightTrans == CUBLAS_OP_N ? outputWidth : inputHeight, weight1DStride));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &weight2DStride, sizeof(weight2DStride)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&biasDesc, CUDA_R_32F, inputHeight, outputWidth, output1DStride));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(biasDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(biasDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &output2DStride, sizeof(output2DStride)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&outputDesc, CUDA_R_32F, inputHeight, outputWidth, output1DStride));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(outputDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(outputDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &output2DStride, sizeof(output2DStride)));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, aDesc, BDesc, biasDesc, outputDesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);

    checkCublasStatus(cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        aDev,
        aDesc,
        bDev,
        BDesc,
        &beta,
        biasDev,
        biasDesc,
        outputDev,
        outputDesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0));

    PrintDeviceTensorf32(false, outputWidth, inputHeight, outputDev, "Output", inputHeight, outputWidth * inputHeight, batches);

    return 0;
}