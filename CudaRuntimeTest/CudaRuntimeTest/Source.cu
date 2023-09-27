#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        exit(-1);
    }
}

int main(int argc, char **argv)
{
    cublasHandle_t cublasHandle;
    checkCublasStatus(cublasCreate(&cublasHandle));

    printf("cublas initialized\n");

	return 0;
}