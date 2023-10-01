#include "Header.cuh"

//template <typename InType, typename OutType = InType, typename ComputeType = OutType>

int main() {
    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    printf("JK");
    return 0;
}