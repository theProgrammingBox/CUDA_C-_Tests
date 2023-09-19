#include "GpuMemoryManager.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

struct Layer {
	cublasHandle_t* cublasHandle;
	GpuMemoryManager* gpuMemoryManager;
	GpuRand* gpuRand;
	size_t* inputHeight;
	float* learningRate;

	size_t inputWidth;
	float* deviceInputTensor;

	Layer(cublasHandle_t* cublasHandle, GpuMemoryManager* gpuMemoryManager, GpuRand* gpuRand, size_t* inputHeight, float* learningRate) :
		cublasHandle(cublasHandle), gpuMemoryManager(gpuMemoryManager), gpuRand(gpuRand), inputHeight(inputHeight), learningRate(learningRate) {}

	void DescribeInputDetails(size_t inputWidth, float* deviceInputTensor) {
		this->inputWidth = inputWidth;
		this->deviceInputTensor = deviceInputTensor;
	}
	virtual void DescribeTensorDetails() = 0;
	virtual void InitializeParameters() = 0;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void PrintParameters() = 0;
};

struct WeightLayer : Layer {
	size_t outputWidth;
	float* deviceWeightTensor;
	float* deviceOutputTensor;

	WeightLayer(cublasHandle_t* cublasHandle, GpuMemoryManager* gpuMemoryManager, GpuRand* gpuRand, size_t* inputHeight, float* learningRate, size_t outputWidth) :
		Layer(cublasHandle, gpuMemoryManager, gpuRand, inputHeight, learningRate), outputWidth(outputWidth) {}

	void DescribeTensorDetails() {
		gpuMemoryManager->ManageStatic(&deviceWeightTensor, inputWidth * outputWidth);
		gpuMemoryManager->ManageDynamic(&deviceOutputTensor, outputWidth);
	}

	void InitializeParameters() {
		gpuRand->Rand(deviceWeightTensor, inputWidth * outputWidth);
		//gpuRand->Rand(deviceOutputTensor, *inputHeight * outputWidth);
	}

	void Forward() {
		float alpha = 1.0f;
		float beta = 0.0f;
		printf("inputWidth: %u, outputWidth: %u, inputHeight: %u\n", inputWidth, outputWidth, *inputHeight);
		FailIf(
			cublasSgemm(
				*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				outputWidth, *inputHeight, inputWidth,
				&alpha,
				deviceWeightTensor, outputWidth,
				deviceInputTensor, inputWidth,
				&beta,
				deviceOutputTensor, outputWidth
			) != CUBLAS_STATUS_SUCCESS, "cublasSgemm failed"
		);
	}

	void Backward() {
	}

	void PrintParameters() {
		PrintDeviceTensorf32(inputWidth, outputWidth, deviceWeightTensor, "deviceWeightTensor");
		PrintDeviceTensorf32(*inputHeight, outputWidth, deviceOutputTensor, "deviceOutputTensor");
	}
};

int main() {
	/*cublasHandle_t cublasHandle;
	GpuMemoryManager gpuMemoryManager;
	GpuRand gpuRand;
	size_t inputHeight;
	float learningRate;

	FailIf(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
	gpuMemoryManager.MapGpuMemory();
	inputHeight = 16;
	learningRate = 0.0001f;


	size_t inputWidth = 3;
	size_t outputWidth = 2;

	float* deviceInputTensor;
	gpuMemoryManager.ManageDynamic(&deviceInputTensor, inputWidth);

	Layer* weightLayer = new WeightLayer(&cublasHandle, &gpuMemoryManager, &gpuRand, &inputHeight, &learningRate, outputWidth);
	weightLayer->DescribeInputDetails(inputWidth, deviceInputTensor);
	weightLayer->DescribeTensorDetails();


	size_t maxInputHeight;
	gpuMemoryManager.Allocate(maxInputHeight);
	weightLayer->InitializeParameters();


	FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
	gpuRand.Rand(deviceInputTensor, inputHeight * inputWidth);
	//weightLayer->Forward();
	//weightLayer->Backward();
	PrintDeviceTensorf32(inputHeight, inputWidth, deviceInputTensor, "deviceInputTensor");
	weightLayer->PrintParameters();
	printf("Press any key to exit\n");*/


	cublasHandle_t cublasHandle;
	GpuRand gpuRand;
	size_t inputHeight, inputWidth, outputWidth;
	float* deviceInputTensor;
	float* deviceWeightTensor;
	float* deviceOutputTensor;

	cublasCreate(&cublasHandle);
	inputHeight = 4;
	inputWidth = 3;
	outputWidth = 2;

	cudaMalloc(&deviceInputTensor, inputHeight * inputWidth * sizeof(float));
	cudaMalloc(&deviceWeightTensor, inputWidth * outputWidth * sizeof(float));
	cudaMalloc(&deviceOutputTensor, inputHeight * outputWidth * sizeof(float));

	gpuRand.Rand(deviceInputTensor, inputHeight * inputWidth);
	gpuRand.Rand(deviceWeightTensor, inputWidth * outputWidth);

	PrintDeviceTensorf32(inputHeight, inputWidth, deviceInputTensor, "deviceInputTensor");
	PrintDeviceTensorf32(inputWidth, outputWidth, deviceWeightTensor, "deviceWeightTensor");

	float alpha = 1.0f;
	float beta = 0.0f;

	cublasSgemm(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		outputWidth, inputHeight, inputWidth,
		&alpha,
		deviceWeightTensor, outputWidth,
		deviceInputTensor, inputWidth,
		&beta,
		deviceOutputTensor, outputWidth
	);

	PrintDeviceTensorf32(inputHeight, outputWidth, deviceOutputTensor, "deviceOutputTensor");


	return 0;
}