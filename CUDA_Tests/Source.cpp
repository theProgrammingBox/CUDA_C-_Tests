#include "GpuMemoryManager.cuh"

struct Layer {
	cublasHandle_t* cublasHandle;
	GpuMemoryManager* gpuMemoryManager;
	GpuRand* gpuRand;
	size_t* inputHeight;
	float* learningRate;

	size_t inputWidth;
	float* deviceForwardInputTensor;
	float* deviceBackwardOutputTensor;

	Layer(cublasHandle_t* cublasHandle, GpuMemoryManager* gpuMemoryManager, GpuRand* gpuRand, size_t* inputHeight, float* learningRate) :
		cublasHandle(cublasHandle), gpuMemoryManager(gpuMemoryManager), gpuRand(gpuRand), inputHeight(inputHeight), learningRate(learningRate) {}

	void AssignInputDim(size_t inputWidth) {
		this->inputWidth = inputWidth;
	}
	virtual void DescribeTensorDetails() = 0;
	void ConnectTensors(float* deviceForwardInputTensor, float* deviceBackwardOutputTensor) {
		this->deviceForwardInputTensor = deviceForwardInputTensor;
		this->deviceBackwardOutputTensor = deviceBackwardOutputTensor;
	}
	virtual void InitializeParameters() = 0;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void PrintForward() = 0;
	virtual void PrintBackward() = 0;
};

struct WeightLayer : Layer {
	size_t outputWidth;
	float* deviceForwardWeightTensor;
	float* deviceForwardOutputTensor;
	float* deviceBackwardWeightTensor;
	float* deviceBackwardInputTensor;

	WeightLayer(cublasHandle_t* cublasHandle, GpuMemoryManager* gpuMemoryManager, GpuRand* gpuRand, size_t* inputHeight, float* learningRate, size_t outputWidth) :
		Layer(cublasHandle, gpuMemoryManager, gpuRand, inputHeight, learningRate), outputWidth(outputWidth) {}

	void DescribeTensorDetails() {
		gpuMemoryManager->ManageStatic(&deviceForwardWeightTensor, inputWidth * outputWidth);
		gpuMemoryManager->ManageDynamic(&deviceForwardOutputTensor, outputWidth);
		gpuMemoryManager->ManageStatic(&deviceBackwardWeightTensor, inputWidth * outputWidth);
		gpuMemoryManager->ManageDynamic(&deviceBackwardInputTensor, inputWidth);
	}

	void InitializeParameters() {
		gpuRand->Rand(deviceForwardWeightTensor, inputWidth * outputWidth);
	}

	void Forward() {
		float alpha = 1.0f;
		float beta = 0.0f;

		FailIf(
			cublasSgemm(
				*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				outputWidth, *inputHeight, inputWidth,
				&alpha,
				deviceForwardWeightTensor, outputWidth,
				deviceForwardInputTensor, inputWidth,
				&beta,
				deviceForwardOutputTensor, outputWidth
			) != CUBLAS_STATUS_SUCCESS, "cublasSgemm failed"
		);
	}

	void Backward() {
		float alpha = 1.0f;
		float beta = 0.0f;

		FailIf(
			cublasSgemm(
				*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
				outputWidth, inputWidth, *inputHeight,
				&alpha,
				deviceBackwardOutputTensor, outputWidth,
				deviceForwardInputTensor, inputWidth,
				&beta,
				deviceBackwardWeightTensor, outputWidth
			) != CUBLAS_STATUS_SUCCESS, "cublasSgemm failed"
		);

		FailIf(
			cublasSgemm(
				*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				inputWidth, *inputHeight, outputWidth,
				&alpha,
				deviceForwardWeightTensor, outputWidth,
				deviceBackwardOutputTensor, outputWidth,
				&beta,
				deviceBackwardInputTensor, inputWidth
			) != CUBLAS_STATUS_SUCCESS, "cublasSgemm failed"
		);
	}

	void PrintForward() {
		PrintDeviceTensorf32(inputWidth, outputWidth, deviceForwardWeightTensor, "deviceForwardWeightTensor");
		PrintDeviceTensorf32(*inputHeight, outputWidth, deviceForwardOutputTensor, "deviceForwardOutputTensor");
	}

	void PrintBackward() {
		PrintDeviceTensorf32(inputWidth, outputWidth, deviceBackwardWeightTensor, "deviceBackwardWeightTensor");
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceBackwardInputTensor, "deviceBackwardInputTensor");
	}
};

int main() {
	cublasHandle_t cublasHandle;
	GpuMemoryManager gpuMemoryManager;
	GpuRand gpuRand;
	size_t inputHeight;
	float learningRate;

	FailIf(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
	gpuMemoryManager.MapGpuMemory();
	inputHeight = 4;
	learningRate = 0.0001f;


	size_t inputWidth = 3;
	size_t outputWidth = 2;

	float* deviceForwardInputTensor;
	float* deviceBackwardOutputTensor;
	gpuMemoryManager.ManageDynamic(&deviceForwardInputTensor, inputWidth);
	gpuMemoryManager.ManageDynamic(&deviceBackwardOutputTensor, outputWidth);

	Layer* weightLayer = new WeightLayer(&cublasHandle, &gpuMemoryManager, &gpuRand, &inputHeight, &learningRate, outputWidth);
	weightLayer->AssignInputDim(inputWidth);
	weightLayer->DescribeTensorDetails();


	size_t maxInputHeight;
	gpuMemoryManager.Allocate(maxInputHeight);
	weightLayer->ConnectTensors(deviceForwardInputTensor, deviceBackwardOutputTensor);
	weightLayer->InitializeParameters();


	FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
	gpuRand.Rand(deviceForwardInputTensor, inputHeight * inputWidth);
	gpuRand.Rand(deviceBackwardOutputTensor, outputWidth * inputHeight);


	weightLayer->Forward();
	weightLayer->Backward();


	PrintDeviceTensorf32(inputHeight, inputWidth, deviceForwardInputTensor, "deviceForwardInputTensor");
	weightLayer->PrintForward();

	PrintDeviceTensorf32(inputHeight, outputWidth, deviceBackwardOutputTensor, "deviceBackwardOutputTensor");
	weightLayer->PrintBackward();


	printf("Press any key to exit\n");


	return 0;
}