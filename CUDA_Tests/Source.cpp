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

	void Initialize(cublasHandle_t* cublasHandle, GpuMemoryManager* gpuMemoryManager, GpuRand* gpuRand, size_t* inputHeight, float* learningRate) {
		this->cublasHandle = cublasHandle;
		this->gpuMemoryManager = gpuMemoryManager;
		this->gpuRand = gpuRand;
		this->inputHeight = inputHeight;
		this->learningRate = learningRate;
	}

	virtual size_t GetOutputDim() = 0;
	void AssignInputDim(size_t inputWidth) {
		this->inputWidth = inputWidth;
	}
	virtual void DescribeTensorDetails() = 0;
	virtual float* GetForwardOutputTensor() = 0;
	virtual float* GetBackwardInputTensor() = 0;
	void AssignForwardInputTensor(float* deviceForwardInputTensor) {
		this->deviceForwardInputTensor = deviceForwardInputTensor;
	}
	void AssignBackwardOutputTensor(float* deviceBackwardOutputTensor) {
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

	WeightLayer(size_t outputWidth) : outputWidth(outputWidth) {}

	size_t GetOutputDim() { return outputWidth; }

	void DescribeTensorDetails() {
		gpuMemoryManager->ManageStatic(&deviceForwardWeightTensor, inputWidth * outputWidth);
		gpuMemoryManager->ManageDynamic(&deviceForwardOutputTensor, outputWidth);
		gpuMemoryManager->ManageStatic(&deviceBackwardWeightTensor, inputWidth * outputWidth);
		gpuMemoryManager->ManageDynamic(&deviceBackwardInputTensor, inputWidth);
	}

	float* GetForwardOutputTensor() { return deviceForwardOutputTensor; }
	float* GetBackwardInputTensor() { return deviceBackwardInputTensor; }

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
		PrintDeviceTensorf32(inputWidth, outputWidth, deviceForwardWeightTensor, "weight - deviceForwardWeightTensor");
		PrintDeviceTensorf32(*inputHeight, outputWidth, deviceForwardOutputTensor, "weight - deviceForwardOutputTensor");
	}

	void PrintBackward() {
		PrintDeviceTensorf32(inputWidth, outputWidth, deviceBackwardWeightTensor, "weight - deviceBackwardWeightTensor");
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceBackwardInputTensor, "weight - deviceBackwardInputTensor");
	}
};

struct ReluLayer : Layer {
	ReluLayer() {}

	size_t GetOutputDim() { return inputWidth; }

	void DescribeTensorDetails() {
	}

	float* GetForwardOutputTensor() { return deviceForwardInputTensor; }
	float* GetBackwardInputTensor() { return deviceBackwardOutputTensor; }

	void InitializeParameters() {
	}

	void Forward() {
		ReluForward(deviceForwardInputTensor, *inputHeight, inputWidth, inputWidth);
	}

	void Backward() {
		ReluBackward(deviceForwardInputTensor, deviceBackwardOutputTensor, *inputHeight, inputWidth, inputWidth);
	}

	void PrintForward() {
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceForwardInputTensor, "relu - deviceForwardOutputTensor");
	}

	void PrintBackward() {
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceBackwardOutputTensor, "relu - deviceBackwardInputTensor");
	}
};

struct NeuralNetweork {
	cublasHandle_t cublasHandle;
	GpuMemoryManager gpuMemoryManager;
	GpuRand gpuRand;
	size_t inputHeight;
	float learningRate;

	std::vector<Layer*> layers;

	size_t inputWidth;
	size_t outputWidth;
	float* deviceForwardInputTensor;
	float* deviceBackwardOutputTensor;

	size_t maxInputHeight;

	NeuralNetweork(size_t inputWidth, size_t outputWidth) :
		inputWidth(inputWidth), outputWidth(outputWidth) {
		FailIf(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
		gpuMemoryManager.MapGpuMemory();
		inputHeight = 4;
		learningRate = 0.0001f;

		gpuMemoryManager.ManageDynamic(&deviceForwardInputTensor, inputWidth);
		gpuMemoryManager.ManageDynamic(&deviceBackwardOutputTensor, outputWidth);
	}

	void AddLayer(Layer* layer) {
		layer->Initialize(&cublasHandle, &gpuMemoryManager, &gpuRand, &inputHeight, &learningRate);
		if (layers.size() > 0) {
			layer->AssignInputDim(layers.back()->GetOutputDim());
			layer->DescribeTensorDetails();
		} else {
			layer->AssignInputDim(inputWidth);
			layer->DescribeTensorDetails();
		}
		layers.emplace_back(layer);
	}

	void Finalize() {
		gpuMemoryManager.Allocate(maxInputHeight);

		for (auto layer : layers) { layer->InitializeParameters(); }

		layers.front()->AssignForwardInputTensor(deviceForwardInputTensor);
		for (size_t i = 1; i < layers.size(); i++) { layers[i]->AssignForwardInputTensor(layers[i - 1]->GetForwardOutputTensor()); }

		layers.back()->AssignBackwardOutputTensor(deviceBackwardOutputTensor);
		for (size_t i = layers.size() - 1; i--;) { layers[i]->AssignBackwardOutputTensor(layers[i + 1]->GetBackwardInputTensor()); }
	}

	void Forward() {
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
		gpuRand.Rand(deviceForwardInputTensor, inputHeight * inputWidth);

		PrintDeviceTensorf32(inputHeight, inputWidth, deviceForwardInputTensor, "input - deviceForwardInputTensor");
		for (auto layer : layers) {
			layer->Forward();
			layer->PrintForward();
		}
		printf("\n");
	}

	void Backward() {
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
		//gpuRand.Rand(deviceBackwardOutputTensor, outputWidth * inputHeight);
		cudaMemcpy(deviceBackwardOutputTensor, deviceForwardInputTensor, inputHeight * inputWidth * sizeof(float), cudaMemcpyDeviceToDevice);
		gpuSub(layers.back()->GetForwardOutputTensor(), deviceBackwardOutputTensor, outputWidth * inputHeight);

		PrintDeviceTensorf32(inputHeight, outputWidth, deviceBackwardOutputTensor, "output - deviceBackwardOutputTensor");
		for (auto layer : layers) {
			layer->Backward();
			layer->PrintBackward();
		}
		printf("\n");
	}
};

int main() {
	size_t inputWidth = 3;
	size_t outputWidth = 3;

	NeuralNetweork neuralNetweork(inputWidth, outputWidth);
	neuralNetweork.AddLayer(new WeightLayer(outputWidth));
	neuralNetweork.AddLayer(new ReluLayer());
	neuralNetweork.Finalize();

	for (size_t i = 0; i < 1; i++) {
		neuralNetweork.Forward();
		neuralNetweork.Backward();
	}


	printf("Press any key to exit\n");


	return 0;
}