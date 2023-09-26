#include "GpuMemoryManager.cuh"

/*
TODO:
- Add Adam
-- memset to 0 with correction to "unbias" the first few iterations
- Test normilization per layer (weight specifically)
-- in the simple case of outputing the input, no normilization is faster
*/

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
	virtual void UpdateParameters() = 0;
	virtual void PrintForward() = 0;
	virtual void PrintBackward() = 0;
	virtual void PrintParameters() = 0;
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
		float alpha = 1.0f / sqrtf(inputWidth);
		//float alpha = 1.0f;
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
		float alpha = 1.0f / sqrtf(*inputHeight);
		//float alpha = 1.0f;
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

		alpha = 1.0f / sqrtf(outputWidth);
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

	void UpdateParameters() {
		FailIf(
			cublasSaxpy(
				*cublasHandle, inputWidth * outputWidth,
				learningRate,
				deviceBackwardWeightTensor, 1,
				deviceForwardWeightTensor, 1
			) != CUBLAS_STATUS_SUCCESS, "cublasSaxpy failed"
		);
	}

	void PrintForward() {
		PrintDeviceTensorf32(*inputHeight, outputWidth, deviceForwardOutputTensor, "weight - deviceForwardOutputTensor");
	}

	void PrintBackward() {
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceBackwardInputTensor, "weight - deviceBackwardInputTensor");
	}

	void PrintParameters() {
		PrintDeviceTensorf32(inputWidth, outputWidth, deviceForwardWeightTensor, "weight - deviceForwardWeightTensor");
	}
};

struct BiasLayer : Layer {
	float* deviceForwardBiasTensor;
	float* deviceBackwardBiasTensor;

	BiasLayer() {}

	size_t GetOutputDim() { return inputWidth; }

	void DescribeTensorDetails() {
		gpuMemoryManager->ManageStatic(&deviceForwardBiasTensor, inputWidth);
		gpuMemoryManager->ManageStatic(&deviceBackwardBiasTensor, inputWidth);
	}

	float* GetForwardOutputTensor() { return deviceForwardInputTensor; }
	float* GetBackwardInputTensor() { return deviceBackwardOutputTensor; }

	void InitializeParameters() {
		gpuRand->Rand(deviceForwardBiasTensor, inputWidth);
	}

	void Forward() {
		BatchAddForward(deviceForwardBiasTensor, deviceForwardInputTensor, *inputHeight, inputWidth);
	}

	void Backward() {
		BatchAddBackward(deviceBackwardBiasTensor, deviceBackwardOutputTensor, *inputHeight, inputWidth);
	}

	void UpdateParameters() {
		FailIf(
			cublasSaxpy(
				*cublasHandle, inputWidth,
				learningRate,
				deviceBackwardBiasTensor, 1,
				deviceForwardBiasTensor, 1
			) != CUBLAS_STATUS_SUCCESS, "cublasSaxpy failed"
		);
	}

	void PrintForward() {
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceForwardInputTensor, "bias - deviceForwardOutputTensor");
	}

	void PrintBackward() {
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceBackwardOutputTensor, "bias - deviceBackwardInputTensor");
	}

	void PrintParameters() {
		PrintDeviceTensorf32(1, inputWidth, deviceForwardBiasTensor, "bias - deviceForwardBiasTensor");
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

	void UpdateParameters() {
	}

	void PrintForward() {
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceForwardInputTensor, "relu - deviceForwardOutputTensor");
	}

	void PrintBackward() {
		PrintDeviceTensorf32(*inputHeight, inputWidth, deviceBackwardOutputTensor, "relu - deviceBackwardInputTensor");
	}

	void PrintParameters() {
	}
};

struct NeuralNetwork {
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

	float meanBeta;
	float varBeta;
	float epsilon;

	float meanCor;
	float varCor;

	NeuralNetwork(size_t inputWidth, size_t outputWidth) :
		inputWidth(inputWidth), outputWidth(outputWidth) {
		FailIf(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
		gpuMemoryManager.MapGpuMemory();
		inputHeight = 4;
		learningRate = 0.0001f;

		gpuMemoryManager.ManageDynamic(&deviceForwardInputTensor, inputWidth);
		gpuMemoryManager.ManageDynamic(&deviceBackwardOutputTensor, outputWidth);

		meanBeta = 0.9f;
		varBeta = 0.999f;
		epsilon = 1e-8f;

		meanCor = 1;
		varCor = 1;
	}

	~NeuralNetwork() {
		for (auto layer : layers) { delete layer; }
		cublasDestroy(cublasHandle);
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
		FailIf(layers.back()->GetOutputDim() != outputWidth, "layers.back()->GetOutputDim() != outputWidth");

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

		for (size_t i = 0; i < layers.size(); i++) { layers[i]->Forward(); }
	}

	void Backward() {
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
		cudaMemcpy(deviceBackwardOutputTensor, deviceForwardInputTensor, inputHeight * inputWidth * sizeof(float), cudaMemcpyDeviceToDevice);
		float alpha = -1.0f;
		FailIf(
			cublasSaxpy(
				cublasHandle, outputWidth * inputHeight,
				&alpha,
				layers.back()->GetForwardOutputTensor(), 1,
				deviceBackwardOutputTensor, 1
			) != CUBLAS_STATUS_SUCCESS, "cublasSaxpy failed"
		);

		for (size_t i = layers.size(); i--;) { layers[i]->Backward(); }
	}

	void UpdateParameters() {
		for (auto layer : layers) { layer->UpdateParameters(); }
	}

	void PrintError() {
		float error = 0.0f;
		FailIf(
			cublasSasum(
				cublasHandle, outputWidth * inputHeight,
				deviceBackwardOutputTensor, 1,
				&error
			) != CUBLAS_STATUS_SUCCESS, "cublasSasum failed"
		);
		error /= outputWidth * inputHeight;
		printf("error: %f\n", error);
	}

	void PrintParameters() {
		for (auto layer : layers) { layer->PrintParameters(); }
	}
};

int main() {
	size_t inputWidth = 3;
	size_t hiddenWidth = 6;
	size_t outputWidth = 3;

	NeuralNetwork neuralNetwork(inputWidth, outputWidth);
	neuralNetwork.inputHeight = 16;
	neuralNetwork.learningRate = 0.01f;

	neuralNetwork.AddLayer(new WeightLayer(hiddenWidth));
	neuralNetwork.AddLayer(new BiasLayer());
	neuralNetwork.AddLayer(new ReluLayer());
	neuralNetwork.AddLayer(new WeightLayer(outputWidth));
	neuralNetwork.Finalize();

	for (size_t i = 0; i < 1000; i++) {
		neuralNetwork.Forward();
		neuralNetwork.Backward();
		neuralNetwork.UpdateParameters();
		if (i % 100 == 0) neuralNetwork.PrintError();
	}
	printf("\n");

	neuralNetwork.PrintParameters();

	printf("Press any key to exit\n");


	return 0;
}