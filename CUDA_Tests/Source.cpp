#include "GpuMemoryManager.cuh"

/*
TODO:
- Test addition
- Export history of parameter details, maybe the layer tensor values as well in case it can give some insight
- test out invsqrt
- maybe pass some parameters to the layers instead of referencing them, might be cleaner
*/

/*
LESSONS:
- adam no normilization just seems hands down a lot better than sgd no normilization
- Lower learning rate means slower convergence, but more accurate results
-- Ill describe it like higher learning potential, but takes longer to learn
- Test normilization per layer (weight specifically)
-- in the simple case of outputing the input, no normilization is faster
-- with adam, unnormilized seems to be faster
*/

struct Layer {
	cublasHandle_t* cublasHandle;
	GpuMemoryManager* gpuMemoryManager;
	GpuRand* gpuRand;
	size_t* inputHeight;
	float* learningRate;
	float* meanBeta;
	float* varBeta;
	float* epsilon;
	float* meanCor;
	float* varCor;

	size_t inputWidth;
	float* deviceForwardInputTensor;
	float* deviceBackwardOutputTensor;

	void Initialize(cublasHandle_t* cublasHandle, GpuMemoryManager* gpuMemoryManager, GpuRand* gpuRand, size_t* inputHeight, float* learningRate, float* meanBeta, float* varBeta, float* epsilon, float* meanCor, float* varCor) {
		this->cublasHandle = cublasHandle;
		this->gpuMemoryManager = gpuMemoryManager;
		this->gpuRand = gpuRand;
		this->inputHeight = inputHeight;
		this->learningRate = learningRate;
		this->meanBeta = meanBeta;
		this->varBeta = varBeta;
		this->epsilon = epsilon;
		this->meanCor = meanCor;
		this->varCor = varCor;
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
	virtual void PrintUpdateParameters() = 0;
	virtual void PrintParameters() = 0;
};

struct WeightLayer : Layer {
	size_t outputWidth;
	float* deviceForwardWeightTensor;
	float* deviceForwardOutputTensor;
	float* deviceBackwardWeightTensor;
	float* deviceBackwardInputTensor;
	float* deviceWeightBackwardMean;
	float* deviceWeightBackwardVar;

	WeightLayer(size_t outputWidth) : outputWidth(outputWidth) {}

	size_t GetOutputDim() { return outputWidth; }

	void DescribeTensorDetails() {
		gpuMemoryManager->ManageStatic(&deviceForwardWeightTensor, inputWidth * outputWidth);
		gpuMemoryManager->ManageDynamic(&deviceForwardOutputTensor, outputWidth);
		gpuMemoryManager->ManageStatic(&deviceBackwardWeightTensor, inputWidth * outputWidth);
		gpuMemoryManager->ManageDynamic(&deviceBackwardInputTensor, inputWidth);
		gpuMemoryManager->ManageStatic(&deviceWeightBackwardMean, inputWidth * outputWidth);
		gpuMemoryManager->ManageStatic(&deviceWeightBackwardVar, inputWidth * outputWidth);
	}

	float* GetForwardOutputTensor() { return deviceForwardOutputTensor; }
	float* GetBackwardInputTensor() { return deviceBackwardInputTensor; }

	void InitializeParameters() {
		gpuRand->Rand(deviceForwardWeightTensor, inputWidth * outputWidth);
		cudaMemset(deviceWeightBackwardMean, 0, inputWidth * outputWidth * sizeof(float));
		cudaMemset(deviceWeightBackwardVar, 0, inputWidth * outputWidth * sizeof(float));
	}

	void Forward() {
		//float alpha = 1.0f / sqrtf(inputWidth);
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
		//float alpha = 1.0f / sqrtf(*inputHeight);
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

		//alpha = 1.0f / sqrtf(outputWidth);
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
		AdamUpdate(
			deviceWeightBackwardMean, deviceWeightBackwardVar,
			deviceBackwardWeightTensor, deviceForwardWeightTensor,
			*meanBeta, *varBeta, *epsilon, *meanCor, *varCor,
			*learningRate, inputWidth * outputWidth
		);
		/*FailIf(
			cublasSaxpy(
				*cublasHandle, inputWidth * outputWidth,
				learningRate,
				deviceBackwardWeightTensor, 1,
				deviceForwardWeightTensor, 1
			) != CUBLAS_STATUS_SUCCESS, "cublasSaxpy failed"
		);*/
	}

	void PrintForward() {
		printf("Weight Forward Print\n");

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceForwardInputTensor, "Input");
		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceForwardWeightTensor, "Weight");

		//float alpha = 1.0f / sqrtf(inputWidth);
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

		PrintDeviceTensorf32(false, *inputHeight, outputWidth, deviceForwardOutputTensor, "Output");
		printf("--------------------\n\n");
	}

	void PrintBackward() {
		printf("Weight Backward Print\n");

		PrintDeviceTensorf32(true, *inputHeight, inputWidth, deviceForwardInputTensor, "Input Transposed");
		PrintDeviceTensorf32(false, *inputHeight, outputWidth, deviceBackwardOutputTensor, "Output Gradient");

		//float alpha = 1.0f / sqrtf(*inputHeight);
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

		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceBackwardWeightTensor, "Weight Gradient");
		printf("\n");

		PrintDeviceTensorf32(false, *inputHeight, outputWidth, deviceBackwardOutputTensor, "Output Gradient");
		PrintDeviceTensorf32(true, inputWidth, outputWidth, deviceForwardWeightTensor, "Weight Transposed");

		//alpha = 1.0f / sqrtf(outputWidth);
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

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceBackwardInputTensor, "Input Gradient");
		printf("--------------------\n\n");
	}

	void PrintUpdateParameters() {
		printf("Weight Update Parameters Print\n");

		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceBackwardWeightTensor, "Weight Gradient");

		AdamUpdate(
			deviceWeightBackwardMean, deviceWeightBackwardVar,
			deviceBackwardWeightTensor, deviceForwardWeightTensor,
			*meanBeta, *varBeta, *epsilon, *meanCor, *varCor,
			*learningRate, inputWidth * outputWidth
		);

		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceWeightBackwardMean, "Weight Gradient Mean");
		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceWeightBackwardVar, "Weight Gradient Variance");
		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceBackwardWeightTensor, "Weight Gradient without learningRate");
		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceForwardWeightTensor, "Weight");
		printf("--------------------\n\n");
	}

	void PrintParameters() {
		printf("Weight Parameter Print\n");

		PrintDeviceTensorf32(false, inputWidth, outputWidth, deviceForwardWeightTensor, "Weight");
		printf("--------------------\n\n");
	}
};

struct BiasLayer : Layer {
	float* deviceForwardBiasTensor;
	float* deviceBackwardBiasTensor;
	float* deviceBiasBackwardMean;
	float* deviceBiasBackwardVar;

	BiasLayer() {}

	size_t GetOutputDim() { return inputWidth; }

	void DescribeTensorDetails() {
		gpuMemoryManager->ManageStatic(&deviceForwardBiasTensor, inputWidth);
		gpuMemoryManager->ManageStatic(&deviceBackwardBiasTensor, inputWidth);
		gpuMemoryManager->ManageStatic(&deviceBiasBackwardMean, inputWidth);
		gpuMemoryManager->ManageStatic(&deviceBiasBackwardVar, inputWidth);
	}

	float* GetForwardOutputTensor() { return deviceForwardInputTensor; }
	float* GetBackwardInputTensor() { return deviceBackwardOutputTensor; }

	void InitializeParameters() {
		gpuRand->Rand(deviceForwardBiasTensor, inputWidth);
		cudaMemset(deviceBiasBackwardMean, 0, inputWidth * sizeof(float));
		cudaMemset(deviceBiasBackwardVar, 0, inputWidth * sizeof(float));
	}

	void Forward() {
		BatchAddForward(deviceForwardBiasTensor, deviceForwardInputTensor, *inputHeight, inputWidth);
	}

	void Backward() {
		BatchAddBackward(deviceBackwardBiasTensor, deviceBackwardOutputTensor, *inputHeight, inputWidth);
	}

	void UpdateParameters() {
		AdamUpdate(
			deviceBiasBackwardMean, deviceBiasBackwardVar,
			deviceBackwardBiasTensor, deviceForwardBiasTensor,
			*meanBeta, *varBeta, *epsilon, *meanCor, *varCor,
			*learningRate, inputWidth
		);
		/*FailIf(
			cublasSaxpy(
				*cublasHandle, inputWidth,
				learningRate,
				deviceBackwardBiasTensor, 1,
				deviceForwardBiasTensor, 1
			) != CUBLAS_STATUS_SUCCESS, "cublasSaxpy failed"
		);*/
	}

	void PrintForward() {
		printf("Bias Forward Print\n");

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceForwardInputTensor, "Input");
		PrintDeviceTensorf32(false, 1, inputWidth, deviceForwardBiasTensor, "Bias");

		BatchAddForward(deviceForwardBiasTensor, deviceForwardInputTensor, *inputHeight, inputWidth);

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceForwardInputTensor, "Output");
		printf("--------------------\n\n");
	}

	void PrintBackward() {
		printf("Bias Backward Print\n");

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceBackwardOutputTensor, "Output Gradient");

		BatchAddBackward(deviceBackwardBiasTensor, deviceBackwardOutputTensor, *inputHeight, inputWidth);

		PrintDeviceTensorf32(false, 1, inputWidth, deviceBackwardBiasTensor, "Bias Gradient");
		printf("--------------------\n\n");
	}

	void PrintUpdateParameters() {
		printf("Bias Parameter Print\n");

		PrintDeviceTensorf32(false, 1, inputWidth, deviceBackwardBiasTensor, "Bias Gradient");

		AdamUpdate(
			deviceBiasBackwardMean, deviceBiasBackwardVar,
			deviceBackwardBiasTensor, deviceForwardBiasTensor,
			*meanBeta, *varBeta, *epsilon, *meanCor, *varCor,
			*learningRate, inputWidth
		);

		PrintDeviceTensorf32(false, 1, inputWidth, deviceBiasBackwardMean, "Bias Gradient Mean");
		PrintDeviceTensorf32(false, 1, inputWidth, deviceBiasBackwardVar, "Bias Gradient variance");
		PrintDeviceTensorf32(false, 1, inputWidth, deviceBackwardBiasTensor, "Bias Gradient without learningRate");
		PrintDeviceTensorf32(false, 1, inputWidth, deviceForwardBiasTensor, "Bias");
		printf("--------------------\n\n");
	}

	void PrintParameters() {
		printf("Bias Parameter Print\n");

		PrintDeviceTensorf32(false, 1, inputWidth, deviceForwardBiasTensor, "Bias");\
		printf("--------------------\n\n");
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
		printf("Relu Forward Print\n");

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceForwardInputTensor, "Input");

		ReluForward(deviceForwardInputTensor, *inputHeight, inputWidth, inputWidth);

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceForwardInputTensor, "Output");
		printf("--------------------\n\n");
	}

	void PrintBackward() {
		printf("Relu Backward Print\n");

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceBackwardOutputTensor, "Output Gradient");

		ReluBackward(deviceForwardInputTensor, deviceBackwardOutputTensor, *inputHeight, inputWidth, inputWidth);

		PrintDeviceTensorf32(false, *inputHeight, inputWidth, deviceBackwardOutputTensor, "Input Gradient");
		printf("--------------------\n\n");
	}

	void PrintUpdateParameters() {
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
		layer->Initialize(&cublasHandle, &gpuMemoryManager, &gpuRand, &inputHeight, &learningRate, &meanBeta, &varBeta, &epsilon, &meanCor, &varCor);
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
		for (auto layer : layers) { layer->Forward(); }
	}

	void Backward() {
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
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
		meanCor *= meanBeta;
		varCor *= varBeta;
		for (auto layer : layers) { layer->UpdateParameters(); }
	}

	void PrintForward() {
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");

		printf("Forward Print\n");
		for (auto layer : layers) { layer->PrintForward(); }
		printf("||||||||||||||||||||\n\n");
	}

	void PrintBackward() {
		FailIf(inputHeight > maxInputHeight, "inputHeight > maxInputHeight");
		float alpha = -1.0f;
		FailIf(
			cublasSaxpy(
				cublasHandle, outputWidth * inputHeight,
				&alpha,
				layers.back()->GetForwardOutputTensor(), 1,
				deviceBackwardOutputTensor, 1
			) != CUBLAS_STATUS_SUCCESS, "cublasSaxpy failed"
		);

		printf("Backward Print\n");
		for (size_t i = layers.size(); i--;) { layers[i]->PrintBackward(); }
		printf("||||||||||||||||||||\n\n");
	}

	void PrintUpdateParameters() {
		meanCor *= meanBeta;
		varCor *= varBeta;

		printf("Update Parameters Print\n");
		for (auto layer : layers) { layer->PrintUpdateParameters(); }
		printf("||||||||||||||||||||\n\n");
	}

	void PrintParameters() {
		printf("Parameter Print\n");
		for (auto layer : layers) { layer->PrintParameters(); }
		printf("||||||||||||||||||||\n\n");
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
};

int main() {
	srand(time(NULL));
	bool debug = true;

	size_t inputWidth = 16;
	size_t hiddenWidth = 16;
	size_t outputWidth = 8;

	NeuralNetwork neuralNetwork(inputWidth, outputWidth);
	neuralNetwork.inputHeight = 16;
	neuralNetwork.learningRate = 0.0001f;

	neuralNetwork.AddLayer(new WeightLayer(hiddenWidth));
	neuralNetwork.AddLayer(new BiasLayer());
	neuralNetwork.AddLayer(new ReluLayer());
	neuralNetwork.AddLayer(new WeightLayer(outputWidth));
	neuralNetwork.Finalize();

	float* hostInputTensor = new float[inputWidth * neuralNetwork.maxInputHeight];
	float* hostOutputTensor = new float[outputWidth * neuralNetwork.maxInputHeight];

	for (size_t i = 0; i < 40000; i++) {
		for (size_t batch = 0; batch < neuralNetwork.inputHeight; batch++) {
			uint8_t a = rand();
			uint8_t b = rand();
			uint8_t c = a + b;

			for (size_t j = 0; j < 8; j++) {
				hostInputTensor[batch * inputWidth + j] = (a >> j) & 1;
				hostInputTensor[batch * inputWidth + j + 8] = (b >> j) & 1;
				hostOutputTensor[batch * outputWidth + j] = (c >> j) & 1;
			}
		}
		cudaMemcpy(neuralNetwork.deviceForwardInputTensor, hostInputTensor, inputWidth * neuralNetwork.inputHeight * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(neuralNetwork.deviceBackwardOutputTensor, hostOutputTensor, outputWidth * neuralNetwork.inputHeight * sizeof(float), cudaMemcpyHostToDevice);
		
		neuralNetwork.Forward();
		neuralNetwork.Backward();
		neuralNetwork.UpdateParameters();
		if (i % 100 == 0) neuralNetwork.PrintError();
	}
	printf("\n");


	if (debug){
		for (size_t batch = 0; batch < neuralNetwork.inputHeight; batch++) {
			uint8_t a = rand();
			uint8_t b = rand();
			uint8_t c = a + b;

			for (size_t j = 0; j < 8; j++) {
				hostInputTensor[batch * inputWidth + j] = (a >> j) & 1;
				hostInputTensor[batch * inputWidth + j + 8] = (b >> j) & 1;
				hostOutputTensor[batch * outputWidth + j] = (c >> j) & 1;
			}
		}
		cudaMemcpy(neuralNetwork.deviceForwardInputTensor, hostInputTensor, inputWidth * neuralNetwork.inputHeight * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(neuralNetwork.deviceBackwardOutputTensor, hostOutputTensor, outputWidth * neuralNetwork.inputHeight * sizeof(float), cudaMemcpyHostToDevice);

		neuralNetwork.PrintForward();
		neuralNetwork.PrintBackward();
		neuralNetwork.PrintUpdateParameters();
		neuralNetwork.PrintError();
	} else {
		neuralNetwork.PrintParameters();
	}

	delete[] hostInputTensor;
	delete[] hostOutputTensor;

	printf("Press any key to exit\n");

	return 0;
}