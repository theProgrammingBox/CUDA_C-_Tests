#pragma once
#include "GpuMemoryManager.cuh"

struct CLU
{
	cublasHandle_t* cublasHandle;
	curandGenerator_t* curandGenerator;

	size_t* inHeight, inWidth, hiddenWidth, hiddenHeight, outWidth, heads;

	size_t nonlinearWidth, jointWidth, productWidth, outputSize, batches;
	float invSqrtInWidth, invsqrtHiddenWidth, invSqrtOutWidth, invSqrtProductWidth, invSqrtInHeight;
	float expDecayMean, expDecayVar;
	float beta1, beta2, epsilon;

	float* input, * weight, * product, * bias, * output;
	float* outputGrad, * productGrad, * biasGrad, * inputGrad, * weightGrad;
	float* weightGradMean, * weightGradVar, * biasGradMean, * biasGradVar;

	static constexpr float one = 1.0f;
	static constexpr float zero = 0.0f;

	CLU
	(
		cublasHandle_t* cublasHandle, curandGenerator_t* curandGenerator,
		size_t* inHeight, size_t inWidth, size_t hiddenWidth, size_t hiddenHeight, size_t outWidth, size_t heads,
		float* input, float* outputGrad,
		float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-16f
	) :
		// todo: input and outputGrad option for copy from cpu to gpu
		cublasHandle(cublasHandle), curandGenerator(curandGenerator),
		inHeight(inHeight), inWidth(inWidth),
		hiddenWidth(hiddenWidth), hiddenHeight(hiddenHeight),
		outWidth(outWidth), heads(heads),
		input(input), outputGrad(outputGrad),
		beta1(beta1), beta2(beta2), epsilon(epsilon)
	{
		nonlinearWidth = hiddenWidth * hiddenHeight;
		jointWidth = nonlinearWidth + outWidth * hiddenWidth;
		productWidth = jointWidth * heads;
		outputSize = outWidth * hiddenHeight;
		batches = heads * *inHeight;	// todo
		// batches is dynamic so define in forward pass, must remain the same for the next backward pass

		invSqrtInWidth = InvSqrt(inWidth);
		invsqrtHiddenWidth = InvSqrt(hiddenWidth);
		invSqrtOutWidth = InvSqrt(outWidth);
		invSqrtProductWidth = InvSqrt(productWidth);
		invSqrtInHeight = InvSqrt(*inHeight);	// todo
		// inheight is dynamic so define in forward pass, must remain the same for the next backward pass

		expDecayMean = 1.0f;
		expDecayVar = 1.0f;

		weight = new float[productWidth * inWidth];
		product = new float[productWidth * *inHeight];	// todo
		bias = new float[productWidth];
		output = new float[outputSize * batches];	// todo
		// product and output are dynamic so define in compile function where the max is calculated

		productGrad = new float[productWidth * *inHeight];	// todo
		biasGrad = new float[productWidth];
		inputGrad = new float[inWidth * *inHeight];	// todo
		weightGrad = new float[productWidth * inWidth];
		// productGrad and inputGrad are dynamic so define in compile function where the max is calculated

		weightGradMean = new float[productWidth * inWidth];
		weightGradVar = new float[productWidth * inWidth];
		biasGradMean = new float[productWidth];
		biasGradVar = new float[productWidth];

		for (size_t i = 0; i < productWidth * inWidth; ++i)
			weight[i] = RandomFloat();
		for (size_t i = 0; i < productWidth; ++i)
			bias[i] = RandomFloat();

		memset(weightGradMean, 0, productWidth * inWidth * sizeof(float));
		memset(weightGradVar, 0, productWidth * inWidth * sizeof(float));
		memset(biasGradMean, 0, productWidth * sizeof(float));
		memset(biasGradVar, 0, productWidth * sizeof(float));
	}

	~CLU()
	{
		delete[] weight;
		delete[] product;
		delete[] bias;
		delete[] output;

		delete[] productGrad;
		delete[] biasGrad;
		delete[] inputGrad;
		delete[] weightGrad;

		delete[] weightGradMean;
		delete[] weightGradVar;
		delete[] biasGradMean;
		delete[] biasGradVar;
	}

	void forward()
	{
		// dynamic placeholder

		//PrintTensorf32(inWidth, *inHeight, input, "input");
		//PrintTensorf32(productWidth, inWidth, weight, "weight");

		cpuSgemmStridedBatched
		(
			false, false,
			productWidth, *inHeight, inWidth,
			&invSqrtInWidth,
			weight, productWidth, 0,
			input, inWidth, 0,
			&zero,
			product, productWidth, 0,
			1
		);
		//PrintTensorf32(productWidth, *inHeight, product, "product");
		//PrintTensorf32(productWidth, 1, bias, "bias");

		for (size_t i = 0; i < *inHeight; ++i)
		{
			cpuSaxpy
			(
				productWidth,
				&one,
				bias, 1,
				product + i * productWidth, 1
			);
		}
		//PrintTensorf32(productWidth, *inHeight, product, "added bias");

		for (size_t i = 0; i < batches; ++i)
		{
			cpuBinaryForward
			(
				nonlinearWidth,
				&one,
				product + i * jointWidth,
				&zero,
				product + i * jointWidth
			);
		}
		//PrintTensorf32(productWidth, *inHeight, product, "full product tensor");
		//PrintTensorf32(hiddenWidth, hiddenHeight, product, "binary forward", 0, productWidth, *inHeight);
		//PrintTensorf32(outWidth, hiddenWidth, product + nonlinearWidth, "Linear forward", 0, productWidth, *inHeight);

		cpuSgemmStridedBatched
		(
			false, false,
			outWidth, hiddenHeight, hiddenWidth,
			&invsqrtHiddenWidth,
			product + nonlinearWidth, outWidth, jointWidth,
			product, hiddenWidth, jointWidth,
			&zero,
			output, outWidth, outputSize,
			batches
		);
		//PrintTensorf32(outputSize, *inHeight, output, "output");
	}

	void backward(float learningrate)
	{
		//PrintTensorf32(outWidth, hiddenHeight, outputGrad, "outputGrad", 0, outputSize, *inHeight);

		cpuSgemmStridedBatched
		(
			true, false,
			hiddenWidth, hiddenHeight, outWidth,
			&invSqrtOutWidth,
			product + nonlinearWidth, outWidth, jointWidth,
			outputGrad, outWidth, outputSize,
			&zero,
			productGrad, hiddenWidth, jointWidth,
			batches
		);
		//PrintTensorf32(hiddenWidth, hiddenHeight, productGrad, "binaryGrad", 0, productWidth, *inHeight);

		cpuSgemmStridedBatched
		(
			false, true,
			outWidth, hiddenWidth, hiddenHeight,
			&invsqrtHiddenWidth,
			outputGrad, outWidth, outputSize,
			product, hiddenWidth, jointWidth,
			&zero,
			productGrad + nonlinearWidth, outWidth, jointWidth,
			batches
		);
		//PrintTensorf32(outWidth, hiddenWidth, productGrad + nonlinearWidth, "linearGrad", 0, productWidth, *inHeight);
		//PrintTensorf32(productWidth, *inHeight, productGrad, "productGrad");

		// did not update this for heads as i am not planning to use it, its just here for completeness
		/*for (size_t i = 0; i < *inHeight; ++i)
		{
			cpuBinaryBackward
			(
				nonlinearWidth,
				&one,
				product + i * productWidth,
				productGrad + i * productWidth,
				product + i * productWidth,
				&zero,
				productGrad + i * productWidth
			);
		}*/
		//PrintTensorf32(productWidth, *inHeight, productGrad, "binaryGrad");

		memset(biasGrad, 0, productWidth * sizeof(float));
		for (size_t i = 0; i < *inHeight; ++i)
		{
			cpuSaxpy
			(
				productWidth,
				&invSqrtInHeight,
				productGrad + i * productWidth, 1,
				biasGrad, 1
			);
		}
		//PrintTensorf32(productWidth, 1, biasGrad, "biasGrad");

		cpuSgemmStridedBatched
		(
			true, false,
			inWidth, *inHeight, productWidth,
			&invSqrtProductWidth,
			weight, productWidth, 0,
			productGrad, productWidth, 0,
			&zero,
			inputGrad, inWidth, 0,
			1
		);
		//PrintTensorf32(inWidth, *inHeight, inputGrad, "inputGrad");

		cpuSgemmStridedBatched
		(
			false, true,
			productWidth, inWidth, *inHeight,
			&invSqrtInHeight,
			productGrad, productWidth, 0,
			input, inWidth, 0,
			&zero,
			weightGrad, productWidth, 0,
			1
		);
		//PrintTensorf32(productWidth, inWidth, weightGrad, "weightGrad");

		expDecayMean *= beta1;
		expDecayVar *= beta2;

		for (size_t i = 0; i < productWidth; ++i)
		{
			float gradient = biasGrad[i];
			float newGradMean = beta1 * biasGradMean[i] + (1.0f - beta1) * gradient;
			float newGradVar = beta2 * biasGradVar[i] + (1.0f - beta2) * gradient * gradient;
			biasGradMean[i] = newGradMean;
			biasGradVar[i] = newGradVar;
			float gradMeanCorrected = newGradMean / (1.0f - expDecayMean);
			float gradVarCorrected = newGradVar / (1.0f - expDecayVar);
			float finalGradient = gradMeanCorrected * InvSqrt(gradVarCorrected + epsilon);
			bias[i] += finalGradient * learningrate;
		}

		for (size_t i = 0; i < productWidth * inWidth; ++i)
		{
			float gradient = weightGrad[i];
			float newGradMean = beta1 * weightGradMean[i] + (1.0f - beta1) * gradient;
			float newGradVar = beta2 * weightGradVar[i] + (1.0f - beta2) * gradient * gradient;
			weightGradMean[i] = newGradMean;
			weightGradVar[i] = newGradVar;
			float gradMeanCorrected = newGradMean / (1.0f - expDecayMean);
			float gradVarCorrected = newGradVar / (1.0f - expDecayVar);
			float finalGradient = gradMeanCorrected * InvSqrt(gradVarCorrected + epsilon);
			weight[i] += finalGradient * learningrate;
		}
	}

	void printParameters() const
	{
		PrintTensorf32(productWidth, inWidth, weight, "weight");
		PrintTensorf32(productWidth, 1, bias, "bias");
	}
}