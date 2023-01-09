//#include <cublas_v2.h>
//#include <curand.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::cout;
using std::vector;
using std::sort;
using std::ceil;
using std::exp;

static struct xorwow32
{
	uint32_t state[6];

	xorwow32(uint32_t seed) : state{
		seed ^ 123456789,
		seed ^ 362436069,
		seed ^ 521288629,
		seed ^ 88675123,
		seed ^ 5783321,
		seed ^ 6615241 } {}

	uint32_t operator()()
	{
		uint32_t t = state[0] ^ (state[0] >> 2);
		memcpy(state, state + 1, 16);
		state[4] ^= (state[4] << 4) ^ (t ^ (t << 1));
		return (state[5] += 362437) + state[4];
	}

	float operator()(float min, float max)
	{
		return min + (max - min) * operator()() * 2.3283064371e-10;	// 0 & 1 inclusive, 2.3283064365e-10 for exclusive 1
	}
} random(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

const static void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

const static void cpuGenerateUniform(float* matrix, uint32_t size, float min, float max)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = random(min, max);
}

const static void cpuClippedLinearUnit(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	float input;
	for (size_t counter = size; counter--;)
	{
		input = inputMatrix[counter] + 1;
		input = (input > 0) * input - 2;
		outputMatrix[counter] = (input < 0) * input + 1;
	}
}

const static void cpuClippedLinearUnitGradient(float* inputMatrix, float* gradientMatrix, float* outputMatrix, uint32_t size) {
	float input;
	float gradient;
	bool greaterOne;
	for (size_t counter = size; counter--;)
	{
		input = inputMatrix[counter];
		gradient = gradientMatrix[counter];
		bool greaterZero = (uint32_t)gradient & 0x80000000;
		outputMatrix[counter] = (greaterZero ^ (input > 1) || greaterZero ^ (input > -1)) * gradient;
	}
}

const static void cpuSoftmax(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	float max = inputMatrix[0];
	for (uint32_t counter = size; counter--;)
		if (inputMatrix[counter] > max)
			max = inputMatrix[counter];
	float sum = 0;
	for (uint32_t counter = size; counter--;)
	{
		outputMatrix[counter] = exp(inputMatrix[counter] - max);
		sum += outputMatrix[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		outputMatrix[counter] *= sum;
}

const static void cpuSoftmaxGradient(float* outputMatrix, float* gradient, uint32_t* sample, float* resultMatrix, uint32_t size)
{
	float sampleValue = outputMatrix[*sample];
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = sampleValue * *gradient * ((counter == *sample) - outputMatrix[counter]);
}

int main() {
	constexpr uint32_t N = 3;
	constexpr uint32_t ITERATIONS = 1000;
	constexpr float LEARNING_RATE = 0.1f;
	
	float goal[N] = {};	// target probabilities
	float bias[N] = {};	// bias that is converted to probabilities
	float res[N] = {};	// resulting probabilities after softmax bias
	float grad[N] = {};	// gradient of the loss function
	uint32_t sample = 0;// sampled index from the probability distribution
	float gradient;		// whether the sampled index should be increased or decreased

	cpuGenerateUniform(bias, N, -1, 1);
	cpuSoftmax(bias, goal, N);	// generate random target probabilities

	cpuGenerateUniform(bias, N, -1, 1);	// generate random bias

	int iter = ITERATIONS;
	while (iter--)
	{
		// calculate probabilities from bias
		cpuSoftmax(bias, res, N);
		
		// Sample from the distribution
		float randNum = random(0, 1);
		for (int i = 0; i < N; i++)
		{
			randNum -= res[i];
			if (randNum <= 0)
			{
				sample = i;
				break;
			}
		}
		
		// should the sampled index be increased or decreased
		gradient = res[sample] > goal[sample] ? -LEARNING_RATE : LEARNING_RATE;
		
		// calculate gradient of the loss function
		cpuSoftmaxGradient(res, &gradient, &sample, grad, N);
		
		// update bias
		for (int i = 0; i < N; i++)
			bias[i] += grad[i];
	}
	
	cout << "Goal: ";
	for (int i = 0; i < N; i++)
		cout << goal[i] << " ";
	cout << "\n";

	cout << "Res: ";
	for (int i = 0; i < N; i++)
		cout << res[i] << " ";
	cout << "\n";

	cout << "Error: ";
	for (int i = 0; i < N; i++)
		cout << abs(goal[i] - res[i]) << " ";
	cout << "\n";

	cout << "Bias: ";
	for (int i = 0; i < N; i++)
		cout << bias[i] << " ";
	cout << "\n";

	return 0;
}
