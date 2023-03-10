#include "iostream"
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::high_resolution_clock;

class Random
{
public:
	Random(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
	}

	static uint32_t MakeSeed(uint32_t seed = 0)	// make seed from time and seed
	{
		uint32_t result = seed;
		result = Hash((uint8_t*)&result, sizeof(result), nanosecond());
		result = Hash((uint8_t*)&result, sizeof(result), microsecond());
		return result;
	}

	void Seed(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
	}

	uint32_t Ruint32()	// XORSHIFT128+
	{
		uint64_t a = state[0];
		uint64_t b = state[1];
		state[0] = b;
		a ^= a << 23;
		state[1] = a ^ b ^ (a >> 18) ^ (b >> 5);
		return uint32_t((state[1] + b) >> 16);
	}

	float Rfloat(float min = 0, float max = 1) { return min + (max - min) * Ruint32() * 2.3283064371e-10; }

	static uint32_t Hash(const uint8_t* key, size_t len, uint32_t seed = 0)	// MurmurHash3
	{
		uint32_t h = seed;
		uint32_t k;
		for (size_t i = len >> 2; i; i--) {
			memcpy(&k, key, sizeof(uint32_t));
			key += sizeof(uint32_t);
			h ^= murmur_32_scramble(k);
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}
		k = 0;
		for (size_t i = len & 3; i; i--) {
			k <<= 8;
			k |= key[i - 1];
		}
		h ^= murmur_32_scramble(k);
		h ^= len;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}

private:
	uint64_t state[2];

	static uint32_t murmur_32_scramble(uint32_t k) {
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		return k;
	}

	static uint32_t nanosecond() { return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
	static uint32_t microsecond() { return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
};

void cpuSgemmStridedBatched(
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

void cpuSaxpy(int N, const float* alpha, const float* X, int incX, float* Y, int incY)
{
	for (int i = N; i--;)
		Y[i * incY] += *alpha * X[i * incX];
}

void cpuLeakyRelu(float* input, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
}

void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
}

void cpuSoftmax(float* input, float* output, uint32_t size)
{
	float sum = 0;
	float max = input[0];
	for (uint32_t counter = size; counter--;)
		max = std::max(max, input[counter]);
	for (uint32_t counter = size; counter--;)
	{
		output[counter] = std::exp(input[counter] - max);
		sum += output[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		output[counter] *= sum;
}

void cpuSoftmaxDerivative(float* inputOutput, float* output, bool endState, uint32_t action, uint32_t size)
{
	float sampledProbability = inputOutput[action];
	float gradient = (endState - sampledProbability);
	for (uint32_t counter = size; counter--;)
		output[counter] = gradient * inputOutput[counter] * ((counter == action) - sampledProbability);
}

void PrintMatrix(float* arr, uint32_t rows, uint32_t cols, const char* label) {
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

namespace GLOBAL
{
	Random RANDOM(Random::MakeSeed());
	constexpr float ZEROF = 0.0f;
	constexpr float ONEF = 1.0f;

	constexpr uint32_t BATCH_SIZE = 1;
	constexpr uint32_t ITERATIONS = 1620 * BATCH_SIZE;
	constexpr float LEARNING_RATE = 0.1f;
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GLOBAL::RANDOM.Rfloat(min, max);
}

int main()
{
	constexpr uint32_t INPUTS = 10;
	constexpr uint32_t OUTPUTS = 9;
	
	float* bias;
	float* softmaxMatrix;
	uint32_t sampledAction;
	float* biasDerivative;

	bias = new float[OUTPUTS];
	softmaxMatrix = new float[OUTPUTS];
	biasDerivative = new float[OUTPUTS];

	cpuGenerateUniform(bias, OUTPUTS, -1, 1);

	for (uint32_t i = 100; i--;)
	{
		cpuSoftmax(bias, softmaxMatrix, OUTPUTS);

		float randomNumber = GLOBAL::RANDOM.Rfloat();
		sampledAction = 0;
		for (;;)
		{
			randomNumber -= softmaxMatrix[sampledAction];
			if (randomNumber <= 0)
				break;
			sampledAction -= (++sampledAction >= OUTPUTS) * OUTPUTS;
		}
		PrintMatrix(softmaxMatrix, 1, OUTPUTS, "softmaxMatrix");
		printf("sampledAction: %d\n", sampledAction);

		cpuSoftmaxDerivative(softmaxMatrix, biasDerivative, sampledAction == 4, sampledAction, OUTPUTS);

		cpuSaxpy(OUTPUTS, &GLOBAL::LEARNING_RATE, biasDerivative, 1, bias, 1);
	}
	
	return 0;
}