#include <chrono>
#include <iostream>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::max;
using std::cout;

/*
TODO:
1. test if if is faster then for loop given a bool. (swap function)
2. test speed of division vs multiplication
3. implement cpuNormDotProduct and its gradient
*/

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

namespace GLOBAL
{
	Random random(Random::MakeSeed(0));
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GLOBAL::random.Rfloat(min, max);
}

void PrintMatrix(uint32_t rows, uint32_t cols, float* arr, const char* label = "") {
	if (label[0] != '\0')
		printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
		{
			printf("%8.2f ", arr[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

float invSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

void cpuNormDotProduct(uint32_t size, float* vec1, float* vec2, float* result) {
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	float dot = 0.0f;
	for (uint32_t i = size; i--;)
	{
		sum1 += vec1[i] * vec1[i];
		sum2 += vec2[i] * vec2[i];
		dot += vec1[i] * vec2[i];
	}
	*result = dot * invSqrt(sum1 * sum2);
}

void cpuNormDotProductDerivitive(uint32_t size, float* vec1, float* vec2, float* vec1Gradient, float* vec2Gradient) {
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	float dot = 0.0f;
	for (uint32_t i = size; i--;)
	{
		sum1 += vec1[i] * vec1[i];
		sum2 += vec2[i] * vec2[i];
		dot += vec1[i] * vec2[i];
	}
	float* vecOne = vec1;
	float* vecTwo = vec2;
	float* vecOneGradient = vec1Gradient;
	float* vecTwoGradient = vec2Gradient;
	float* total1 = &sum1;
	float* total2 = &sum2;
	
	for (uint32_t i = 2; i--;)
	{
		float invSqrtTotal = invSqrt(sum1 * sum2 * *total1 * *total1);
		for (uint32_t j = size; j--;)
		{
			vecOneGradient[j] = (vecTwo[j] * (*total1 - vecOne[j] * vecOne[j]) + vecOne[j] * (vecOne[j] * vecTwo[j] - dot)) * invSqrtTotal;
		}
		std::swap(vecOne, vecTwo);
		std::swap(vecOneGradient, vecTwoGradient);
		std::swap(total1, total2);
	}
}

#include <fstream>

using std::ofstream;
using std::ifstream;
using std::ios;

int main()
{
	uint32_t rows = 2;
	float* arr1 = new float[rows];
	float* arr2 = new float[rows];
	float* arr1Gradient = new float[rows];
	float* arr2Gradient = new float[rows];
	
	cpuGenerateUniform(arr1, rows, -1, 1);
	cpuGenerateUniform(arr2, rows, -1, 1);

	PrintMatrix(rows, 1, arr1, "arr1");
	PrintMatrix(rows, 1, arr2, "arr2");
	
	float result = 0.0f;
	cpuNormDotProduct(rows, arr1, arr2, &result);
	printf("result: %f\n", result);

	cpuNormDotProductDerivitive(rows, arr1, arr2, arr1Gradient, arr2Gradient);
	PrintMatrix(rows, 1, arr1Gradient, "arr1Gradient");
	PrintMatrix(rows, 1, arr2Gradient, "arr2Gradient");

	return 0;
}