//#include <cublas_v2.h>
//#include <curand.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>

using std::cout;
using std::vector;
using std::sort;
using std::exp;
using std::min;
using std::max;
using std::ofstream;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;

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

namespace GlobalVars
{
	Random random(Random::MakeSeed(0));
	constexpr uint32_t ACTIONS = 3;
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min, float max)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GlobalVars::random.Rfloat(min, max);
}

const static void cpuClippedLinearUnit(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	for (size_t counter = size; counter--;)
		outputMatrix[counter] = min(1.0f, max(-1.0f, inputMatrix[counter]));
}

const static void cpuClippedLinearUnitGradient(float* inputMatrix, float* gradientMatrix, float* outputMatrix, uint32_t size) {
	float input;
	float gradient;
	bool greaterZero;
	for (size_t counter = size; counter--;)
	{
		input = inputMatrix[counter];
		gradient = gradientMatrix[counter];
		greaterZero = gradient > 0;
		gradient = (greaterZero << 1) - 1;
		outputMatrix[counter] = (((input > 1) ^ greaterZero) || ((input >= -1) ^ greaterZero)) * gradient;
	}
}

int main()
{
	const uint32_t size = 10;

	float* inputMatrix = new float[size];
	float* outputMatrix = new float[size];
	float* gradientMatrix = new float[size];
	float* gradientOutputMatrix = new float[size];
	
	cpuGenerateUniform(inputMatrix, size, -2.0f, 2.0f);
	
	// print input matrix
	for (uint32_t counter = 0; counter < size; counter++)
		cout << inputMatrix[counter] << ' ';
	cout << '\n';
	
	cpuClippedLinearUnit(inputMatrix, outputMatrix, size);
	
	// print output matrix
	for (uint32_t counter = 0; counter < size; counter++)
		cout << outputMatrix[counter] << ' ';
	cout << '\n';

	cpuGenerateUniform(gradientMatrix, size, -1.0f, 1.0f);
	
	// print gradient matrix
	for (uint32_t counter = 0; counter < size; counter++)
		cout << gradientMatrix[counter] << ' ';
	cout << '\n';
	
	cpuClippedLinearUnitGradient(inputMatrix, gradientMatrix, gradientOutputMatrix, size);
	
	// print gradient output matrix
	for (uint32_t counter = 0; counter < size; counter++)
		cout << gradientOutputMatrix[counter] << ' ';
	cout << '\n';

	return 0;
}