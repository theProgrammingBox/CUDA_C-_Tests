#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <chrono>
#include <iostream>

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

namespace GLOBAL
{
	Random random(Random::MakeSeed(0));
	constexpr float ZEROF = 0.0f;
	constexpr float ONEF = 1.0f;
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GLOBAL::random.Rfloat(min, max);
}

float invSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

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

void cpuLeakyRelu(float* input, float* output, uint32_t size)
{
	// reflection of relu
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
}

void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	// reflection of relu derivative
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
}

class Example : public olc::PixelGameEngine
{
public:
	uint32_t vecDim = 2;
	uint32_t inputDim = 2;
	float* vecGoal;
	float* input;
	float* weight;
	float* product;
	float* activation;
	float* activationDerivitive;
	float* productDerivitive;
	float* weightDerivitive;

	float orgin[2];
	
	bool OnUserCreate() override
	{
		vecGoal = new float[vecDim];
		input = new float[inputDim];
		weight = new float[inputDim * vecDim];
		product = new float[vecDim];
		activation = new float[vecDim];
		activationDerivitive = new float[vecDim];
		productDerivitive = new float[vecDim];
		weightDerivitive = new float[inputDim * vecDim];

		cpuGenerateUniform(vecGoal, vecDim, -1, 1);
		cpuGenerateUniform(input, inputDim, -1, 1);
		cpuGenerateUniform(weight, inputDim * vecDim, -1, 1);

		orgin[0] = ScreenWidth() * 0.5f;
		orgin[1] = ScreenHeight() * 0.5f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetMouse(0).bPressed)
		{
			vecGoal[0] = (GetMouseX() - orgin[0]) * 0.01f;
			vecGoal[1] = (GetMouseY() - orgin[1]) * 0.01f;
		}

		cpuSgemmStridedBatched(
			false, false,
			vecDim, 1, inputDim,
			&GLOBAL::ONEF,
			weight, vecDim, 0,
			input, inputDim, 0,
			&GLOBAL::ZEROF,
			product, vecDim, 0,
			1);

		cpuLeakyRelu(product, activation, vecDim);

		Clear(olc::BLACK);
		DrawLine(orgin[0], orgin[1], orgin[0] + activation[0] * 100, orgin[1] + activation[1] * 100, olc::RED);
		DrawLine(orgin[0], orgin[1], orgin[0] + vecGoal[0] * 100, orgin[1] + vecGoal[1] * 100, olc::GREEN);
		
		float vecOneSquaredMagnitude = activation[0] * activation[0] + activation[1] * activation[1];
		float vecTwoSquaredMagnitude = vecGoal[0] * vecGoal[0] + vecGoal[1] * vecGoal[1];
		float magnitudeProduct = vecOneSquaredMagnitude * vecTwoSquaredMagnitude;
		float inverseSqrtMagnitudeProduct = invSqrt(magnitudeProduct);
		float vecOneDotVecTwo = activation[0] * vecGoal[0] + activation[1] * vecGoal[1];
		float cosTheta = vecOneDotVecTwo * inverseSqrtMagnitudeProduct;
		float cosThetaTarget = 1.0f;
		
		float vec1DerivativeMagnitude = 0;
		float vec2DerivativeMagnitude = 0;
		for (uint32_t counter = vecDim; counter--;)
		{
			/*// old
			float vec1Derivative = (vec2[counter] - vec1[counter] * vecOneDotVecTwo / vecOneSquaredMagnitude) * inverseSqrtMagnitudeProduct;
			float vec2Derivative = (vec1[counter] - vec2[counter] * vecOneDotVecTwo / vecTwoSquaredMagnitude) * inverseSqrtMagnitudeProduct;*/

			// new
			activationDerivitive[counter] = (vecGoal[counter] * vecOneSquaredMagnitude - activation[counter] * vecOneDotVecTwo) * inverseSqrtMagnitudeProduct;
		}

		cpuLeakyReluDerivative(product, activationDerivitive, productDerivitive, vecDim);

		cpuSgemmStridedBatched(
			false, true,
			vecDim, inputDim, 1,
			&GLOBAL::ONEF,
			productDerivitive, vecDim, 0,
			input, inputDim, 0,
			&GLOBAL::ZEROF,
			weightDerivitive, vecDim, 0,
			1);
		
		for (uint32_t counter = inputDim * vecDim; counter--;)
			weight[counter] += weightDerivitive[counter] * 0.01f;
		
		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(256, 256, 4, 4))
		demo.Start();
	return 0;
}