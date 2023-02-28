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

class Example : public olc::PixelGameEngine
{
public:
	uint32_t vecDim = 2;
	float* vec1;
	float* vec2;
	float* vec1Derivative;
	float* vec2Derivative;

	float orgin[2];
	
	bool OnUserCreate() override
	{
		vec1 = new float[vecDim];
		vec2 = new float[vecDim];
		vec1Derivative = new float[vecDim];
		vec2Derivative = new float[vecDim];

		cpuGenerateUniform(vec1, vecDim, -1, 1);
		cpuGenerateUniform(vec2, vecDim, -1, 1);

		orgin[0] = ScreenWidth() * 0.5f;
		orgin[1] = ScreenHeight() * 0.5f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetMouse(0).bHeld)
		{
			vec1[0] = (GetMouseX() - orgin[0]) * 0.01f;
			vec1[1] = (GetMouseY() - orgin[1]) * 0.01f;
		}
		if (GetMouse(1).bHeld)
		{
			vec2[0] = (GetMouseX() - orgin[0]) * 0.01f;
			vec2[1] = (GetMouseY() - orgin[1]) * 0.01f;
		}

		Clear(olc::BLACK);
		DrawLine(orgin[0], orgin[1], orgin[0] + vec1[0] * 100, orgin[1] + vec1[1] * 100, olc::RED);
		DrawLine(orgin[0], orgin[1], orgin[0] + vec2[0] * 100, orgin[1] + vec2[1] * 100, olc::GREEN);
		
		float vecOneSquaredMagnitude = vec1[0] * vec1[0] + vec1[1] * vec1[1];
		float vecTwoSquaredMagnitude = vec2[0] * vec2[0] + vec2[1] * vec2[1];
		float magnitudeProduct = vecOneSquaredMagnitude * vecTwoSquaredMagnitude;
		float inverseSqrtMagnitudeProduct = invSqrt(magnitudeProduct);
		float vecOneDotVecTwo = vec1[0] * vec2[0] + vec1[1] * vec2[1];
		float cosTheta = vecOneDotVecTwo * inverseSqrtMagnitudeProduct;
		float cosThetaTarget = 1.0f;
		
		float cosThetaDerivative = 1;// cosThetaTarget - cosTheta;
		for (uint32_t counter = vecDim; counter--;)
		{
			vec1Derivative[counter] = (vec2[counter] - vec1[counter] * vecOneDotVecTwo / vecOneSquaredMagnitude) * inverseSqrtMagnitudeProduct * cosThetaDerivative;
			vec2Derivative[counter] = (vec1[counter] - vec2[counter] * vecOneDotVecTwo / vecTwoSquaredMagnitude) * inverseSqrtMagnitudeProduct * cosThetaDerivative;
			vec1[counter] += vec1Derivative[counter] * 0.001f;
			vec2[counter] += vec2Derivative[counter] * 0.001f;
		}
		
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