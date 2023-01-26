#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::cout;
using std::vector;
using std::sort;
using std::exp;
using std::min;
using std::max;
using std::ofstream;

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

void cpuCLU(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	for (size_t counter = size; counter--;)
		outputMatrix[counter] = min(1.0f, max(-1.0f, inputMatrix[counter]));
}

void cpuSoftmax(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	float sum = 0;
	for (uint32_t counter = size; counter--;)
	{
		outputMatrix[counter] = exp(inputMatrix[counter]);
		sum += outputMatrix[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		outputMatrix[counter] *= sum;
}

namespace GlobalVars
{
	Random random(Random::MakeSeed(0));
	constexpr uint32_t INPUT = 10;
	constexpr uint32_t ACTIONS = 9;
}

class Species
{
public:
	static constexpr uint32_t HIDDEN = 16;
	float* weight1;
	float* weight2;

	Species()
	{
		weight1 = new float[HIDDEN * GlobalVars::INPUT];
		weight2 = new float[GlobalVars::ACTIONS * HIDDEN];
	}

	~Species()
	{
		delete[] weight1;
		delete[] weight2;
	}
};

class Agent
{
public:
	static constexpr float ONE = 1.0f;
	static constexpr float ZERO = 0.0f;
	Species* species;
	
	struct Layer
	{
		Agent* agent;
		float* inputMatrix;
		float* hiddenMatrix;
		float* outputMatrix;
		float* actionMatrix;

		Layer() : inputMatrix(new float[GlobalVars::INPUT]), hiddenMatrix(new float[Species::HIDDEN]), outputMatrix(new float[GlobalVars::ACTIONS]), actionMatrix(new float[GlobalVars::ACTIONS]) {}
		
		Layer(Layer&& other) noexcept : inputMatrix(other.inputMatrix), hiddenMatrix(other.hiddenMatrix), outputMatrix(other.outputMatrix), actionMatrix(other.actionMatrix)
		{
			other.inputMatrix = nullptr;
			other.hiddenMatrix = nullptr;
			other.outputMatrix = nullptr;
			other.actionMatrix = nullptr;
		}
		
		~Layer()
		{
			delete[] inputMatrix;
			delete[] hiddenMatrix;
			delete[] outputMatrix;
			delete[] actionMatrix;
		}
		
		uint32_t FeedForward(Agent* agent, float* input)
		{
			this->agent = agent;
			memcpy(inputMatrix, input, sizeof(float) * GlobalVars::INPUT);
			cpuSgemmStridedBatched(false, false, Species::HIDDEN, 1, GlobalVars::INPUT, &ONE, agent->species->weight1, Species::HIDDEN, 0, inputMatrix, GlobalVars::INPUT, 0, &ZERO, hiddenMatrix, Species::HIDDEN, 0, 1);
			cpuCLU(hiddenMatrix, hiddenMatrix, Species::HIDDEN);
			cpuSgemmStridedBatched(false, false, GlobalVars::ACTIONS, 1, Species::HIDDEN, &ONE, agent->species->weight2, GlobalVars::ACTIONS, 0, hiddenMatrix, Species::HIDDEN, 0, &ZERO, outputMatrix, GlobalVars::ACTIONS, 0, 1);
			cpuCLU(outputMatrix, outputMatrix, GlobalVars::ACTIONS);
			cpuSoftmax(outputMatrix, actionMatrix, GlobalVars::ACTIONS);
			
			float number = GlobalVars::random.Rfloat(0.0f, 1.0f);
			uint32_t sample = 0;
			while (true)
			{
				number -= actionMatrix[sample];
				if (number < 0) break;
				sample++;
				sample -= (sample == GlobalVars::ACTIONS) * GlobalVars::ACTIONS;
			}
			return sample;
		}
	};
	
	vector<Layer> layers;

	void AddToLayers(float* input)
	{
		layers.emplace_back();
		cout << "Action:" << layers.back().FeedForward(this, input) << '\n';;
	}

	void RemoveFromVector() { layers.pop_back(); }
	void ClearVector() { layers.clear(); }
	void Reset(Species* species)
	{
		this->species = species;
		ClearVector();
	}
};

int main()
{
	Species species;
	Agent agent;
	agent.Reset(&species);

	float* input = new float[GlobalVars::INPUT];
	for (uint32_t counter = GlobalVars::INPUT; counter--;)
		input[counter] = GlobalVars::random.Rfloat(-1.0f, 1.0f);
	
	for (uint32_t counter = 10; counter--;)
		agent.AddToLayers(input);

	return 0;
}