#include <chrono>
#include <iostream>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::max;
using std::cout;

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
}

void SimpleMatrixInit(uint32_t rows, uint32_t cols, float* arr) {
	memset(arr, 0, rows * cols * sizeof(float));
	uint32_t maxSteps = max(rows, cols);
	float stepx = (float)cols / maxSteps;
	float stepy = (float)rows / maxSteps;
	float x = 0.0f;
	float y = 0.0f;
	for (uint32_t step = maxSteps; step--;)
	{
		arr[uint32_t(y) * cols + uint32_t(x)] = (GlobalVars::random.Ruint32() & 1 << 1) - 1.0f + GlobalVars::random.Rfloat(-0.1f, 0.1f);
		x += stepx;
		y += stepy;
	}
}

void PrintMatrixFast(uint32_t rows, uint32_t cols, float* arr) {
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
		{
			printf("%7.4f ", arr[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void PrintMatrixSlow(uint32_t rows, uint32_t cols, float* arr) {
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
		{
			cout << arr[i * cols + j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
}

int main()
{
	const uint32_t rows = 13;
	const uint32_t cols = 11;
	
	float matrix[rows * cols];
	SimpleMatrixInit(rows, cols, matrix);
	
	auto start = high_resolution_clock::now();
	for (uint32_t counter = 4; counter--;)
		PrintMatrixSlow(rows, cols, matrix);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Fast: " << duration.count() << " microseconds\n";
	
	start = high_resolution_clock::now();
	for (uint32_t counter = 4; counter--;)
		PrintMatrixFast(rows, cols, matrix);
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	cout << "Slow: " << duration.count() << " microseconds\n";
	
	return 0;
}