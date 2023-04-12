#include <iostream>
#include <chrono>

int main()
{
	const uint32_t samples = 1000000000;
	uint32_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	int32_t temp;
	int32_t constSeed;
	uint32_t count;
	for (uint8_t i = samples; i--;)
	{
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
	}

	count = 0;
	auto start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = samples; i--;)
	{
		constSeed = seed;
		temp = seed;
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
		temp += seed;
		count += abs(temp) < constSeed;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	printf("Time: %f, count: %d\n", elapsed.count(), count);

	count = 0;
	start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = samples; i--;)
	{
		constSeed = seed;
		temp = seed;
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
		temp += seed;
		count += (temp ^ (temp >> 31)) - (temp >> 31) < constSeed;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	printf("Time: %f, count: %d\n", elapsed.count(), count);

	count = 0;
	start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = samples; i--;)
	{
		constSeed = seed;
		temp = seed;
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
		temp += seed;
		count += abs(temp) < constSeed;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	printf("Time: %f, count: %d\n", elapsed.count(), count);

	count = 0;
	start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = samples; i--;)
	{
		constSeed = seed;
		temp = seed;
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
		temp += seed;
		count += (temp ^ (temp >> 31)) - (temp >> 31) < constSeed;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	printf("Time: %f, count: %d\n", elapsed.count(), count);

	count = 0;
	start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = samples; i--;)
	{
		constSeed = seed;
		temp = seed;
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
		temp += seed;
		count += abs(temp) < constSeed;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	printf("Time: %f, count: %d\n", elapsed.count(), count);

	count = 0;
	start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = samples; i--;)
	{
		constSeed = seed;
		temp = seed;
		seed ^= seed << 13;
		seed ^= seed >> 17;
		seed ^= seed << 5;
		temp += seed;
		count += (temp ^ (temp >> 31)) - (temp >> 31) < constSeed;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	printf("Time: %f, count: %d\n", elapsed.count(), count);
	
	return 0;
}