//#include <cublas_v2.h>
//#include <curand.h>
#include <iostream>
#include <vector>
#include <chrono>

using std::cout;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

struct xorwow32
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
};

int main()
{
	xorwow32 rand(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	for (int i = 0; i < 10; i++)
	{
		//cout << rand() << "\n";
		cout << rand(-1.0f, 1.0f) << "\n";
	}

	return 0;
}