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

struct xorwow32	//curand
{
	uint32_t state[6];
	uint32_t operator()()
	{
		uint32_t t = state[0] ^ (state[0] >> 2);
		memcpy(state, state + 1, 4 * sizeof(uint32_t));
		state[4] ^= (state[4] << 4) ^ (t ^ (t << 1));
		return (state[5] += 362437) + state[4];
	}
	
	xorwow32(uint32_t seed)
	{
		state[0] = seed ^ 123456789;
		state[1] = seed ^ 362436069;
		state[2] = seed ^ 521288629;
		state[3] = seed ^ 88675123;
		state[4] = seed ^ 5783321;
		state[5] = seed ^ 6615241;
	}
};

int main()
{
	xorwow32 rand(0);
	for (int i = 0; i < 10; i++)
		cout << rand() << "\n";

	return 0;
}