//#include <cublas_v2.h>
//#include <curand.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

using std::cout;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::sort;

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

static xorwow32 random(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());


int main()
{
	// initialize the vector with floats
	// make an array holding the float's references
	// sort the vector
	// print the vector
	// print the dereferenced references

	// the vector should be sorted
	// the dereferenced references should not be sorted
	
	vector<int*> arr;

	//	temp vars	//
	int counter;
	int** tempArr;
	int** tempArr2;
	//		//		//

	for (int i = 4; i--;) arr.push_back(new int((random() & 7) - 3));

	int** arr2 = new int* [arr.size()];
	
	for (counter = arr.size(), tempArr = arr.data(), tempArr2 = arr2;
		counter--; tempArr++, tempArr2++) *tempArr2 = *tempArr;
	
	cout << "PreSort:\n";
	for (counter = arr.size(), tempArr = arr.data();
		counter--; tempArr++) cout << **tempArr << ' ';
	cout << "\n\n";

	sort(arr.begin(), arr.end(), [](int* a, int* b) { return *a < *b; });

	cout << "PostSort:\n";
	for (counter = arr.size(), tempArr = arr.data();
		counter--; tempArr++) cout << **tempArr << ' ';
	cout << "\n\n";
	
	cout << "DeRef:\n";
	for (counter = arr.size(), tempArr = arr2;
		counter--; tempArr++) cout << **tempArr << ' ';
	cout << "\n\n";

	return 0;
}