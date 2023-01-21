//#include <cublas_v2.h>
//#include <curand.h>
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
using std::ceil;
using std::exp;
using std::ofstream;

int main()
{
	// Testing precidence
	float num;

	num = 3;
	cout << (~uint32_t(num) >> 31 << 1 - 1) << "\n";
	/*cout << (1 - (uint32_t(num) & 0x80000000 >> 31)) << "\n";
	cout << "Expected: 1\n";*/
	cout << "----------------\n";

	num = -2;
	cout << (~uint32_t(num) >> 31 << 1 - 1) << "\n";
	/*cout << (1 - (uint32_t(num) & 0x80000000 >> 31)) << "\n";
	cout << "Expected: -1\n";*/

	return 0;
}