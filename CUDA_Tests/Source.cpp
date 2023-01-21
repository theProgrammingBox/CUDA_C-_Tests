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
	bool g;
	for (float num = -2.0f; num < 2.0f; num += 0.1f)
	{
		cout << "Num: " << num << '\n';
		g = num > 0;
		cout << g << '\n';
		cout << ((g << 1) - 1.0f) << '\n';
	}

	return 0;
}