//#include <cublas_v2.h>
//#include <curand.h>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

void fill(float* arr, float val, uint32_t size)
{
	for (uint32_t i = size; i--;)
		memcpy(arr + i, &val, sizeof(float));
}

int main()
{
	const uint32_t N = 10;
	float* arr = new float[N];

	fill(arr, -1.0f, N);

	cout << "arr: ";
	for (uint32_t i = 0; i < N; ++i)
	{
		cout << arr[i] << " ";
	}
	cout << "\n";

	return 0;
}