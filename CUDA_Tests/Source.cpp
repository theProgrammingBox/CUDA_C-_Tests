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

int main()
{
	vector<int> arr;

	arr.push_back(10);
	arr.push_back(2);

	int** arr2 = new int* [2];

	int* tempArr = arr.data();
	*arr2 = &*tempArr;
	*(arr2 + 1) = &*(tempArr + 1);

	cout << *arr2[0] << " " << *arr2[1] << "\n";

	//sort the vector
	sort(arr.begin(), arr.end());

	cout << *arr2[0] << " " << *arr2[1] << "\n";

	return 0;
}