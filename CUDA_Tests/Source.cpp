#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

void MapGpuMem(std::vector<std::pair<size_t, float*>*>* fragments)
{
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	freeMem >>= 2;
	//printf("Free memory: %zu floats\n", freeMem);

	std::pair<size_t, float*>* frag;
	size_t low, high, guess;
	cudaError_t err;
	do
	{
		frag = new std::pair<size_t, float*>(high, nullptr);
		low = 1, high = freeMem;
		do
		{
			guess = (low + high) >> 1;
			err = cudaMalloc((void**)&frag->second, guess << 2);
			err == cudaSuccess ? low = guess + 1 : high = guess - 1;
			cudaFree(frag->second);
			//printf("Low: %zu, High: %zu, Guess: %zu\n", low, high, guess);
		} while (low <= high);
		low--;

		if (low > 0)
		{
			frag->first = low;

			cudaMalloc((void**)&frag->second, low << 2);
			/*err = cudaMalloc((void**)&frag->second, low << 2);
			if (err != cudaSuccess)
				printf("Failed to allocate memory of size %zu\n", low);
			else
				printf("Allocated %zu floats\n", low);*/

			freeMem -= low;
			//printf("Free memory: %zu floats\n", freeMem);

			fragments->emplace_back(frag);
		}
	} while (low > 0);
	delete frag;
}

void PrintGpuMem(std::vector<std::pair<size_t, float*>*>* fragments)
{
	for (std::pair<size_t, float*>* frag : *fragments)
	{
		printf("Allocated %zu floats at %p\n", frag->first, frag->second);
		cudaFree(frag->second);
		delete frag;
	}
}

int main()
{
	float* a;
	cudaMalloc((void**)&a, 345 << 2);

	std::vector<std::pair<size_t, float*>*> fragments;
	MapGpuMem(&fragments);
	PrintGpuMem(&fragments);

	cudaFree(a);

	return 0;
}