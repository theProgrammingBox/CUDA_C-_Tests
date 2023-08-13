#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

void MapGpuMem(std::vector<std::pair<size_t, float*>*>* fragments)
{
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	freeMem >>= 2;

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
		} while (low <= high);
		low--;

		if (low > 0)
		{
			frag->first = low;
			cudaMalloc((void**)&frag->second, low << 2);
			freeMem -= low;
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
	std::vector<std::pair<size_t, float*>*> fragments;
	MapGpuMem(&fragments);
	PrintGpuMem(&fragments);

	return 0;
}