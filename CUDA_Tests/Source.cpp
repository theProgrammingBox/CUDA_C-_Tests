#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

/*
Project Description:
This project is a CUDA GPU Memory Mapping test. The goal is to find all the
memory fragmentation sizes that can be allocated on the GPU. This is done by
allocating a large block of memory and then freeing it until the allocation
fails. The size of the allocation is then recorded and not deallocated. This
process is repeated until a memory of size 1 can no longer be allocated. The
size of the allocations are then printed to the console.
*/

int main()
{
	std::vector<std::pair<size_t, float*>*> fragments;

	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	freeMem >>= 2;
	printf("Free memory: %llu floats\n", freeMem);

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
			printf("Low: %llu, High: %llu, Guess: %llu\n", low, high, guess);
		} while (low <= high);
		low--;

		if (low > 0)
		{
			frag->first = low;

			err = cudaMalloc((void**)&frag->second, low << 2);
			if (err != cudaSuccess)
				printf("Failed to allocate memory of size %llu\n", low);
			else
				printf("Allocated %llu floats\n", low);

			freeMem -= low;
			printf("Free memory: %llu floats\n", freeMem);

			fragments.push_back(frag);
		}
	} while (low > 0);
	delete frag;

	for (auto& frag : fragments)
	{
		printf("Allocated %llu floats at %p\n", frag->first, frag->second);
		cudaFree(frag->second);
		delete frag;
	}

	return 0;
}