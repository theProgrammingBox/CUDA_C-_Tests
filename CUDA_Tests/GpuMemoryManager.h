#pragma once

#include "Header.cuh"

struct GpuMemoryManager
{
	struct MemFrag
	{
		size_t size;
		float* address;
	};

	std::vector<MemFrag*> MemFrags;

	~GpuMemoryManager()
	{
		for (MemFrag* frag : MemFrags)
		{
			cudaFree(frag->address);
			delete frag;
		}
	}

	void MapGpuMem()
	{
		printf("Mapping GPU memory\n");
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);

		MemFrag* frag;
		size_t low, high, guess;
		cudaError_t err;
		do
		{
			frag = new MemFrag;
			low = 1, high = freeMem;
			do
			{
				guess = (low + high) >> 1;
				err = cudaMalloc((void**)&frag->address, guess);
				err == cudaSuccess ? low = guess + 1 : high = guess - 1;
				cudaFree(frag->address);
			} while (low <= high);
			low--;

			if (low > 0)
			{
				frag->size = low;
				cudaMalloc((void**)&frag->address, low);
				freeMem -= low;
				MemFrags.emplace_back(frag);
			}
		} while (low > 0);
		delete frag;
	}

	void PrintGpuMem()
	{
		for (MemFrag* frag : MemFrags)
			printf("Allocated %zu bytes at %p\n", frag->size, frag->address);
	}
};