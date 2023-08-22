#include <stdio.h>	// printf
#include <assert.h>	// assert
#include <vector>	// std::vector

typedef uint32_t u32;
typedef float f32;
typedef double f64;

struct GpuMemoryManager
{
	struct MemFrag
	{
		f32* address;
		size_t size;
		size_t staticSize;
		size_t dynamicSize;
		f64 ratio;
	};

	struct TensorData
	{
		f32** address;
		size_t size;
		f64 ratio;
		MemFrag* frag;
	};

	std::vector<MemFrag*> MemFrags;
	std::vector<TensorData*> dynamicTensors;
	std::vector<TensorData*> staticTensors;

	GpuMemoryManager()
	{
		printf("Initializing GPU memory manager...\n");

		for (u32 i = 0; i < 2; ++i)
		{
			MemFrag* frag = new MemFrag;
			frag->address = nullptr;
			frag->size = i * 1024 + 1024;
			frag->dynamicSize = 0;
			MemFrags.emplace_back(frag);
		}

		assert(MemFrags.size() > 0);
	}

	void ManageDynamic(f32** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		dynamicTensors.emplace_back(tensorData);
	}

	void ManageStatic(f32** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		staticTensors.emplace_back(tensorData);
	}

	void Print()
	{
		for (auto& frag : MemFrags)
			printf("Frag size: %d, address: %p, ratio: %f\n", frag->size, frag->address, frag->ratio);
		printf("\n");

		for (auto& tensor : dynamicTensors)
			printf("Dynamic tensor size: %d, address: %p, ratio: %f\n", tensor->size, tensor->address, tensor->ratio);
		printf("\n");

		for (auto& tensor : staticTensors)
			printf("Static tensor size: %d, address: %p, ratio: %f\n", tensor->size, tensor->address, tensor->ratio);
		printf("\n");
	}

	void Allocate()
	{
		size_t fragSize = 0;
		size_t dynamicTensorSize = 0;

		for (auto& frag : MemFrags)
			fragSize += frag->size;
		for (auto& tensor : staticTensors)
			fragSize -= tensor->size;
		for (auto& tensor : dynamicTensors)
			dynamicTensorSize += tensor->size;

		assert(fragSize > 0);

		for (auto& frag : MemFrags)
			frag->ratio = (f64)frag->size / fragSize;
		for (auto& tensor : staticTensors)
			tensor->ratio = (f64)tensor->size / fragSize;
		for (auto& tensor : dynamicTensors)
			tensor->ratio = (f64)tensor->size / dynamicTensorSize;

		Print();

		f64 bestScore = DBL_MAX;
		std::vector<MemFrag*> bestCombination(staticTensors.size() + dynamicTensors.size());
		allocateStatic(0, bestScore, bestCombination);

		size_t largestN = 0;
		f64 smallestRatio = 0;
		for (u32 i = 0; i < staticTensors.size(); ++i)
		{
			bestCombination[i]->ratio -= staticTensors[i]->ratio;
			bestCombination[i]->staticSize += staticTensors[i]->size;
		}

		for (auto& frag : MemFrags)
			printf("ratio: %f, static Size: %d\n", frag->ratio, frag->staticSize);
		printf("\n");

		for (u32 i = 0; i < dynamicTensors.size(); ++i)
		{
			bestCombination[i + staticTensors.size()]->ratio -= dynamicTensors[i]->ratio;
			bestCombination[i + staticTensors.size()]->dynamicSize += dynamicTensors[i]->size;
		}

		for (auto& frag : MemFrags)
			printf("ratio: %f, dynamic size: %d\n", frag->ratio, frag->dynamicSize);
		printf("\n");

		for (auto& frag : MemFrags)
		{
			if (frag->ratio < smallestRatio)
			{
				smallestRatio = frag->ratio;
				largestN = (frag->size - frag->staticSize) / frag->dynamicSize;
				printf("largestN: %d\n", largestN);
			}
		}

		// dynamic size is largestN * dynamicSize, now double check that static plus dynamic is <= frag size
		for (auto& frag : MemFrags)
			printf("size: %zu, static size: %zu, dynamic size: %zu, leftover: %zu\n", frag->size, frag->staticSize, frag->dynamicSize, frag->size - frag->staticSize - frag->dynamicSize * largestN);
		printf("\n");
	}

	void allocateStatic(u32 tensorIdx, f64& bestScore, std::vector<MemFrag*>& bestCombination)
	{
		if (tensorIdx == staticTensors.size())
		{
			for (MemFrag* frag : MemFrags)
				if (frag->ratio < 0)
					return;

			allocateDynamic(0, bestScore, bestCombination);
			return;
		}

		for (MemFrag* frag : MemFrags)
		{
			staticTensors[tensorIdx]->frag = frag;
			f64 score = frag->ratio;
			frag->ratio -= staticTensors[tensorIdx]->ratio;
			allocateStatic(tensorIdx + 1, bestScore, bestCombination);
			frag->ratio = score;
		}
	}

	void allocateDynamic(u32 tensorIdx, f64& bestScore, std::vector<MemFrag*>& bestCombination)
	{
		if (tensorIdx == dynamicTensors.size())
		{
			f64 score = 0;
			for (MemFrag* frag : MemFrags)
				score += abs(frag->ratio);
			
			if (score < bestScore)
			{
				bestScore = score;
				for (int i = 0; i < staticTensors.size(); ++i)
					bestCombination[i] = staticTensors[i]->frag;
				for (int i = 0; i < dynamicTensors.size(); ++i)
					bestCombination[i + staticTensors.size()] = dynamicTensors[i]->frag;
			}
			return;
		}

		for (MemFrag* frag : MemFrags)
		{
			dynamicTensors[tensorIdx]->frag = frag;
			f64 score = frag->ratio;
			frag->ratio -= dynamicTensors[tensorIdx]->ratio;
			allocateDynamic(tensorIdx + 1, bestScore, bestCombination);
			frag->ratio = score;
		}
	}
};

int main()
{
	GpuMemoryManager gpuMemoryManager;

	f32* staticArr1 = nullptr;
	f32* staticArr2 = nullptr;
	f32* dynamicArr1 = nullptr;
	f32* dynamicArr2 = nullptr;

	gpuMemoryManager.ManageStatic(&staticArr1, 32);
	gpuMemoryManager.ManageStatic(&staticArr2, 64);
	gpuMemoryManager.ManageDynamic(&dynamicArr1, 3);
	gpuMemoryManager.ManageDynamic(&dynamicArr2, 4);

	gpuMemoryManager.Allocate();
	gpuMemoryManager.Print();

	return 0;
}