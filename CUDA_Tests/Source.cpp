#include <stdio.h>	// printf
#include <assert.h>	// assert
#include <vector>	// std::vector

typedef uint32_t u32;
typedef float f32;
typedef double f64;

struct GpuMemoryManager
{
	struct TensorData
	{
		f32** address;
		size_t size;
		f64 ratio;
	};

	struct MemFrag
	{
		f32* address;
		size_t size;
		f64 ratio;
	};

	std::vector<MemFrag> MemFrags;
	std::vector<TensorData*> dynamicTensors;
	std::vector<TensorData*> staticTensors;

	GpuMemoryManager()
	{
		printf("Initializing GPU memory manager...\n");

		for (u32 i = 0; i < 2; ++i)
		{
			MemFrag frag;
			frag.address = nullptr;
			frag.size = i * 1024 + 1024;
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

	void bruteForceStatic(size_t idx, double currentSum, std::vector<MemFrag>& currentCombination)
	{
		if (idx == staticTensors.size())
		{
			if (currentSum > 0)
			{
				bruteForceDynamic(0, currentSum, currentCombination);
			}
			return;
		}

		for (auto& frag : MemFrags)
		{
			double newSum = currentSum + staticTensors[idx]->ratio - frag.ratio;
			currentCombination.push_back(frag);

			bruteForceStatic(idx + 1, newSum, currentCombination);

			currentCombination.pop_back();
		}
	}

	void bruteForceDynamic(size_t idx, double currentSum, std::vector<MemFrag>& currentCombination)
	{
		if (idx == dynamicTensors.size())
		{
			if (std::abs(currentSum) < min)
			{
				min = std::abs(currentSum);
				bestCombination = currentCombination;
			}
			return;
		}

		for (auto& frag : MemFrags)
		{
			double newSum = currentSum + dynamicTensors[idx]->ratio - frag.ratio;
			currentCombination.push_back(frag);

			bruteForceDynamic(idx + 1, newSum, currentCombination);

			currentCombination.pop_back();
		}
	}

	void Allocate()
	{
		size_t fragSize = 0;
		size_t dynamicTensorSize = 0;

		for (auto& frag : MemFrags)
			fragSize += frag.size;
		for (auto& tensor : staticTensors)
			fragSize -= tensor->size;
		for (auto& tensor : dynamicTensors)
			dynamicTensorSize += tensor->size;

		assert(fragSize > 0);

		double min = DBL_MAX;
		std::vector<MemFrag> bestCombination;

		std::vector<MemFrag> currentCombination;
		bruteForceStatic(0, 0.0, currentCombination);
	}
};

int main()
{
	GpuMemoryManager gpuMemoryManager;

	f32* staticArr1 = nullptr;
	f32* staticArr2 = nullptr;
	f32* dynamicArr1 = nullptr;
	f32* dynamicArr2 = nullptr;

	gpuMemoryManager.ManageStatic(&staticArr1, 1024);
	gpuMemoryManager.ManageStatic(&staticArr2, 2000);
	gpuMemoryManager.ManageDynamic(&dynamicArr1, 3);
	gpuMemoryManager.ManageDynamic(&dynamicArr2, 4);

	gpuMemoryManager.Allocate();
	gpuMemoryManager.Print();

	return 0;
}