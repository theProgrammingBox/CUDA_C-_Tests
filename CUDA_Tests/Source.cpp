#include <stdio.h>	// printf
#include <vector>	// std::vector

void FailIf(bool condition, const char* message)
{
	if (condition)
	{
		fprintf(stderr, "%s", message);
		exit(0);
	}
}

struct GpuMemoryManager
{
	struct MemoryData
	{
		float* address;
		size_t size;
		size_t dynamicSize;
		float ratio;
	};

	struct TensorData
	{
		float** address;
		size_t size;
		float ratio;
		MemoryData* memoryPtr;
	};

	std::vector<MemoryData*> availableMemory;
	std::vector<TensorData*> dynamicTensors;
	std::vector<TensorData*> staticTensors;

	GpuMemoryManager()
	{
		printf("Initializing GPU memory manager...\n\n");
		for (uint32_t i = 0; i < 2; ++i)
		{
			MemoryData* memoryPtr = new MemoryData;
			memoryPtr->size = i * 102 + 13;
			memoryPtr->address = new float[memoryPtr->size];
			memset(memoryPtr->address, 0, memoryPtr->size * sizeof(float));
			memoryPtr->dynamicSize = 0;
			availableMemory.emplace_back(memoryPtr);
			FailIf(memoryPtr->size <= 0, "Memory size is <= 0\n");
		}
		FailIf(availableMemory.size() <= 0, "No available memory\n");
	}

	~GpuMemoryManager()
	{
		for (auto& memoryPtr : availableMemory)
			delete[] memoryPtr->address;
	}

	void ManageStatic(float** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		staticTensors.emplace_back(tensorData);
		FailIf(tensorData->size <= 0, "Static Tensor size is <= 0\n");
	}

	void ManageDynamic(float** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		dynamicTensors.emplace_back(tensorData);
		FailIf(tensorData->size <= 0, "Dynamic Tensor size is <= 0\n");
	}

	void allocateStatic(uint32_t tensorIdx, float& largestRatio, std::vector<MemoryData*>& bestCombination, size_t& largestN)
	{
		if (tensorIdx == staticTensors.size())
			allocateDynamic(0, largestRatio, bestCombination, largestN);
		else
			for (MemoryData* memoryPtr : availableMemory)
				if (memoryPtr->size >= staticTensors[tensorIdx]->size)
				{
					staticTensors[tensorIdx]->memoryPtr = memoryPtr;
					memoryPtr->ratio -= staticTensors[tensorIdx]->ratio;
					memoryPtr->size -= staticTensors[tensorIdx]->size;
					allocateStatic(tensorIdx + 1, largestRatio, bestCombination, largestN);
					memoryPtr->ratio += staticTensors[tensorIdx]->ratio;
					memoryPtr->size += staticTensors[tensorIdx]->size;
				}
	}

	void allocateDynamic(uint32_t tensorIdx, float& largestRatio, std::vector<MemoryData*>& bestCombination, size_t& largestN)
	{
		if (tensorIdx == dynamicTensors.size())
		{
			float smallestRatio = 1;
			size_t size = 0;
			size_t dynamicSize = 0;
			for (auto& memoryPtr : availableMemory)
				if (memoryPtr->dynamicSize > 0 && memoryPtr->ratio < smallestRatio)
				{
					smallestRatio = memoryPtr->ratio;
					size = memoryPtr->size;
					dynamicSize = memoryPtr->dynamicSize;
				}

			if (smallestRatio > largestRatio)
			{
				if (dynamicSize > 0)
					largestN = size / dynamicSize;
				largestRatio = smallestRatio;

				for (int i = 0; i < staticTensors.size(); ++i)
					bestCombination[i] = staticTensors[i]->memoryPtr;
				for (int i = 0; i < dynamicTensors.size(); ++i)
					bestCombination[i + staticTensors.size()] = dynamicTensors[i]->memoryPtr;
			}
		}
		else
			for (MemoryData* memoryPtr : availableMemory)
			{
				dynamicTensors[tensorIdx]->memoryPtr = memoryPtr;
				memoryPtr->ratio -= dynamicTensors[tensorIdx]->ratio;
				memoryPtr->dynamicSize += dynamicTensors[tensorIdx]->size;
				allocateDynamic(tensorIdx + 1, largestRatio, bestCombination, largestN);
				memoryPtr->ratio += dynamicTensors[tensorIdx]->ratio;
				memoryPtr->dynamicSize -= dynamicTensors[tensorIdx]->size;
			}
	}

	void Allocate(size_t& largestN)
	{
		size_t fragSize = 0;
		size_t dynamicTensorSize = 0;

		for (auto& memoryPtr : availableMemory)
			fragSize += memoryPtr->size;
		for (auto& tensor : staticTensors)
		{
			FailIf(tensor->size > fragSize, "Static tensor size is larger than total memory size\n");
			fragSize -= tensor->size;
		}
		for (auto& tensor : dynamicTensors)
			dynamicTensorSize += tensor->size;

		for (auto& memoryPtr : availableMemory)
			memoryPtr->ratio = (float)memoryPtr->size / fragSize;
		for (auto& tensor : staticTensors)
			tensor->ratio = (float)tensor->size / fragSize;
		for (auto& tensor : dynamicTensors)
			tensor->ratio = (float)tensor->size / dynamicTensorSize;

		largestN = 0;
		float largestRatio = -1;
		std::vector<MemoryData*> bestCombination(staticTensors.size() + dynamicTensors.size());
		allocateStatic(0, largestRatio, bestCombination, largestN);

		FailIf(bestCombination[0] == nullptr, "No combination found\n");

		// allocate memory
		for (auto& memoryPtr : availableMemory)
			memoryPtr->dynamicSize = 0;

		for (int i = 0; i < staticTensors.size(); ++i)
		{
			MemoryData* memoryPtr = bestCombination[i];
			*staticTensors[i]->address = memoryPtr->address + memoryPtr->dynamicSize;
			memoryPtr->dynamicSize += staticTensors[i]->size;
		}

		for (int i = 0; i < dynamicTensors.size(); ++i)
		{
			MemoryData* memoryPtr = bestCombination[i + staticTensors.size()];
			*dynamicTensors[i]->address = memoryPtr->address + memoryPtr->dynamicSize;
			memoryPtr->dynamicSize += dynamicTensors[i]->size * largestN;
		}

		// clean up
		for (auto& tensor : dynamicTensors)
			delete tensor;
		for (auto& tensor : staticTensors)
			delete tensor;

		dynamicTensors.clear();
		staticTensors.clear();
	}

	void PrintMemory() const
	{
		for (auto& memoryPtr : availableMemory)
		{
			for (int i = 0; i < memoryPtr->size; ++i)
				printf("%1.0f ", memoryPtr->address[i]);
			printf("\n\n");
		}
	}
};

int main()
{
	GpuMemoryManager gpuMemoryManager;

	size_t batches;
	float* staticArr1, * staticArr2, * dynamicArr1, * dynamicArr2;
	size_t sSize1 = 10;
	size_t sSize2 = 120;
	size_t dCoef1 = 3;
	size_t dCoef2 = 5;

	//gpuMemoryManager.ManageStatic(&staticArr1, sSize1);
	gpuMemoryManager.ManageStatic(&staticArr2, sSize2);
	/*gpuMemoryManager.ManageDynamic(&dynamicArr1, dCoef1);
	gpuMemoryManager.ManageDynamic(&dynamicArr2, dCoef2);*/

	gpuMemoryManager.Allocate(batches);
	printf("batches: %zu\n\n", batches);

	/*for (int i = 0; i < sSize1; ++i)
		staticArr1[i] = i;*/
	for (int i = 0; i < sSize2; ++i)
		staticArr2[i] = i;
	/*for (int i = 0; i < dCoef1 * batches; ++i)
		dynamicArr1[i] = i;
	for (int i = 0; i < dCoef2 * batches; ++i)
		dynamicArr2[i] = i;*/

	gpuMemoryManager.PrintMemory();

	return 0;
}