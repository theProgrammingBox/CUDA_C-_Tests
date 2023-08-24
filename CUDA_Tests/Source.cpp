#include <stdio.h>	// printf
#include <vector>	// std::vector

/*
seperate working vectors from struct
*/

void FailIf(bool condition, const char* message)
{
	if (condition)
	{
		fprintf(stderr, "%s\n", message);
		abort();
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
		MemoryData* frag;
	};

	std::vector<MemoryData*> availableMemory;
	std::vector<TensorData*> dynamicTensors;
	std::vector<TensorData*> staticTensors;

	GpuMemoryManager()
	{
		printf("Initializing GPU memory manager...\n");

		for (uint32_t i = 0; i < 2; ++i)
		{
			MemoryData* frag = new MemoryData;
			frag->address = nullptr;
			frag->size = i * 102400 + 10240;
			frag->dynamicSize = 0;
			availableMemory.emplace_back(frag);
		}

		FailIf(availableMemory.size() <= 0, "Memory size is <= 0");
	}

	void ManageDynamic(float** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		dynamicTensors.emplace_back(tensorData);
	}

	void ManageStatic(float** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		staticTensors.emplace_back(tensorData);
	}

	void Print()
	{
		for (auto& frag : availableMemory)
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

		for (auto& frag : availableMemory)
			fragSize += frag->size;
		for (auto& tensor : staticTensors)
		{
			FailIf(tensor->size > fragSize, "Static tensor size is larger than total memory size");
			fragSize -= tensor->size;
		}
		for (auto& tensor : dynamicTensors)
			dynamicTensorSize += tensor->size;

		for (auto& frag : availableMemory)
			frag->ratio = (float)frag->size / fragSize;
		for (auto& tensor : staticTensors)
			tensor->ratio = (float)tensor->size / fragSize;
		for (auto& tensor : dynamicTensors)
			tensor->ratio = (float)tensor->size / dynamicTensorSize;

		size_t largestN = 0;
		float bestScore = FLT_MAX;
		std::vector<MemoryData*> bestCombination(staticTensors.size() + dynamicTensors.size());
		allocateStatic(0, bestScore, bestCombination, largestN);
		printf("largest N: %d\n\n", largestN);

		for (auto& tensor : dynamicTensors)
			delete tensor;
		for (auto& tensor : staticTensors)
			delete tensor;

		dynamicTensors.clear();
		staticTensors.clear();
	}

	void allocateStatic(uint32_t tensorIdx, float& bestScore, std::vector<MemoryData*>& bestCombination, size_t& largestN)
	{
		if (tensorIdx == staticTensors.size())
			allocateDynamic(0, bestScore, bestCombination, largestN);
		else
			for (MemoryData* frag : availableMemory)
				if (frag->size >= staticTensors[tensorIdx]->size)
				{
					staticTensors[tensorIdx]->frag = frag;
					float preRatio = frag->ratio;
					size_t preSize = frag->size;
					frag->ratio -= staticTensors[tensorIdx]->ratio;
					frag->size -= staticTensors[tensorIdx]->size;
					allocateStatic(tensorIdx + 1, bestScore, bestCombination, largestN);
					frag->ratio = preRatio;
					frag->size = preSize;
				}
	}

	void allocateDynamic(uint32_t tensorIdx, float& bestScore, std::vector<MemoryData*>& bestCombination, size_t& largestN)
	{
		if (tensorIdx == dynamicTensors.size())
		{
			float score = 0;
			for (MemoryData* frag : availableMemory)
				score += abs(frag->ratio);

			if (score < bestScore)
			{
				float smallestRatio = 1;
				size_t size, dynamicSize;
				for (auto& frag : availableMemory)
					if (frag->dynamicSize > 0 && frag->ratio < smallestRatio)
					{
						smallestRatio = frag->ratio;
						size = frag->size;
						dynamicSize = frag->dynamicSize;
					}
				largestN = size / dynamicSize;

				bestScore = score;
				for (int i = 0; i < staticTensors.size(); ++i)
					bestCombination[i] = staticTensors[i]->frag;
				for (int i = 0; i < dynamicTensors.size(); ++i)
					bestCombination[i + staticTensors.size()] = dynamicTensors[i]->frag;
			}
		}
		else
			for (MemoryData* frag : availableMemory)
			{
				dynamicTensors[tensorIdx]->frag = frag;
				float preRatio = frag->ratio;
				float preDynamicSize = frag->dynamicSize;
				frag->ratio -= dynamicTensors[tensorIdx]->ratio;
				frag->dynamicSize += dynamicTensors[tensorIdx]->size;
				allocateDynamic(tensorIdx + 1, bestScore, bestCombination, largestN);
				frag->ratio = preRatio;
				frag->dynamicSize = preDynamicSize;
			}
	}
};

int main()
{
	GpuMemoryManager gpuMemoryManager;

	float* staticArr1 = nullptr;
	float* staticArr2 = nullptr;
	float* dynamicArr1 = nullptr;
	float* dynamicArr2 = nullptr;

	gpuMemoryManager.ManageStatic(&staticArr1, 1000);
	gpuMemoryManager.ManageStatic(&staticArr2, 2000);
	gpuMemoryManager.ManageDynamic(&dynamicArr1, 3);
	gpuMemoryManager.ManageDynamic(&dynamicArr2, 4);

	gpuMemoryManager.Allocate();

	return 0;
}