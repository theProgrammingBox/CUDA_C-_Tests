#include <stdio.h>	// printf
#include <vector>	// std::vector

/*
seperate working vectors from struct
*/

struct GpuMemoryManager
{
	struct MemoryData
	{
		float* address;
		size_t size;
		size_t staticSize;
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

	std::vector<MemoryData*> MemFrags;
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
			MemFrags.emplace_back(frag);
		}

		if (MemFrags.size() <= 0)
		{
			fprintf(stderr, "Memory size is <= 0\n");
			exit(EXIT_FAILURE);
		}
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
		{
			if (tensor->size > fragSize)
			{
				fprintf(stderr, "Static tensor size is larger than total memory size\n");
				exit(EXIT_FAILURE);
			}
			fragSize -= tensor->size;
		}
		for (auto& tensor : dynamicTensors)
			dynamicTensorSize += tensor->size;

		for (auto& frag : MemFrags)
			frag->ratio = (float)frag->size / fragSize;
		for (auto& tensor : staticTensors)
			tensor->ratio = (float)tensor->size / fragSize;
		for (auto& tensor : dynamicTensors)
			tensor->ratio = (float)tensor->size / dynamicTensorSize;

		//Print();

		float bestScore = FLT_MAX;
		std::vector<MemoryData*> bestCombination(staticTensors.size() + dynamicTensors.size());
		allocateStatic(0, bestScore, bestCombination);

		for (uint32_t i = 0; i < staticTensors.size(); ++i)
		{
			bestCombination[i]->ratio -= staticTensors[i]->ratio;
			bestCombination[i]->staticSize += staticTensors[i]->size;
		}

		for (uint32_t i = 0; i < dynamicTensors.size(); ++i)
		{
			bestCombination[i + staticTensors.size()]->ratio -= dynamicTensors[i]->ratio;
			bestCombination[i + staticTensors.size()]->dynamicSize += dynamicTensors[i]->size;
		}

		size_t largestN = 0;
		float smallestRatio = 1;
		for (auto& frag : MemFrags)
		{
			if (frag->ratio <= smallestRatio)
			{
				smallestRatio = frag->ratio;
				if (frag->dynamicSize > 0)
					largestN = (frag->size - frag->staticSize) / frag->dynamicSize;
			}
		}
		printf("largest N: %d\n", largestN);

		for (auto& frag : MemFrags)
			printf("size: %zu, static size: %zu, dynamic size: %zu, leftover: %zu\n", frag->size, frag->staticSize, frag->dynamicSize, frag->size - frag->staticSize - frag->dynamicSize * largestN);
		printf("\n");
	}

	void allocateStatic(uint32_t tensorIdx, float& bestScore, std::vector<MemoryData*>& bestCombination)
	{
		if (tensorIdx == staticTensors.size())
			allocateDynamic(0, bestScore, bestCombination);
		else
			for (MemoryData* frag : MemFrags)
				if (frag->ratio >= staticTensors[tensorIdx]->ratio)
				{
					staticTensors[tensorIdx]->frag = frag;
					float score = frag->ratio;
					frag->ratio -= staticTensors[tensorIdx]->ratio;
					allocateStatic(tensorIdx + 1, bestScore, bestCombination);
					frag->ratio = score;
				}
	}

	void allocateDynamic(uint32_t tensorIdx, float& bestScore, std::vector<MemoryData*>& bestCombination)
	{
		if (tensorIdx == dynamicTensors.size())
		{
			float score = 0;
			for (MemoryData* frag : MemFrags)
				score += abs(frag->ratio);

			if (score < bestScore)
			{
				bestScore = score;
				for (int i = 0; i < staticTensors.size(); ++i)
					bestCombination[i] = staticTensors[i]->frag;
				for (int i = 0; i < dynamicTensors.size(); ++i)
					bestCombination[i + staticTensors.size()] = dynamicTensors[i]->frag;
			}
		}
		else
			for (MemoryData* frag : MemFrags)
			{
				dynamicTensors[tensorIdx]->frag = frag;
				float score = frag->ratio;
				frag->ratio -= dynamicTensors[tensorIdx]->ratio;
				allocateDynamic(tensorIdx + 1, bestScore, bestCombination);
				frag->ratio = score;
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