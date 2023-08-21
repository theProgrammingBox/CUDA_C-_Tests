#include <stdio.h>	// printf
#include <assert.h>	// assert
#include <vector>	// std::vector

struct GpuMemoryManager
{
	struct MemFrag
	{
		float* address;
		size_t size;
		float ratio;
		size_t usedSize;
	};

	struct TensorData
	{
		float** address;
		size_t size;
		float ratio;
		size_t idx;
	};

	std::vector<MemFrag> MemFrags;
	std::vector<TensorData> dynamicTensor;
	std::vector<TensorData> staticTensor;

	GpuMemoryManager()
	{
		printf("Initializing GPU memory manager...\n");

		for (int i = 0; i < 2; ++i)
		{
			MemFrag frag;
			frag.address = nullptr;
			frag.size = i * 1024 + 1024;
			frag.usedSize = 0;
			MemFrags.emplace_back(frag);
		}

		assert(MemFrags.size() > 0);
	}

	void ManageDynamic(float** tensorPtr, size_t size)
	{
		TensorData tensorData;
		tensorData.address = tensorPtr;
		tensorData.size = size;
		tensorData.idx = 0;
		dynamicTensor.emplace_back(tensorData);
	}

	void ManageStatic(float** tensorPtr, size_t size)
	{
		TensorData tensorData;
		tensorData.address = tensorPtr;
		tensorData.size = size;
		tensorData.idx = 0;
		staticTensor.emplace_back(tensorData);
	}

	void Print()
	{
		for (auto& frag : MemFrags)
			printf("Frag size: %d, address: %p, ratio: %f\n", frag.size, frag.address, frag.ratio);
		printf("\n");

		for (auto& tensor : dynamicTensor)
			printf("Dynamic tensor size: %d, address: %p, ratio: %f\n", tensor.size, tensor.address, tensor.ratio);
		printf("\n");

		for (auto& tensor : staticTensor)
			printf("Static tensor size: %d, address: %p, ratio: %f\n", tensor.size, tensor.address, tensor.ratio);
		printf("\n");
	}

	void Allocate()
	{
		/*qsort(MemFrags.data(), MemFrags.size(), sizeof(MemFrag), [](const void* a, const void* b) -> int
			{
			return ((MemFrag*)b)->size - ((MemFrag*)a)->size;
		});

		qsort(dynamicTensor.data(), dynamicTensor.size(), sizeof(TensorData), [](const void* a, const void* b) -> int
			{
			return ((TensorData*)b)->size - ((TensorData*)a)->size;
		});

		qsort(staticTensor.data(), staticTensor.size(), sizeof(TensorData), [](const void* a, const void* b) -> int
			{
			return ((TensorData*)b)->size - ((TensorData*)a)->size;
		});*/

		for (auto& tensor : staticTensor)
		{
			MemFrags[0].usedSize += tensor.size;
		}

		for (auto& frag : MemFrags)
			printf("Frag used size: %d\n", frag.usedSize);

		/*size_t fragSize = 0;
		for (auto& frag : MemFrags)
			fragSize += frag.size;

		size_t staticSize = 0;
		for (auto& tensor : staticTensor)
			staticSize += tensor.size;

		size_t dynamicTensorSize = 0;
		for (auto& tensor : dynamicTensor)
			dynamicTensorSize += tensor.size;
		for (auto& tensor : dynamicTensor)
			tensor.ratio = (float)tensor.size / dynamicTensorSize;

		for (auto& frag : MemFrags)
			frag.ratio = (float)frag.size / fragSize;
		for (auto& tensor : staticTensor)
			tensor.ratio = (float)tensor.size / fragSize;*/
	}
};

int main()
{
	GpuMemoryManager gpuMemoryManager;

	float* staticArr1 = nullptr;
	float* staticArr2 = nullptr;
	float* dynamicArr1 = nullptr;
	float* dynamicArr2 = nullptr;

	gpuMemoryManager.ManageStatic(&staticArr1, 1024);
	gpuMemoryManager.ManageStatic(&staticArr2, 2000);
	gpuMemoryManager.ManageDynamic(&dynamicArr1, 3);
	gpuMemoryManager.ManageDynamic(&dynamicArr2, 4);

	gpuMemoryManager.Allocate();
	gpuMemoryManager.Print();

	return 0;
}