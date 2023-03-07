#include "iostream"
#include "vector"

class Test
{
public:
	struct TestStructParam
	{
		float matrix[10];

		TestStructParam()
		{
			for (int i = 0; i < 10; i++)
				matrix[i] = i;
		}
	};
	
	struct TestStructCompute
	{
		TestStructParam* param;
		float matrix[10];

		TestStructCompute()
		{
		}

		uint32_t Compute(TestStructParam* param, float num, uint32_t idx)
		{
			this->param = param;
			for (int i = 0; i < 10; i++)
				matrix[i] = param->matrix[i];
			matrix[idx] = num;

			return matrix[idx];
		}
	};

	TestStructParam param;
	std::vector<TestStructCompute*> computes;

	Test()
	{
	}

	~Test()
	{
	}
	
	uint32_t AddCompute(float num, uint32_t idx)
	{
		TestStructCompute compute;
		computes.emplace_back(&compute);
		return compute.Compute(&param, num, idx);
	}

	void Print()
	{
		for (auto& compute : computes)
		{
			for (int i = 0; i < 10; i++)
				printf("%f ", compute->matrix[i]);
			printf("\n");
		}
	}
};

int main()
{
	Test test;
	printf("%d", test.AddCompute(50.0f, 5));
	test.Print();
	
	return 0;
}