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

		void Compute(TestStructParam* param, float num, uint32_t idx)
		{
			this->param = param;
			for (int i = 0; i < 10; i++)
				matrix[i] = param->matrix[i];
			matrix[idx] = num;
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
	
	void AddCompute()
	{
		TestStructCompute compute;
		computes.emplace_back(&compute);
		//computes.back().Compute(&param, 10.0f, 0);
		compute.Compute(&param, 10.0f, 0);

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
	test.AddCompute();
	
	return 0;
}