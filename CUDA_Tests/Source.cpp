#include "iostream"
#include "vector"

class Test
{
public:
	struct TestStructParam
	{
		float* matrix;

		TestStructParam()
		{
			matrix = new float[10];
		}
	};
	
	struct TestStructCompute
	{
		TestStructParam* param;
		float* matrix;

		TestStructCompute(TestStructParam* param)
		{
			this->param = param;
			matrix = new float[10];
		}
	};

	TestStructParam param;
	std::vector<TestStructCompute> computes;

	Test()
	{
	}

	~Test()
	{
		delete[] param.matrix;
		for (auto& compute : computes)
		{
			delete[] compute.matrix;
		}
	}
	
	void AddCompute()
	{
		computes.emplace_back(TestStructCompute(&param));
	}
};

int main()
{
	Test test;
	test.AddCompute();
	
	return 0;
}