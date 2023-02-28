#include <fstream>

void cpuLeakyRelu(float* input, float* output, uint32_t size)
{
	// reflection of relu
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
}

void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	// reflection of relu derivative
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
}

void PrintMatrix(float* arr, uint32_t rows, uint32_t cols, const char* label) {
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

int main()
{
	float* input = new float[100];
	float* output = new float[100];
	float* gradient = new float[100];
	
	for (size_t counter = 100; counter--;)
		input[counter] = (float)counter / 100.0f;

	cpuLeakyRelu(input, output, 100);
	PrintMatrix(input, 10, 10, "Input");

	cpuLeakyReluDerivative(input, gradient, output, 100);
	PrintMatrix(output, 10, 10, "Output");

	return 0;
}