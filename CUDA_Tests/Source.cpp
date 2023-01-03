#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

int main()
{
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandHandle;
	curandCreateGenerator(&curandHandle, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandHandle, 1234ULL);

	const uint32_t N = 10;
	float* winGPU, * loseGPU;
	cudaMalloc(&winGPU, N * sizeof(float));
	cudaMalloc(&loseGPU, N * sizeof(float));
	
	curandGenerateUniform(curandHandle, winGPU, N);
	curandGenerateUniform(curandHandle, loseGPU, N);

	const uint32_t AGENTS = 10;

	vector<float**> agents;
	for (uint32_t i = 0; i < AGENTS; i++)
	{
		float** agent = new float*;
		agents.push_back(agent);
	}
	
	vector<float***> history;
	float*** historyarr = new float** [AGENTS];
	for (uint32_t i = 0; i < AGENTS; i++)
	{
		historyarr[i] = agents[i];
	}
	history.push_back(historyarr);

	for (uint32_t i = 0; i < AGENTS; i++)
	{
		*agents[i] = i & 2 ? winGPU : loseGPU;
	}

	// print
	float* state = new float[N];
	for (uint32_t i = 0; i < history.size(); i++)
	{
		for (uint32_t j = 0; j < AGENTS; j++)
		{
			cudaMemcpy(state, *history[i][j], N * sizeof(float), cudaMemcpyDeviceToHost);
			for (uint32_t k = 0; k < N; k++)
			{
				cout << state[k] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}

	return 0;
}