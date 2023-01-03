#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

int main()
{
	typedef float* agentTarget;
	typedef agentTarget* agentTargetReference;
	typedef agentTargetReference* agentTargetReferenceMatrix;
	
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandHandle;
	curandCreateGenerator(&curandHandle, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandHandle, 1234ULL);

	const uint32_t TARGET_DIM = 10;
	agentTarget winGPU;
	agentTarget loseGPU;
	cudaMalloc(&winGPU, TARGET_DIM * sizeof(float));
	cudaMalloc(&loseGPU, TARGET_DIM * sizeof(float));

	curandGenerateUniform(curandHandle, winGPU, TARGET_DIM);
	curandGenerateUniform(curandHandle, loseGPU, TARGET_DIM);

	const uint32_t AGENTS = 10;

	vector<agentTargetReference> agentTargetIDs;
	for (uint32_t i = 0; i < AGENTS; i++)
	{
		agentTargetReference agentID = new agentTarget;
		agentTargetIDs.push_back(agentID);
	}

	vector<agentTargetReferenceMatrix> history;
	agentTargetReferenceMatrix historyArr = new agentTargetReference[AGENTS];
	for (uint32_t i = 0; i < AGENTS; i++)
	{
		historyArr[i] = agentTargetIDs[i];
	}
	history.push_back(historyArr);

	for (uint32_t i = 0; i < AGENTS; i++)
	{
		*agentTargetIDs[i] = i & 2 ? winGPU : loseGPU;
	}

	// print
	agentTarget state = new float[TARGET_DIM];
	for (uint32_t i = 0; i < history.size(); i++)
	{
		for (uint32_t j = 0; j < AGENTS; j++)
		{
			cudaMemcpy(state, *history[i][j], TARGET_DIM * sizeof(float), cudaMemcpyDeviceToHost);
			for (uint32_t k = 0; k < TARGET_DIM; k++)
			{
				cout << state[k] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}

	return 0;
}