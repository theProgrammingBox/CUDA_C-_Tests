//#include <cublas_v2.h>
//#include <curand.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::cout;
using std::vector;
using std::sort;
using std::ceil;
using std::exp;

static struct xorwow32
{
	uint32_t state[6];

	xorwow32(uint32_t seed) : state{
		seed ^ 123456789,
		seed ^ 362436069,
		seed ^ 521288629,
		seed ^ 88675123,
		seed ^ 5783321,
		seed ^ 6615241 } {}

	uint32_t operator()()
	{
		uint32_t t = state[0] ^ (state[0] >> 2);
		memcpy(state, state + 1, 16);
		state[4] ^= (state[4] << 4) ^ (t ^ (t << 1));
		return (state[5] += 362437) + state[4];
	}

	float operator()(float min, float max)
	{
		return min + (max - min) * operator()() * 2.3283064371e-10;	// 0 & 1 inclusive, 2.3283064365e-10 for exclusive 1
	}
} random(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

const static void cpuGenerateUniform(float* matrix, uint32_t size, float min, float max)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = random(min, max);
}

const static void cpuSoftmax(float* inputMatrix, float* outputMatrix, uint32_t size)
{
	float sum = 0;
	for (uint32_t counter = size; counter--;)
	{
		outputMatrix[counter] = exp(inputMatrix[counter]);
		sum += outputMatrix[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		outputMatrix[counter] *= sum;
}

const static void cpuSoftmaxGradient(float* outputMatrix, float* gradient, uint32_t* sample, float* resultMatrix, uint32_t size)
{
	float sampleValue = outputMatrix[*sample];
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = sampleValue * *gradient * ((counter == *sample) - outputMatrix[counter]);
}

int main() {
	constexpr uint32_t AGENTS = 128;
	constexpr uint32_t ACTIONS = 2;
	constexpr uint32_t ITERATIONS = 10000;
	constexpr float LEARNING_RATE = 0.1f;
	constexpr float TOP_PERCENT = 0.2f;

	// Prisoner's Dilemma, score is time in prison
	float score[ACTIONS * ACTIONS] = {
		2, 3,	// (Snitch1 & Snitch2) | (Silent1 & Snitch2)
		1, 0	// (Snitch1 & Silent2) | (Silent1 & Silent2)
	};

	struct Agent
	{
		float bias[ACTIONS];			// the bias state to be used in the softmax
		float probabilities[ACTIONS];	// the result of the softmax
		uint32_t sample;				// the sampled action from the softmax
		float actionGradient;			// whether the action was good or bad
		float score;					// the score of the action
		float gradient[ACTIONS];		// the gradient of the action

		Agent() { memset(bias, 0, sizeof(bias)); }	// set initial bias state to 0 for equal probability
	};

	vector<Agent> agents(AGENTS);	// the agents
	float randomNum;				// random number used to sample from the probability distribution
	
	uint32_t iteration = ITERATIONS;
	while (iteration--)
	{

		for (Agent& agent : agents)
		{
			// calculate the probability distribution
			cpuSoftmax(agent.bias, agent.probabilities, ACTIONS);

			// sample the action
			randomNum = random(0, 1);
			for (uint32_t counter = ACTIONS; counter--;)
			{
				randomNum -= agent.probabilities[counter];
				if (randomNum <= 0)
				{
					agent.sample = counter;
					break;
				}
			}
		}

		// face each other
		for (uint32_t counter = 0; counter < AGENTS; counter += 2)
		{
			uint32_t sample1 = agents[counter].sample;
			uint32_t sample2 = agents[counter + 1].sample;
			agents[counter].score = score[sample1 + sample2 * ACTIONS];
			agents[counter + 1].score = score[sample2 + sample1 * ACTIONS];
		}

		// sort the agents by least time in prison
		sort(agents.begin(), agents.end(), [](const Agent& a, const Agent& b) { return a.score < b.score; });
		
		// set the top agents to have a positive gradient and the bottom agents to have a negative gradient
		for (uint32_t counter = 0; counter < AGENTS; counter++)
			agents[counter].actionGradient = (counter < ceil(AGENTS* TOP_PERCENT)) ? 1 : -1;

		// calculate the gradient for each agent
		for (Agent& agent : agents)
			cpuSoftmaxGradient(agent.probabilities, &agent.actionGradient, &agent.sample, agent.gradient, ACTIONS);
		
		// update the bias state
		for (Agent& agent : agents)
			for (uint32_t counter = ACTIONS; counter--;)
				agent.bias[counter] += LEARNING_RATE * agent.gradient[counter];
	}

	// print the final probability distribution
	for (Agent& agent : agents)
	{
		cpuSoftmax(agent.bias, agent.probabilities, ACTIONS);
		cout << agent.probabilities[0] << " " << agent.probabilities[1] << "\n";
	}

	return 0;
}
