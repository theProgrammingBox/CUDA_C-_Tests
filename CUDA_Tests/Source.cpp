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

/*
What I learned from this:
1. Only having one agent facing itself will lead to an convergence based on its own actions.
Based on the kind of environment, the final probability may not accurately represent the nash equilibrium of a diverse population.

2. Having more agents will lead to a more accurate representation of the nash equilibrium or a close approximation.
This is because the agents will be able to evolve from each other's actions, which allows the agents to learn from different "personalities"
The algorithm of match making is very important because it will determine who the agents will face and evolve from.
This can often lead to local behavior, like a fish evolving in a pond next to an ocean.

3. The number of agents you choose to "survive" is also very important.
This factor is ultimately a constraint that can effect the average nash equilibrium of the agents.
This may be because although many agents have the same ranking, the number of agents that survive is limited, leading to a the same actions leading to different results.
(We need a way to not punish ties because it is causing a problem in the nash equilibrium)

4. Adding batches when sampling one agent's performance will lead to a more accurate representation of the nash equilibrium.
*/

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
	
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = *gradient * outputMatrix[counter] * ((counter == *sample) - outputMatrix[*sample]);
	
	/*float rest = *gradient < 0 ? 1 : -1;
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = rest + ((counter == *sample) * *gradient * 2);*/
}

int main() {
	constexpr uint32_t AGENTS = 32;
	constexpr uint32_t BATCHES = 4;
	constexpr uint32_t ACTIONS = 2;
	constexpr uint32_t ITERATIONS = 1000;
	constexpr float LEARNING_RATE = 0.1f;
	constexpr float TOP_PERCENT = 0.6f;

	// Prisoner's Dilemma, score is time in prison
	float score[ACTIONS * ACTIONS] = {
		20, 0,	// (Snitch1 & Snitch2) | (Silent1 & Snitch2)
		0, 0	// (Snitch1 & Silent2) | (Silent1 & Silent2)
	};

	struct Agent
	{
		float bias[ACTIONS];				// the bias state to be used in the softmax
		float probabilities[ACTIONS];		// the result of the softmax
		uint32_t sample[BATCHES];			// the sampled action from the softmax
		float actionGradient;				// whether the actions were good or bad
		float score;						// the score of the action
		float gradient[ACTIONS * BATCHES];	// the gradient of the action

		Agent() { memset(bias, 0, sizeof(bias)); }	// set initial bias state to 0 for equal probability
	};

	vector<Agent> agents(AGENTS);	// the agents
	float randomNumber;				// random number used to sample from the probability distribution
	
	uint32_t iteration = ITERATIONS;
	while (iteration--)
	{
		for (Agent& agent : agents)
		{
			// calculate the probability distribution
			cpuSoftmax(agent.bias, agent.probabilities, ACTIONS);

			// sample the action
			for (uint32_t batch = BATCHES; batch--;)
			{
				randomNumber = random(0.0f, 1.0f);
				for (uint32_t action = ACTIONS; action--;)
				{
					randomNumber -= agent.probabilities[action];
					if (randomNumber <= 0)
					{
						agent.sample[batch] = action;
						break;
					}
				}
			}
		}

		// randomize the order of the agents
		for (uint32_t counter = AGENTS; counter--;)
		{
			uint32_t index = random() % AGENTS;
			uint32_t index2 = random() % AGENTS;
			Agent temp = agents[index];
			agents[index] = agents[index2];
			agents[index2] = temp;
		}
		
		// face each other
		for (uint32_t counter = 0; counter < AGENTS; counter += 2)
		{
			agents[counter].score = 0;
			agents[counter + 1].score = 0;
			for (uint32_t batch = BATCHES; batch--;)
			{
				uint32_t sample1 = agents[counter].sample[batch];
				uint32_t sample2 = agents[counter + 1].sample[batch];
				agents[counter].score += score[sample1 + sample2 * ACTIONS];
				agents[counter + 1].score += score[sample2 + sample1 * ACTIONS];
			}
		}

		// sort the agents by least time in prison
		sort(agents.begin(), agents.end(), [](const Agent& a, const Agent& b) { return a.score < b.score; });
		
		// set the top agents to have a positive gradient and the bottom agents to have a negative gradient
		for (uint32_t counter = AGENTS; counter--;)
			agents[counter].actionGradient = (counter < ceil(AGENTS * TOP_PERCENT)) ? 1 : -1;

		// calculate and apply the gradient for each agent
		for (Agent& agent : agents)
		{
			for (uint32_t batch = BATCHES; batch--;)
			{
				cpuSoftmaxGradient(agent.probabilities, &agent.actionGradient, &agent.sample[batch], agent.gradient + batch * ACTIONS, ACTIONS);
				for (uint32_t counter = ACTIONS; counter--;)
					agent.bias[counter] += LEARNING_RATE * agent.gradient[batch * ACTIONS + counter];
				// incomplete, average over batches, or not if you want a kind of momentum like effect
			}
		}
	}

	// print the final probability distribution
	for (Agent& agent : agents)
	{
		cpuSoftmax(agent.bias, agent.probabilities, ACTIONS);
		cout << agent.probabilities[0] << " " << agent.probabilities[1] << "\n";/**/
		/*//score
		cout << "Score: " << agent.score << ", ";
		//actions
		cout << "Actions: ";
		for (uint32_t batch = 0; batch < BATCHES; batch++)
			cout << agent.sample[batch] << " ";
		//action gradient
		cout << ", Action Gradient: " << agent.actionGradient << ", ";
		//bias
		cout << "Bias: ";
		for (uint32_t counter = 0; counter < ACTIONS; counter++)
			cout << agent.bias[counter] << " ";
		//gradient
		cout << ", Gradient: ";
		for (uint32_t counter = 0; counter < ACTIONS * BATCHES; counter++)
			cout << agent.gradient[counter] << " ";
		cout << "\n";*/
	}

	return 0;
}