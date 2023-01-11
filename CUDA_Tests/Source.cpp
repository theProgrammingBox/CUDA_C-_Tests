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
(Actually, 50% might just be the best for most cases because it ensures that there are equal numbers of positive and negative gradients)
For example, if all agents pick the same action, but only 1% survive, then the negative gradient will overpower the positive gradient pushing for that action.

4. Adding batches when sampling one agent's performance will lead to a more accurate representation of the nash equilibrium.


IMPORTANT LESSONS
1. Low learning rate allows stable convergence
2. Massive batch size allows stable convergence
3. In the case of Rock Paper Scissors, keeping the top 10% of agents is a lot better than keeping the top 90% of agents(see if this applies to other games)
4. In the case of Rock Paper Scissors, having more agents leads to results closer to the nash equilibrium. 4 was better then 2 agents, 64 was around the same to 4 agents
5. Interestingly, if Rock vs Rock gives 10 instead of 0, then a few agents will converge to just playing Rock, but the rest will converge to Nash equilibrium. Only a portion of the agents can converge to Rock
(I think this is because once a few agents converge to just playing Rock, then the rest of the Nash equilibrium agents will prevent more agents from converging to just playing Rock)
(Those few agents got lucky and became rich while the other agents are busy keeping each other in check)
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
	float max = inputMatrix[0];
	for (uint32_t counter = size; counter--;)
		if (inputMatrix[counter] > max)
			max = inputMatrix[counter];
	float sum = 0;
	for (uint32_t counter = size; counter--;)
	{
		outputMatrix[counter] = exp(inputMatrix[counter] - max);
		sum += outputMatrix[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		outputMatrix[counter] *= sum;
}

const static void cpuSoftmaxGradient(float* outputMatrix, bool* isSurvivor, uint32_t* action, float* resultMatrix, uint32_t size, float alpha)
{
	int agentGradient = (*isSurvivor << 1) - 1;
	float sampledProbability = outputMatrix[*action];
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = alpha * resultMatrix[counter] + agentGradient * outputMatrix[counter] * ((counter == *action) - sampledProbability);
}

int main() {
	constexpr uint32_t AGENTS = 16;
	constexpr uint32_t BATCHES = 128;
	constexpr uint32_t ACTIONS = 3;
	constexpr uint32_t ITERATIONS = 10000;
	constexpr float LEARNING_RATE = 1.0f;
	constexpr float TOP_PERCENT = 0.1f;
	constexpr float gradientScalar = 1.0f / (AGENTS * BATCHES);

	/*// Prisoner's Dilemma, score is time in prison
	float score[ACTIONS * ACTIONS] = {
		2, 3,	// (Snitch1 & Snitch2) | (Silent1 & Snitch2)
		0, 1	// (Snitch1 & Silent2) | (Silent1 & Silent2)
	};*/
	
	// Rock Paper Scissors
	float score[ACTIONS * ACTIONS] = {
		0, 1, -1,
		-1, 0, 1,
		1, -1, 0
	};

	struct Agent
	{
		float bias[ACTIONS];			// the bias state to be used in the softmax
		float probabilities[ACTIONS];	// the result of the softmax
		uint32_t sample[BATCHES];		// the sampled action from the softmax
		bool isSurvivor;				// whether the actions were good or bad
		float score;					// the score of the action
		float gradient[ACTIONS];		// the gradient of the action

		Agent()
		{
			memset(bias, 0, sizeof(bias));	// set initial bias state to 0 for equal probability
			//cpuGenerateUniform(bias, ACTIONS, -1, 1);	// set initial bias state to random values
		}
	};
	
	vector<Agent> agents(AGENTS);	// the agents
	
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
				float number = random(0.0f, 1.0f);
				uint32_t action = 0;
				while (true)
				{
					number -= agent.probabilities[action];
					if (number < 0)
						break;
					action++;
					action -= (action == ACTIONS) * ACTIONS;
				}
				agent.sample[batch] = action;
			}
		}

		// reset score
		for (uint32_t counter = AGENTS; counter--;)
			agents[counter].score = 0;
		
		for (uint32_t batch = BATCHES; batch--;)
		{
			// randomize the order of the agents
			for (uint32_t counter = AGENTS; counter--;)
			{
				uint32_t index = random() % AGENTS;
				Agent temp = agents[counter];
				agents[counter] = agents[index];
				agents[index] = temp;
			}

			// face each other
			for (uint32_t counter = 0; counter < AGENTS; counter += 2)
			{
				uint32_t sample1 = agents[counter].sample[batch];
				uint32_t sample2 = agents[counter + 1].sample[batch];
				agents[counter].score += score[sample1 + sample2 * ACTIONS];
				agents[counter + 1].score += score[sample2 + sample1 * ACTIONS];
			}
		}

		// sort the agents by largest score
		sort(agents.begin(), agents.end(), [](const Agent& a, const Agent& b) { return a.score > b.score; });
		
		// set the top agents to survive and the bottom agents to die
		for (uint32_t counter = AGENTS; counter--;)
			agents[counter].isSurvivor = counter < ceil(AGENTS* TOP_PERCENT);
		
		// calculate and apply the gradient for each agent
		for (Agent& agent : agents)
		{
			memset(agent.gradient, 0, sizeof(agent.gradient));
			
			// calculate the gradient
			for (uint32_t batch = BATCHES; batch--;)
				cpuSoftmaxGradient(agent.probabilities, &agent.isSurvivor, &agent.sample[batch], agent.gradient, ACTIONS, 1);
			
			// apply the gradient
			for (uint32_t counter = ACTIONS; counter--;)
				agent.bias[counter] += agent.gradient[counter] * gradientScalar;
		}
	}

	// print the results
	for (Agent& agent : agents)
	{
		cpuSoftmax(agent.bias, agent.probabilities, ACTIONS);
		cout << "Probability distribution: ";
		for (uint32_t counter = 0; counter < ACTIONS; counter++)
			cout << agent.probabilities[counter] << " ";
		cout << '\n';
		
		cout << "Score: " << agent.score << '\n';
		cout << "Action Gradient: " << agent.isSurvivor << '\n';
		
		/*cout << "Actions: ";
		for (uint32_t batch = 0; batch < BATCHES; batch++)
			cout << agent.sample[batch] << " ";
		cout << '\n';*/
		
		cout << "Bias: ";
		for (uint32_t counter = 0; counter < ACTIONS; counter++)
			cout << agent.bias[counter] << " ";
		cout << '\n';
		
		cout << "Gradient: ";
		for (uint32_t counter = 0; counter < ACTIONS; counter++)
			cout << agent.gradient[counter] << " ";
		cout << "\n\n";
	}

	return 0;
}