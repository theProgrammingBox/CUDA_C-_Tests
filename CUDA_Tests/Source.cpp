//#include <cublas_v2.h>
//#include <curand.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::cout;
using std::vector;
using std::sort;
using std::ceil;
using std::exp;
using std::ofstream;

/*
IMPORTANT LESSONS
1. Low learning rate allows stable convergence
2. Massive batch size allows stable convergence
3. In the case of Rock Paper Scissors, keeping the top 10% of agents is a lot better than keeping the top 90% of agents(see if this applies to other games)
4. In the case of Rock Paper Scissors, having more agents leads to results closer to the nash equilibrium. 4 was better then 2 agents, 64 was around the same to 4 agents
5. Interestingly, if Rock vs Rock gives 10 instead of 0, then a few agents will converge to just playing Rock, but the rest will converge to Nash equilibrium. Only a portion of the agents can converge to Rock
(I think this is because once a few agents converge to just playing Rock, then the rest of the Nash equilibrium agents will prevent more agents from converging to just playing Rock)
(Those few agents got lucky and became rich while the other agents are busy keeping each other in check)
(It seems that the number of agents that are chosen to survive plays a huge part in the number of agents that converge to playing Rock)
(It may be because the top few are perfecting what they are doing since they survived, while the rest are still exploring since they lost and don't exactly know what to do)
(To prevent this, the shuffling	algorithms can be changed so that I allows local behaviors, but splits and combines the population to prevent local behaviors from becoming too strong)
(splitting to encourage diversity, combining to prevent local behaviors from becoming too strong)

Whats Next:
1. Visualizing the results
2. Top agent teachers in combination of exploration for the rest of the agents
(Teachers have less influence over exploration)
(Actually, this can lead to winner's bias, idk)
3. Choosing Teamates/enemies based on their actions
4. Tic Tac Toe
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
		uint32_t t = *state ^ (*state >> 2);
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

const static void cpuSoftmaxGradient(float* outputMatrix, bool* isSurvivor, uint32_t* action, float* resultMatrix, uint32_t size, float alpha)
{
	int agentGradient = (*isSurvivor << 1) - 1;
	float sampledProbability = outputMatrix[*action];
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = alpha * resultMatrix[counter] + agentGradient * outputMatrix[counter] * ((counter == *action) - sampledProbability);
}

int main() {
	constexpr uint32_t AGENTS = 2;		// Should be an even number due to my pvp algorithm
	constexpr uint32_t BATCHES = 32;
	constexpr uint32_t ACTIONS = 3;		//2 for prisoner's dilemma, 3 for rock paper scissors
	constexpr uint32_t ITERATIONS = 10000;
	constexpr float LEARNING_RATE = 0.001f;
	constexpr float TOP_PERCENT = 0.2f;
	constexpr uint32_t TOP_AGENTS = AGENTS * TOP_PERCENT;
	constexpr float gradientScalar = LEARNING_RATE / BATCHES;
	
	// Rock Paper Scissors
	float score[ACTIONS * ACTIONS] = {
		0, 1, -1,
		-1, 0, 1,
		1, -1, 0
	};/**/

	/*// Prisoner's Dilemma, score is time in prison
	float score[ACTIONS * ACTIONS] = {
		2, 3,	// (Snitch1 & Snitch2) | (Silent1 & Snitch2)
		0, 1	// (Snitch1 & Silent2) | (Silent1 & Silent2)
	};*/

	struct Agent
	{
		float bias[ACTIONS];			// the bias state to be used in the softmax
		float probabilities[ACTIONS];	// the result of the softmax
		uint32_t sample;				// the sampled action from the probabilities
		bool isSurvivor;				// whether the actions were good or bad
		float score;					// the score of the action
		float biasGradient[ACTIONS];	// the biasGradient of the action

		Agent()
		{
			//memset(bias, 0, sizeof(bias));			// set initial bias state to 0 for equal probability
			cpuGenerateUniform(bias, ACTIONS, -1, 1);	// set initial bias state to random values
		}
	};
	
	vector<Agent> agents(AGENTS);	// the agents
	
	ofstream dataFile("data.txt");
	dataFile << ITERATIONS << '\n';
	dataFile << AGENTS << '\n';
	dataFile << ACTIONS << '\n';
	
	uint32_t iteration = ITERATIONS;
	while (iteration--)
	{
		for (Agent& agent : agents)
		{
			// calculate the probabilities of each action
			cpuSoftmax(agent.bias, agent.probabilities, ACTIONS);

			// reset the biasGradient
			memset(agent.biasGradient, 0, sizeof(agent.biasGradient));
		}

		// run the game BATCHES times before updating the agents
		for (uint32_t batch = BATCHES; batch--;)
		{
			// every agent make a move
			for (Agent& agent : agents)
			{
				//reset the score
				agent.score = 0;

				// sample an action
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
				agent.sample = action;
			}

			// randomize the order of the agents for random matchups
			vector<Agent*> matchVector(AGENTS);
			for (uint32_t counter = AGENTS; counter--;)
				matchVector[counter] = &agents[counter];

			for (uint32_t counter = AGENTS; counter--;)
			{
				uint32_t index = random() % AGENTS;
				Agent* temp = matchVector[counter];
				matchVector[counter] = matchVector[index];
				matchVector[index] = temp;
			}

			// pvp
			for (uint32_t counter = 0; counter < AGENTS; counter += 2)
			{
				uint32_t sample1 = matchVector[counter]->sample;
				uint32_t sample2 = matchVector[counter + 1]->sample;
				matchVector[counter]->score += score[sample1 + sample2 * ACTIONS];
				matchVector[counter + 1]->score += score[sample2 + sample1 * ACTIONS];
			}

			// sort the agents
			vector<Agent*> sortedAgents(AGENTS);
			for (uint32_t counter = AGENTS; counter--;)
				sortedAgents[counter] = &agents[counter];

			// sort by highest score
			sort(sortedAgents.begin(), sortedAgents.end(), [](const Agent* a, const Agent* b) { return a->score > b->score; });

			/*// sort by lowest prison time
			sort(sortedAgents.begin(), sortedAgents.end(), [](const Agent* a, const Agent* b) { return a->score < b->score; });*/

			// calculate and apply the biasGradient for each agent
			uint32_t counter = 0;
			for (Agent* agent : sortedAgents)
			{
				// see if the agent is in the top percentile
				agent->isSurvivor = counter++ <= TOP_AGENTS;

				// calculate the biasGradient
				for (uint32_t batch = BATCHES; batch--;)
					cpuSoftmaxGradient(agent->probabilities, &agent->isSurvivor, &agent->sample, agent->biasGradient, ACTIONS, 1);
			}
		}

		for (Agent& agent : agents)
		{
			// apply the biasGradient
			for (uint32_t action = ACTIONS; action--;)
				agent.bias[action] += agent.biasGradient[action] * gradientScalar;

			// save the probabilities of each agent
			for (uint32_t counter = 0; counter < ACTIONS; counter++)
				dataFile << agent.probabilities[counter] << ' ';
			dataFile << '\n';
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
		cout << "Is Agent a Survivor?: " << agent.isSurvivor << '\n';
		
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
			cout << agent.biasGradient[counter] << " ";
		cout << "\n\n";
	}

	dataFile.close();

	return 0;
}