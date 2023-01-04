//#include <cublas_v2.h>
//#include <curand.h>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

int main()
{
	const uint32_t BOARD_SIZE = 3;
	const uint32_t STATE_DIM = BOARD_SIZE * BOARD_SIZE;
	const uint32_t ACTION_DIM = 4;
	
	class agentAttributes
	{
	public:
		uint32_t x;
		uint32_t y;
		float endState;
		bool isAlive;

		agentAttributes()
		{
			x = rand() % BOARD_SIZE;
			y = rand() % BOARD_SIZE;
			isAlive = true;
		}
	};

	struct Moment
	{
		uint32_t numAgents;
		float* states;
		float* actions;
		agentAttributes** agentReferences;

		Moment(uint32_t agentsPresent)
		{
			numAgents = agentsPresent;
			states = new float[STATE_DIM * agentsPresent];
			actions = new float[ACTION_DIM * agentsPresent];
			agentReferences = new agentAttributes* [agentsPresent];
			memset(states, 0, sizeof(float) * STATE_DIM * agentsPresent);
		}

		~Moment()
		{
			delete[] states;
			delete[] actions;
			delete[] agentReferences;
		}
	};

	vector<agentAttributes> agents;
	vector<Moment> history;

	const uint32_t AGENTS = 2;

	for (uint32_t i = AGENTS; i--;)
		agents.push_back(agentAttributes());
	
	uint32_t numAlive = AGENTS;
	do
	{
		Moment moment(numAlive);
		for (uint32_t i = AGENTS; i--;)
			if (agents[i].isAlive)
				moment.agentReferences[i] = &agents[i];
		
		agentAttributes** agent = moment.agentReferences;
		float* state = moment.states;
		for (uint32_t i = moment.numAgents; i--; agent++, state += STATE_DIM)
		{
			state[(*agent)->x + (*agent)->y * BOARD_SIZE] = 1;
		}

		// mat mul placeholder
		for (uint32_t i = moment.numAgents; i--;)
		{
			agent[i]->x += rand() % 3 - 1;
			agent[i]->y += rand() % 3 - 1;
			if (agent[i]->x >= BOARD_SIZE || agent[i]->y >= BOARD_SIZE)
				agent[i]->isAlive = false;
		}

		history.push_back(moment);
	} while (numAlive);

	return 0;
}