//#include <cublas_v2.h>
//#include <curand.h>
#include <iostream>
#include <vector>
#include <chrono>

using std::cout;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

uint32_t seed = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();

uint32_t random()
{
	// xorshift32
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

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
			x = random() % BOARD_SIZE;
			y = random() % BOARD_SIZE;
			isAlive = true;
		}
	};

	vector<agentAttributes> agents;

	class History
	{
	public:
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
				agentReferences = new agentAttributes * [agentsPresent];
				memset(states, 0, sizeof(float) * STATE_DIM * agentsPresent);
			}
			
			Moment(Moment&& other) noexcept
			{
				numAgents = other.numAgents;
				states = other.states;
				actions = other.actions;
				agentReferences = other.agentReferences;
			}
		};

		vector<Moment> history;

		~History()
		{
			for (auto& moment : history)
			{
				delete[] moment.states;
				delete[] moment.actions;
				delete[] moment.agentReferences;
			}
		}

		void addMoment(Moment&& moment)
		{
			history.push_back(std::move(moment));
		}

		uint32_t numMoments()
		{
			return history.size();
		}
	};

	History history;

	const uint32_t AGENTS = 2;

	for (uint32_t i = AGENTS; i--;)
		agents.push_back(agentAttributes());

	uint32_t numAlive = AGENTS;
	do
	{
		History::Moment moment(numAlive);
		agentAttributes** agentReferences = moment.agentReferences;
		
		for (uint32_t i = AGENTS; i--;)
			if (agents[i].isAlive)
				*agentReferences++ = &agents[i];
		
		agentReferences = moment.agentReferences;
		float* state = moment.states;
		for (uint32_t i = moment.numAgents; i--; agentReferences++, state += STATE_DIM)
			state[(*agentReferences)->x + (*agentReferences)->y * BOARD_SIZE] = 1;

		agentReferences = moment.agentReferences;
		for (uint32_t i = moment.numAgents; i--; agentReferences++)
		{
			uint32_t move = random() % 4;
			switch (move)
			{
			case 0:
				if ((*agentReferences)->x > 0)
					(*agentReferences)->x--;
				else
				{
					(*agentReferences)->isAlive = false;
					numAlive--;
				}
				break;
			case 1:
				if ((*agentReferences)->x < BOARD_SIZE - 1)
					(*agentReferences)->x++;
				else
				{
					(*agentReferences)->isAlive = false;
					numAlive--;
				}
				break;
			case 2:
				if ((*agentReferences)->y > 0)
					(*agentReferences)->y--;
				else
				{
					(*agentReferences)->isAlive = false;
					numAlive--;
				}
				break;
			case 3:
				if ((*agentReferences)->y < BOARD_SIZE - 1)
					(*agentReferences)->y++;
				else
				{
					(*agentReferences)->isAlive = false;
					numAlive--;
				}
				break;
			}
		}
		
		history.addMoment(std::move(moment));
	} while (numAlive);

	cout << "History size: " << history.numMoments() << "\n";
	for (uint32_t i = history.numMoments(); i--;)
	{
		cout << "Moment " << i << "\n";
		History::Moment& moment = history.history[i];
		for (uint32_t j = moment.numAgents; j--;)
		{
			cout << "Agent " << j << "\n";
			for (uint32_t k = BOARD_SIZE; k--;)
			{
				for (uint32_t l = BOARD_SIZE; l--;)
					cout << moment.states[j * STATE_DIM + k + l * BOARD_SIZE] << " ";
				cout << "\n";
			}
			cout << "\n";
		}
	}

	return 0;
}