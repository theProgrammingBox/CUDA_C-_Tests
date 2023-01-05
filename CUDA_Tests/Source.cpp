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

struct xorwow32
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
};

xorwow32 random(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

int main()
{
	const uint32_t BOARD_SIZE = 3;
	const uint32_t STATE_DIM = BOARD_SIZE * BOARD_SIZE;
	const uint32_t ACTION_DIM = 4;
	const uint32_t MAX_EPISODES = 4;

	class agentAttributes
	{
	public:
		uint32_t x;
		uint32_t y;
		float endState;
		bool isAlive;
		uint32_t score;

		agentAttributes()
		{
			x = random() % BOARD_SIZE;
			y = random() % BOARD_SIZE;
			isAlive = true;
			score = 0;
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
			//float* statesGPU;

			Moment(uint32_t agentsPresent)
			{
				numAgents = agentsPresent;
				states = new float[STATE_DIM * agentsPresent];
				actions = new float[ACTION_DIM * agentsPresent];
				agentReferences = new agentAttributes * [agentsPresent];
				memset(states, 0, sizeof(float) * STATE_DIM * agentsPresent);
				//cudaMalloc(&statesGPU, sizeof(float) * STATE_DIM * agentsPresent);
			}

			Moment(Moment&& other) noexcept
			{
				numAgents = other.numAgents;
				states = other.states;
				actions = other.actions;
				agentReferences = other.agentReferences;
				//statesGPU = other.statesGPU;
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
				//cudaFree(moment.statesGPU);
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

	const uint32_t AGENTS = 8;

	for (uint32_t i = AGENTS; i--;)
		agents.push_back(agentAttributes());

	uint32_t numAlive = AGENTS;
	uint32_t episode = MAX_EPISODES;
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
			(*agentReferences)->score += (*agentReferences)->isAlive;
		}

		history.addMoment(std::move(moment));
	} while (numAlive && --episode);

	// print history
	for (uint32_t i = history.numMoments(); i--;)
	{
		cout << "Moment " << i << "\n";
		History::Moment& moment = history.history[i];
		agentAttributes** agentReferences = moment.agentReferences;
		float* state = moment.states;
		for (uint32_t j = moment.numAgents; j--; agentReferences++, state += STATE_DIM)
		{
			cout << "Agent\n";
			for (uint32_t k = BOARD_SIZE; k--;)
			{
				for (uint32_t l = BOARD_SIZE; l--;)
					cout << state[l + k * BOARD_SIZE] << " ";
				cout << "\n";
			}
			cout << "\n";
		}
	}

	//print agents
	for (uint32_t i = AGENTS; i--;)
		cout << "Agent " << i << " score: " << agents[i].score << "\n";

	return 0;
}