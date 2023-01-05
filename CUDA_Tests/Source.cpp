//#include <cublas_v2.h>
//#include <curand.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

using std::cout;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::sort;

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

static xorwow32 random(duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

class Environment
{
public:
	void Run()
	{
		// create agents
		for (uint32_t i = AGENTS; i--;)
			agents.push_back(agentAttributes());

		// start a game
		uint32_t numAlive = AGENTS;
		uint32_t episode = MAX_EPISODES;
		do
		{
			// add the number of agents alive to the current moment
			History::Moment moment(numAlive);

			agentReferences = moment.agentReferences;
			agent = agents.data();
			for (uint32_t i = AGENTS; i--; agent++)
				if (agent->isAlive)
					*agentReferences++ = agent;

			// update their boards
			agentReferences = moment.agentReferences;
			state = moment.states;
			for (uint32_t i = moment.numAgents; i--; agentReferences++, state += STATE_DIM)
				state[(*agentReferences)->x + (*agentReferences)->y * BOARD_SIZE] = 1;

			// make a random moves as a plave holder, use matmul later on
			agentReferences = moment.agentReferences;
			action = moment.actions;
			for (uint32_t i = moment.numAgents; i--; agentReferences++, action++)
			{
				*action = random() & 3;
				switch (*action)
				{
				case 0:
					(*agentReferences)->isAlive = (*agentReferences)->x > 0;
					(*agentReferences)->x -= (*agentReferences)->isAlive;
					break;
				case 1:
					(*agentReferences)->isAlive = (*agentReferences)->x < BOARD_SIZE - 1;
					(*agentReferences)->x += (*agentReferences)->isAlive;
					break;
				case 2:
					(*agentReferences)->isAlive = (*agentReferences)->y > 0;
					(*agentReferences)->y -= (*agentReferences)->isAlive;
					break;
				case 3:
					(*agentReferences)->isAlive = (*agentReferences)->y < BOARD_SIZE - 1;
					(*agentReferences)->y += (*agentReferences)->isAlive;
					break;
				}
				numAlive -= !(*agentReferences)->isAlive;
				(*agentReferences)->score += (*agentReferences)->isAlive;
			}

			// add the moment to the history
			history.addMoment(std::move(moment));
		} while (numAlive && --episode);

		// sort agents by score, then if they are alive
		sort(agents.begin(), agents.end(), [](const agentAttributes& a, const agentAttributes& b) { return a.score > b.score; });

		// set the endstate based on TOP_AGENTS
		agent = agents.data();
		for (uint32_t i = TOP_AGENTS; i--; agent++)
			agent->endState = 1;

		//print agents
		agent = agents.data();
		for (uint32_t i = AGENTS; i--; agent++)
			cout << "Agent " << i << " Score: " << agent->score << " End State: " << agent->endState << "\n";
		cout << "\n";

		// print history
		moments = history.history.data();
		for (uint32_t i = history.numMoments(); i--; moments++)
		{
			agentReferences = moments->agentReferences;
			state = moments->states;
			action = moments->actions;
			cout << "Moment " << i << "\n";
			for (uint32_t j = moments->numAgents; j--; agentReferences++, state += STATE_DIM, action++)
			{
				cout << "Agent State:\n";
				for (uint32_t k = 0; k < BOARD_SIZE; k++)
				{
					for (uint32_t l = 0; l < BOARD_SIZE; l++)
						cout << state[k + l * BOARD_SIZE] << " ";
					cout << "\n";
				}
				
				cout << "Agent Next Action: ";
				switch (*action)
				{
				case 0:
					cout << "Left\n";
					break;
				case 1:
					cout << "Right\n";
					break;
				case 2:
					cout << "Up\n";
					break;
				case 3:
					cout << "Down\n";
					break;
				}
				cout << "End State: " << (*agentReferences)->endState << "\n";
				cout << "Score: " << (*agentReferences)->score << "\n\n";
			}
		}
	}

private:
	static constexpr uint32_t BOARD_SIZE = 3;
	static constexpr uint32_t STATE_DIM = BOARD_SIZE * BOARD_SIZE;
	static constexpr uint32_t ACTION_DIM = 4;
	static constexpr uint32_t MAX_EPISODES = 4;
	static constexpr uint32_t AGENTS = 8;
	static constexpr float TOP_PERCENT = 0.4f;
	static constexpr uint32_t TOP_AGENTS = AGENTS * TOP_PERCENT;

	struct agentAttributes
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
			endState = -1;
			score = 0;
		}
	};
	
	// a class to handle the deconstruction of the moments in history
	class History
	{
	public:
		struct Moment
		{
			uint32_t numAgents;
			float* states;
			uint8_t* actions;
			float* error;
			agentAttributes** agentReferences;
			//float* statesGPU;

			Moment(uint32_t agentsPresent)
			{
				numAgents = agentsPresent;
				states = new float[STATE_DIM * agentsPresent];
				actions = new uint8_t[agentsPresent];
				error = new float[ACTION_DIM * agentsPresent];
				agentReferences = new agentAttributes * [agentsPresent];
				memset(states, 0, sizeof(float) * STATE_DIM * agentsPresent);
				//cudaMalloc(&statesGPU, sizeof(float) * STATE_DIM * agentsPresent);
			}

			Moment(Moment&& other) noexcept
			{
				numAgents = other.numAgents;
				states = other.states;
				actions = other.actions;
				error = other.error;
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
				delete[] moment.error;
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

	//			TEMP VARS				//
	agentAttributes** agentReferences;
	float* state;
	uint8_t* action;
	History::Moment* moments;
	agentAttributes* agent;
	//////////////////////////////////////
	
	vector<agentAttributes> agents;
	History history;
};

int main()
{
	Environment env;
	env.Run();

	return 0;
}