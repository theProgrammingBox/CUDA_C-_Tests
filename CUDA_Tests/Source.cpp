//#include <cublas_v2.h>
//#include <curand.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::cout;
using std::vector;
using std::sort;

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

const static void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

const static void cpuGenerateUniform(float* matrix, uint32_t size, float min, float max)
{
	for (uint32_t i = size; i--;)
		matrix[i] = random(min, max);
}

class Environment
{
public:
	Environment()
	{
		InitParams();
	}

	~Environment()
	{
		ClearAgents();
		ClearHistory();
		ClearParams();
	}

	void Run()
	{
		while (true)
		{
			AddNewAgents(NUM_AGENTS);
			ForwardPropagate();
			KeepTopAgents(TOP_PERCENT);
			BackPropagate();
			ApplyGradients();
			ClearAgents();
			ClearHistory();
		}
	}

private:
	static constexpr uint32_t NUM_AGENTS = 64;
	static constexpr uint32_t HIDDEN_SIZE = 8;
	static constexpr uint32_t BOARD_SIZE = 3;
	static constexpr uint32_t INPUT_SIZE = BOARD_SIZE * BOARD_SIZE;
	static constexpr uint32_t OUTPUT_SIZE = 5;
	static constexpr uint32_t MAX_MOMENTS = 100;
	static constexpr uint32_t INPUT_WIDTH = HIDDEN_SIZE + INPUT_SIZE;
	static constexpr uint32_t OUTPUT_WIDTH = HIDDEN_SIZE + OUTPUT_SIZE;
	static constexpr float TOP_PERCENT = 0.4;

	float* initialState;			// each new agent's initial state, stored in GPU memory
	float* weights;					// weights of the network, stored in GPU memory
	float* initialStateGradient;	// gradient of the initial state, stored in GPU memory
	float* weightsGradient;			// gradient of the weights, stored in GPU memory
	
	static constexpr float ONE = 1.0f;
	static constexpr float ZERO = 0.0f;
	
	struct Agent
	{
		uint32_t px;			// player x
		uint32_t py;			// player y
		uint32_t gx;			// goal x
		uint32_t gy;			// goal y
		float* hiddenState;		// pointer to the persistent memory of the agent, points to GPU memory
		bool isAlive;			// is agent alive rn, if not, don't compute its action
		bool endState;			// bool that holds survivor status for backprop
	};

	struct Moment
	{
		uint32_t agentsAlive;			// number of agents alive in this moment
		Agent** agentPointers;			// array of pointers to each agent alive at this moment
		float** hiddenStatePointers;	// array of pointers to each agent's hidden state, points to GPU memory
		float* inputs;					// array of each agent's joined hidden state and environment input, stored in GPU memory
		float* outputs;					// array of each agent's joined new hidden state and action probabilities, stored in GPU memory
		uint32_t* actions;				// array of each agent's action represented as an index


		Moment(uint32_t agentsAlive) : 
			agentsAlive(agentsAlive),
			agentPointers(new Agent* [agentsAlive]),
			hiddenStatePointers(new float* [agentsAlive]),
			inputs(new float[INPUT_WIDTH * agentsAlive]),
			outputs(new float[OUTPUT_WIDTH * agentsAlive]),
			actions(new uint32_t[agentsAlive]) {}

		~Moment()
		{
			delete[] agentPointers;
			delete[] hiddenStatePointers;
			delete[] inputs;
			delete[] outputs;
			delete[] actions;
		}
	};

	vector<Agent*> agentPointers;	// vector of pointers to each agent
	vector<Moment> history;			// vector of moments

	uint32_t AgentsAlive()
	{
		uint32_t agentsAlive = 0;
		for (Agent* agent : agentPointers)
			agentsAlive += agent->isAlive;
		return agentsAlive;
	}

	void RandomizeParams()
	{
		cpuGenerateUniform(weights, INPUT_WIDTH * OUTPUT_WIDTH, -1.0f, 1.0f);
		cpuGenerateUniform(initialState, HIDDEN_SIZE, -1.0f, 1.0f);
	}

	void InitParams()
	{
		weights = new float[INPUT_WIDTH * OUTPUT_WIDTH];
		initialState = new float[HIDDEN_SIZE];
		weightsGradient = new float[INPUT_WIDTH * OUTPUT_WIDTH];
		initialStateGradient = new float[HIDDEN_SIZE];
		RandomizeParams();
		memset(weightsGradient, 0, INPUT_WIDTH * OUTPUT_WIDTH * sizeof(float));
		memset(initialStateGradient, 0, HIDDEN_SIZE * sizeof(float));
	}

	void RandomizeAgentPosition(Agent* a)	// environment func
	{
		a->px = random() % BOARD_SIZE;
		a->py = random() % BOARD_SIZE;
	}

	void RandomizeAgentGoal(Agent* a)	// environment func
	{
		do
		{
			a->gx = random() % BOARD_SIZE;
			a->gy = random() % BOARD_SIZE;
		} while (a->gx == a->px && a->gy == a->py);
	}

	void AddNewAgents(uint32_t numAgents)	// idk func
	{
		for (uint32_t counter = numAgents; counter--;)
		{
			Agent* a = new Agent;
			RandomizeAgentPosition(a);
			RandomizeAgentGoal(a);
			a->hiddenState = initialState;
			a->isAlive = true;
			a->endState = false;
			agentPointers.push_back(a);
		}
	}

	void AddAgentsAliveToMoment(Moment* moment)	// idk func
	{
		Agent** agentPointersIterator = moment->agentPointers;
		float** hiddenStatePointersIterator = moment->hiddenStatePointers;
		for (Agent* agent : agentPointers)
		{
			if (agent->isAlive)
			{
				*agentPointersIterator = agent;
				*hiddenStatePointersIterator = agent->hiddenState;
				agentPointersIterator++;
				hiddenStatePointersIterator++;
			}
		}
	}

	void GetInput(Agent* agent, float* input)	// environment func
	{
		memset(input, 0, INPUT_SIZE * sizeof(float));
		input[agent->px + agent->py * BOARD_SIZE] = -1;
		input[agent->gx + agent->gy * BOARD_SIZE + INPUT_SIZE] = 1;
	}

	void InitMomentInputs(Moment* moment)	// trainer func
	{
		float* matrixIterator = moment->inputs;
		float* shiftedMatrixIterator = matrixIterator + HIDDEN_SIZE;
		Agent** agentPointersIterator = moment->agentPointers;
		float** hiddenStatePointersIterator = moment->hiddenStatePointers;
		for (uint32_t counter = moment->agentsAlive; counter--;)
		{
			memcpy(matrixIterator, *hiddenStatePointersIterator, HIDDEN_SIZE * sizeof(float));
			GetInput(*agentPointersIterator, shiftedMatrixIterator);
			hiddenStatePointersIterator++;
			agentPointersIterator++;
			matrixIterator += INPUT_WIDTH;
			shiftedMatrixIterator += INPUT_WIDTH;
		}
	}

	void ForwardPropagateMoment(Moment* moment)	// trainer func
	{
		cpuSgemmStridedBatched(
			false, false,
			OUTPUT_WIDTH, moment->agentsAlive, INPUT_WIDTH,
			&ONE,
			weights, OUTPUT_WIDTH, ZERO,
			moment->inputs, INPUT_WIDTH, ZERO,
			&ZERO,
			moment->outputs, OUTPUT_WIDTH, ZERO,
			0);

		//activation function placeholder
	}

	void GetAction(float* outputs, uint32_t* action)	// trainer func
	{
		float max = outputs[0];
		for (uint32_t i = 1; i < OUTPUT_SIZE; i++)
			if (outputs[i] > max) max = outputs[i];
		float sum = 0;
		for (uint32_t i = OUTPUT_SIZE; i--;)
		{
			outputs[i] = exp(outputs[i] - max);
			sum += outputs[i];
		}

		float r = random(0, sum);
		for (uint32_t i = 0; i < OUTPUT_SIZE; i++)
		{
			r -= outputs[i];
			if (r <= 0)
			{
				*action = i;
				return;
			}
		}
	}

	void Act(Agent* agent, uint32_t* action)	// environment func
	{
		switch (*action)
		{
		case 0:
			agent->isAlive = agent->px > 0;
			agent->px -= agent->isAlive;
			break;
		case 1:
			agent->isAlive = agent->px < BOARD_SIZE - 1;
			agent->px += agent->isAlive;
			break;
		case 2:
			agent->isAlive = agent->py > 0;
			agent->py -= agent->isAlive;
			break;
		case 3:
			agent->isAlive = agent->py < BOARD_SIZE - 1;
			agent->py += agent->isAlive;
			break;
		}
	}

	void ActMomentOutputs(Moment* moment)
	{
		float* matrixIterator = moment->outputs;
		float* shiftedMatrixIterator = matrixIterator + HIDDEN_SIZE;
		Agent** agentPointersIterator = moment->agentPointers;
		uint32_t* actionsIterator = moment->actions;
		for (uint32_t counter = moment->agentsAlive; counter--;)
		{
			(*agentPointersIterator)->hiddenState = matrixIterator;
			GetAction(shiftedMatrixIterator, actionsIterator);
			Act(*agentPointersIterator, actionsIterator);
			agentPointersIterator++;
			actionsIterator++;
			matrixIterator += OUTPUT_WIDTH;
			shiftedMatrixIterator += OUTPUT_WIDTH;
		}
	}

	void ForwardPropagate()
	{
		uint32_t numMoments = MAX_MOMENTS;
		uint32_t agentsAlive;
		while ((agentsAlive = AgentsAlive()) && numMoments--)
		{
			cout << "agentsAlive: " << agentsAlive << '\n';
			Moment moment(agentsAlive);

			AddAgentsAliveToMoment(&moment);
			InitMomentInputs(&moment);
			ForwardPropagateMoment(&moment);
			ActMomentOutputs(&moment);
			history.push_back(moment);
		};
	}

	void BackPropagate()
	{
		// placeholder for logic
	}

	void ApplyGradients()
	{
		// placeholder for logic

		memset(weightsGradient, 0, INPUT_WIDTH * OUTPUT_WIDTH * sizeof(float));
		memset(initialStateGradient, 0, HIDDEN_SIZE * sizeof(float));
	}

	void KeepTopAgents(float topPercent)
	{
		sort(agentPointers.begin(), agentPointers.end(), [](Agent* a, Agent* b) { return a->isAlive > b->isAlive; });
		
		for (uint32_t counter = agentPointers.size() * topPercent; counter--;)
		{
			agentPointers[counter]->endState = true;
		}
	}

	void ClearAgents()
	{
		for (Agent* a : agentPointers) delete a;
		agentPointers.clear();
	}

	void ClearHistory()
	{
		for (Moment& m : history) delete& m;
		history.clear();
	}

	void ClearParams()
	{
		delete[] weights;
		delete[] initialState;
		delete[] weightsGradient;
		delete[] initialStateGradient;
	}
};

int main() {
	Environment env;
	env.Run();

	return 0;
}
