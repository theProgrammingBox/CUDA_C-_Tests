//#include <cublas_v2.h>
//#include <curand.h>
#include <chrono>
#include <iostream>
#include <vector>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::cout;
using std::vector;

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

class Environment
{
public:
	static constexpr uint32_t NUM_AGENTS = 64;
	static constexpr uint32_t HIDDEN_SIZE = 8;
	static constexpr uint32_t BOARD_SIZE = 3;
	static constexpr uint32_t INPUT_SIZE = BOARD_SIZE * BOARD_SIZE;
	static constexpr uint32_t OUTPUT_SIZE = 5;
	static constexpr uint32_t MAX_MOMENTS = 100;

	Environment()
	{
		agentsAlive = 0;

		// initialize global parmeters
		weights = new float[(HIDDEN_SIZE + INPUT_SIZE) * (HIDDEN_SIZE + OUTPUT_SIZE)];
		initialState = new float[HIDDEN_SIZE];

		cpuGenerateUniform(weights, (HIDDEN_SIZE + INPUT_SIZE) * (HIDDEN_SIZE + OUTPUT_SIZE), -1.0f, 1.0f);
		cpuGenerateUniform(initialState, HIDDEN_SIZE, -1.0f, 1.0f);
	}

	void Run()
	{
		AddXAgents(NUM_AGENTS);

		uint32_t numAlive = agentPointers.size();
		uint32_t numMoments = MAX_MOMENTS;
		do
		{
			Step();
		} while (numAlive && numMoments--);

		EraseAgents();
	}

private:
	struct Agent
	{
		uint32_t px;			// player x
		uint32_t py;			// player y
		uint32_t gx;			// goal x
		uint32_t gy;			// goal y
		float* hiddenState;	// location of hidden state in  memory
		bool isAlive;			// is agent alive rn, if not, don't bother with it
		bool endState;			// bool that holds survivor status for backprop
	};

	struct Moment
	{
		uint32_t agentsAlive;			// number of agents alive in this Moment
		Agent** agentPointers;			// array of agent pointers
		float** hiddenStatePointers;	// array of ponters to hidden states in  memory
		uint32_t* actions;				// matrix of actions represented as integers

		float* inputs;	// matrix of inputs in  memory, contains hidden state inputs and inputs
		float* outputs;	// matrix of outputs in  memory, contains hidden states outputs and outputs

		Moment(uint32_t agentsAlive) : 
			agentsAlive(agentsAlive),
			agentPointers(new Agent* [agentsAlive]),
			hiddenStatePointers(new float* [agentsAlive]),
			actions(new uint32_t[agentsAlive]),
			inputs(new float[(HIDDEN_SIZE + INPUT_SIZE) * agentsAlive]),
			outputs(new float[(HIDDEN_SIZE + OUTPUT_SIZE) * agentsAlive]) {}

		~Moment()
		{
			delete[] agentPointers;
			delete[] hiddenStatePointers;
			delete[] actions;
			delete[] inputs;
			delete[] outputs;
		}
	};

	float* initialState;	// matrix of initial states in  memory
	float* weights;		// matrix of weights in  memory

	vector<Agent*> agentPointers;	// vector that holds all agents in heap memory
	vector<Moment> history;

	uint32_t agentsAlive;

	//		temp vars		//
	uint32_t counter;
	Agent** agentPointersIterator;
	float** hiddenStatePointersIterator;
	float* matrixIterator;
	float* shiftedMatrixIterator;
	float input[INPUT_SIZE];
	float output[OUTPUT_SIZE];
	//			//			//

	void RandomizeAgentPosition(Agent* a)
	{
		a->px = random() % BOARD_SIZE;
		a->py = random() % BOARD_SIZE;
	}

	void RandomizeAgentGoal(Agent* a)
	{
		do
		{
			a->gx = random() % BOARD_SIZE;
			a->gy = random() % BOARD_SIZE;
		} while (a->gx == a->px && a->gy == a->py);
	}

	void AddXAgents(uint32_t numAgents)
	{
		for (counter = numAgents; counter--;)
		{
			Agent* a = new Agent;
			RandomizeAgentPosition(a);
			RandomizeAgentGoal(a);
			a->hiddenState = initialState;
			a->isAlive = true;
			a->endState = false;
			agentPointers.push_back(a);
			agentsAlive++;
		}
	}

	void KillAgent(Agent* a)
	{
		a->isAlive = false;
		agentsAlive--;
	}

	void EraseAgents()
	{
		for (Agent* a : agentPointers) delete a;
		agentPointers.clear();
		agentsAlive = 0;
	}

	void cpuSgemmStridedBatched(
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

	void cpuGenerateUniform(float* matrix, uint32_t size, float min, float max)
	{
		for (uint32_t i = size; i--;)
			matrix[i] = random(min, max);
	}

	void GetInput(Agent* a, float* input)
	{
		memset(input, 0, INPUT_SIZE * sizeof(float));
		input[a->px + a->py * BOARD_SIZE] = -1;
		input[a->gx + a->gy * BOARD_SIZE + INPUT_SIZE] = 1;
	}

	void Step()
	{
		Moment moment(agentsAlive);
		
		agentPointersIterator = moment.agentPointers;
		hiddenStatePointersIterator = moment.hiddenStatePointers;
		for (Agent* agent : agentPointers)
		{
			if (agent->isAlive)
			{
				*agentPointersIterator++ = agent;
				*hiddenStatePointersIterator++ = agent->hiddenState;
			}
		}

		matrixIterator = moment.inputs;
		shiftedMatrixIterator = matrixIterator + HIDDEN_SIZE;
		agentPointersIterator = moment.agentPointers;
		hiddenStatePointersIterator = moment.hiddenStatePointers;
		for (counter = moment.agentsAlive; counter--; matrixIterator += HIDDEN_SIZE + INPUT_SIZE, shiftedMatrixIterator += HIDDEN_SIZE + INPUT_SIZE)
		{
			memcpy(matrixIterator, *hiddenStatePointersIterator++, HIDDEN_SIZE * sizeof(float));
			GetInput(*agentPointersIterator++, shiftedMatrixIterator);
		}

		//mat mul placeholder
		
		history.push_back(moment);
	}
};

int main() {
	Environment env;
	env.Run();

	return 0;
}
