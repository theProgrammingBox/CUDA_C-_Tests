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
	for (uint32_t counter = size; counter--;)
		matrix[counter] = random(min, max);
}

const static void cpuClippedLinearUnit(float* inputMatrix, float* outputMatrix, size_t size)
{
	float input;
	for (size_t counter = size; counter--;)
	{
		input = inputMatrix[counter] + 1;
		input = (input > 0) * input - 2;
		outputMatrix[counter] = (input < 0) * input + 1;
	}
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

/*
TODO:
1. Debug forward propagate
2. Add back propagate
3. Add apply gradient
4. Seperate Trainer and Enviroment class
5. Transition to GPU memory
*/

class Environment
{
private:
	static constexpr uint32_t NUM_AGENTS = 2;
	static constexpr uint32_t HIDDEN_SIZE = 8;
	static constexpr uint32_t BOARD_SIZE = 3;
	static constexpr uint32_t INPUT_SIZE = BOARD_SIZE * BOARD_SIZE;
	static constexpr uint32_t OUTPUT_SIZE = 5;
	static constexpr uint32_t MAX_MOMENTS = 4;
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
		uint32_t score;			// score
		float* hiddenState;		// pointer to the persistent memory of the agent, points to GPU memory
		bool isAlive;			// is agent alive rn, if not, don't compute its action
		bool endState;			// bool that holds survivor status, set after full episode
	};

	struct Moment
	{
		uint32_t agentsAlive;			// number of agents alive in this moment
		Agent** agentPointers;			// array of pointers to each agent alive at this moment
		float** hiddenStatePointers;	// array of pointers to each agent's hidden state, points to GPU memory
		float* inputs;					// array of each agent's joined hidden state and environment input, stored in GPU memory
		float* outputs;					// array of each agent's joined new hidden state and action probabilities, stored in GPU memory
		float* activations;				// array of each agent's activation, stored in GPU memory
		uint32_t* actions;				// array of each agent's action represented as an index


		Moment(uint32_t agentsAlive) :
			agentsAlive(agentsAlive),
			agentPointers(new Agent* [agentsAlive]),
			hiddenStatePointers(new float* [agentsAlive]),
			inputs(new float[INPUT_WIDTH * agentsAlive]),
			outputs(new float[OUTPUT_WIDTH * agentsAlive]),
			activations(new float[OUTPUT_WIDTH * agentsAlive]),
			actions(new uint32_t[agentsAlive]) {}
	};

	vector<Agent*> agentPointers;	// vector of pointers to each agent
	vector<Moment> history;			// vector of moments

	void RandomizeParams()	// trainer func
	{
		cpuGenerateUniform(weights, INPUT_WIDTH * OUTPUT_WIDTH, -1.0f, 1.0f);
		cpuGenerateUniform(initialState, HIDDEN_SIZE, -1.0f, 1.0f);
	}

	void InitParams()	// trainer func
	{
		weights = new float[INPUT_WIDTH * OUTPUT_WIDTH];
		initialState = new float[HIDDEN_SIZE];
		weightsGradient = new float[INPUT_WIDTH * OUTPUT_WIDTH];
		initialStateGradient = new float[HIDDEN_SIZE];
		RandomizeParams();
		memset(weightsGradient, 0, INPUT_WIDTH * OUTPUT_WIDTH * sizeof(float));
		memset(initialStateGradient, 0, HIDDEN_SIZE * sizeof(float));
	}

	uint32_t AgentsAlive()	// enviroment func
	{
		uint32_t agentsAlive = 0;
		for (Agent* agentPointer : agentPointers)
			agentsAlive += agentPointer->isAlive;
		return agentsAlive;
	}

	bool PlayerOnGoal(Agent* agentPointer)	// environment func
	{
		return agentPointer->px == agentPointer->gx && agentPointer->py == agentPointer->gy;
	}

	void RandomizeAgentPosition(Agent* agentPointer)	// environment func
	{
		agentPointer->px = random() % BOARD_SIZE;
		agentPointer->py = random() % BOARD_SIZE;
	}

	void RandomizeAgentGoal(Agent* agentPointer)	// environment func
	{
		do
		{
			agentPointer->gx = random() % BOARD_SIZE;
			agentPointer->gy = random() % BOARD_SIZE;
		} while (PlayerOnGoal(agentPointer));
	}

	void AddNewAgents(uint32_t numAgents)	// environment func
	{
		for (uint32_t counter = numAgents; counter--;)
		{
			Agent* agentPointer = new Agent;
			RandomizeAgentPosition(agentPointer);
			RandomizeAgentGoal(agentPointer);
			agentPointer->score = 0;
			agentPointer->hiddenState = initialState;
			agentPointer->isAlive = true;
			agentPointer->endState = false;
			agentPointers.push_back(agentPointer);
		}
	}

	void AddAgentsAliveToMoment(Moment* moment)	// environment func
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
		input[agent->gx + agent->gy * BOARD_SIZE] = 1;
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
			1);

		float* matrixIterator = moment->outputs;
		float* shiftedMatrixIterator = matrixIterator + HIDDEN_SIZE;
		float* activationsIterator = moment->activations;
		float* shiftedActivationsIterator = activationsIterator + HIDDEN_SIZE;
		for (uint32_t counter = moment->agentsAlive; counter--;)
		{
			cpuClippedLinearUnit(matrixIterator, activationsIterator, HIDDEN_SIZE);
			cpuSoftmax(shiftedMatrixIterator, shiftedActivationsIterator, OUTPUT_SIZE);
			matrixIterator += OUTPUT_WIDTH;
			shiftedMatrixIterator += OUTPUT_WIDTH;
			activationsIterator += OUTPUT_WIDTH;
			shiftedActivationsIterator += OUTPUT_WIDTH;
		}
	}

	void GetAction(float* outputs, uint32_t* action)	// trainer func
	{
		float r = random(0, 1);
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

	void AgentAct(Agent* agent, uint32_t* action)	// environment func
	{
		switch (*action)
		{
		case 0:
			agent->isAlive = agent->px > 0;
			agent->px -= agent->isAlive;
			//cout << "left\n";
			break;
		case 1:
			agent->isAlive = agent->px < BOARD_SIZE - 1;
			agent->px += agent->isAlive;
			//cout << "right\n";
			break;
		case 2:
			agent->isAlive = agent->py > 0;
			agent->py -= agent->isAlive;
			//cout << "up\n";
			break;
		case 3:
			agent->isAlive = agent->py < BOARD_SIZE - 1;
			agent->py += agent->isAlive;
			//cout << "down\n";
			break;
		case 4:
			//cout << "stay\n";
			break;
		}
	}

	void EnvironmentAct(Agent* agentPointer)	// environment func
	{
		if (PlayerOnGoal(agentPointer))
		{
			agentPointer->score++;
			RandomizeAgentGoal(agentPointer);
		}
	}

	void ActMomentOutputs(Moment* moment)	// idk func, leaning towards environment func
	{
		float* activationsIterator = moment->activations;
		float* shiftedActivationsIterator = activationsIterator + HIDDEN_SIZE;
		Agent** agentPointersIterator = moment->agentPointers;
		uint32_t* actionsIterator = moment->actions;
		for (uint32_t counter = moment->agentsAlive; counter--;)
		{
			(*agentPointersIterator)->hiddenState = activationsIterator;
			GetAction(shiftedActivationsIterator, actionsIterator);
			AgentAct(*agentPointersIterator, actionsIterator);
			EnvironmentAct(*agentPointersIterator);
			activationsIterator += OUTPUT_WIDTH;
			shiftedActivationsIterator += OUTPUT_WIDTH;
			agentPointersIterator++;
			actionsIterator++;
		}
	}

	void ForwardPropagate()	// idk func, leaning towards environment func
	{
		uint32_t numMoments = MAX_MOMENTS;
		uint32_t agentsAlive;
		while ((agentsAlive = AgentsAlive()) && numMoments--)
		{
			Moment moment(agentsAlive);
			AddAgentsAliveToMoment(&moment);
			InitMomentInputs(&moment);
			ForwardPropagateMoment(&moment);
			ActMomentOutputs(&moment);
			history.push_back(moment);
		};
	}

	void KeepTopAgents(float topPercent)	// idk func, leaning towards environment func
	{
		sort(agentPointers.begin(), agentPointers.end(), [](Agent* a, Agent* b) { return a->score > b->score; });
		for (uint32_t counter = ceil(agentPointers.size() * topPercent); counter--;)
			agentPointers[counter]->endState = true;
	}

	void BackPropagate()	//idk func, leaning towards trainer func
	{
		// placeholder for logic
	}

	void ApplyGradients()	// idk func, leaning towards trainer func
	{
		// placeholder for logic

		memset(weightsGradient, 0, INPUT_WIDTH * OUTPUT_WIDTH * sizeof(float));
		memset(initialStateGradient, 0, HIDDEN_SIZE * sizeof(float));
	}

	void ClearAgents()	// idk func, leaning towards environment func
	{
		for (Agent* a : agentPointers) delete a;
		agentPointers.clear();
	}

	void ClearHistory()	// idk func, leaning towards environment func
	{
		for (Moment& m : history)
		{
			delete[] m.agentPointers;
			delete[] m.hiddenStatePointers;
			delete[] m.inputs;
			delete[] m.outputs;
			delete[] m.activations;
			delete[] m.actions;
		}
		history.clear();
	}

	void ClearParams()	// idk func, leaning towards trainer func
	{
		delete[] weights;
		delete[] initialState;
		delete[] weightsGradient;
		delete[] initialStateGradient;
	}
	
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

	void PrintParamsInfo()
	{
		cout << "Weights:\n";
		for (uint32_t i = 0; i < INPUT_WIDTH; i++)
		{
			for (uint32_t j = 0; j < OUTPUT_WIDTH; j++)
				cout << weights[i * OUTPUT_WIDTH + j] << ' ';
			cout << '\n';
		}
		cout << '\n';

		cout << "Initial State:\n";
		for (uint32_t i = 0; i < HIDDEN_SIZE; i++)
			cout << initialState[i] << ' ';
		cout << "\n\n\n";
	}

	void PrintAgentsInfo()
	{
		cout << "There are " << agentPointers.size() << " total agents.\n\n";
		for (Agent* agentPointer : agentPointers)
		{
			cout << "Agent " << agentPointer << "\n";
			cout << "Player Position: " << agentPointer->px << ", " << agentPointer->py << '\n';
			cout << "Goal Position: " << agentPointer->gx << ", " << agentPointer->gy << '\n';
			cout << "Score: " << agentPointer->score << '\n';
			cout << "Hidden State: ";
			for (uint32_t i = 0; i < HIDDEN_SIZE; i++)
				cout << agentPointer->hiddenState[i] << ' ';
			cout << '\n';
			cout << "Agent is " << (agentPointer->isAlive ? "alive" : "dead") << '\n';
			cout << "Agent is " << (agentPointer->endState ? "a survivor" : "not a survivor") << '\n';
			cout << '\n';
		}
		cout << '\n';
	}

	void PrintHistoryInfo()
	{
		cout << "There are " << history.size() << " total moments.\n\n";
		for (Moment& moment : history)
		{
			cout << "There are " << moment.agentsAlive << " agents alive at this moment.\n\n";
			for (uint32_t i = 0; i < moment.agentsAlive; i++)
			{
				cout << "Agent " << moment.agentPointers[i] << "\n";
				cout << "Player Position: " << moment.agentPointers[i]->px << ", " << moment.agentPointers[i]->py << '\n';
				cout << "Goal Position: " << moment.agentPointers[i]->gx << ", " << moment.agentPointers[i]->gy << '\n';
				cout << "Score: " << moment.agentPointers[i]->score << '\n';
				cout << "Hidden State: ";
				for (uint32_t j = 0; j < HIDDEN_SIZE; j++)
					cout << moment.hiddenStatePointers[i][j] << ' ';
				cout << '\n';
				cout << "Agent is " << (moment.agentPointers[i]->isAlive ? "alive" : "dead") << '\n';
				cout << "Agent is " << (moment.agentPointers[i]->endState ? "a survivor" : "not a survivor") << '\n';
				cout << '\n';
			}
			
			enum { DETAILED = true, BASIC = false };
			if (BASIC)
			{
				cout << "Inputs:\n";
				for (uint32_t i = 0; i < moment.agentsAlive; i++)
				{
					for (uint32_t j = 0; j < INPUT_WIDTH; j++)
						cout << moment.inputs[i * INPUT_WIDTH + j] << ' ';
					cout << '\n';
				}
				cout << '\n';
				cout << "Outputs:\n";
				for (uint32_t i = 0; i < moment.agentsAlive; i++)
				{
					for (uint32_t j = 0; j < OUTPUT_WIDTH; j++)
						cout << moment.outputs[i * OUTPUT_WIDTH + j] << ' ';
					cout << '\n';
				}
				cout << '\n';
				cout << "Activations:\n";
				for (uint32_t i = 0; i < moment.agentsAlive; i++)
				{
					for (uint32_t j = 0; j < OUTPUT_WIDTH; j++)
						cout << moment.activations[i * OUTPUT_WIDTH + j] << ' ';
					cout << '\n';
				}
				cout << '\n';
			}
			else
			{
				cout << "Inputs:\n";
				for (uint32_t i = 0; i < moment.agentsAlive; i++)
				{
					for (uint32_t j = 0; j < BOARD_SIZE; j++)
					{
						for (uint32_t k = 0; k < BOARD_SIZE; k++)
							cout << moment.inputs[i * INPUT_WIDTH + HIDDEN_SIZE + j * BOARD_SIZE + k] << ' ';
						cout << '\n';
					}
					cout << '\n';
				}
				cout << '\n';
				cout << "Outputs:\n";
				for (uint32_t i = 0; i < moment.agentsAlive; i++)
				{
					for (uint32_t j = 0; j < OUTPUT_SIZE; j++)
						cout << moment.activations[i * OUTPUT_WIDTH + HIDDEN_SIZE + j] << ' ';
					cout << '\n';
				}
				cout << '\n';
			}
			cout << "Actions:\n";
			for (uint32_t i = 0; i < moment.agentsAlive; i++)
			{
				cout << "Agent " << moment.agentPointers[i] << " moved ";
				switch (moment.actions[i])
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
				case 4:
					cout << "Stay\n";
					break;
				}
			}
			cout << "\n\n";
		}
	}

	void Run()
	{
		PrintParamsInfo();
		/*while (true)
		{*/
			AddNewAgents(NUM_AGENTS);
			ForwardPropagate();
			PrintHistoryInfo();
			KeepTopAgents(TOP_PERCENT);
			PrintAgentsInfo();
			/*BackPropagate();
			ApplyGradients();*/
			ClearAgents();
			ClearHistory();
		/*}*/
	}

};

int main() {
	Environment env;
	env.Run();

	return 0;
}
