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

const static void cpuSoftmaxGradient(float* outputMatrix, float* gradient, uint32_t* sample, float* resultMatrix, uint32_t size)
{
	float sampleValue = outputMatrix[*sample];
	for (uint32_t counter = size; counter--;)
		resultMatrix[counter] = sampleValue * *gradient * ((counter == *sample) - outputMatrix[counter]);
}

int main() {
	constexpr uint32_t N = 2;
	constexpr uint32_t ITERATIONS = 100000;
	constexpr float LEARNING_RATE = 0.1f;

	// Prisoner's Dilemma, score is time in prison
	float score[N * N] = {
		5, 10,	// (Snitch1 & Snitch2) | (Silent1 & Snitch2)
		0, 1	// (Snitch1 & Silent2) | (Silent1 & Silent2)
	};

	enum Personality {
		MALICIOUS = 0,		// always tries to force the other player's time in prison to be greater than their own
		LOVER = 1,			// always tries to force the other player's time in prison to be less than their own
		COOPERATIVE = 2,	// always tries to minimize their own time in prison
		SUISIDAL = 3,		// always tries to maximize their own time in prison
	};
	
	uint32_t sample1 = 0;	// player 1's sampled index from the probability distribution
	uint32_t sample2 = 0;	// player 2's sampled index from the probability distribution
	float probabilityGrad1;	// whether player 1's sampled action should be increased or decreased
	float probabilityGrad2;	// whether player 2's sampled action should be increased or decreased
	float randNum;			// random number used to sample from the probability distribution
	float score1;			// player 1's score
	float score2;			// player 2's score
	
	float bias[N];
	float result[N];
	float gradient1[N];
	float gradient2[N];
	
	for (uint32_t personality = 4; personality--;)
	{
		// Initialize biases so the probability distribution is uniform
		memset(bias, 0, N * sizeof(float));

		int iter = ITERATIONS;
		while (iter--)
		{
			// calculate probabilities from bias
			cpuSoftmax(bias, result, N);

			// Sample from the distribution
			randNum = random(0, 1);
			for (int i = 0; i < N; i++)
			{
				randNum -= result[i];
				if (randNum <= 0)
				{
					sample1 = i;
					break;
				}
			}

			randNum = random(0, 1);
			for (int i = 0; i < N; i++)
			{
				randNum -= result[i];
				if (randNum <= 0)
				{
					sample2 = i;
					break;
				}
			}

			// Calculate the score
			score1 = score[sample2 * N + sample1];
			score2 = score[sample1 * N + sample2];

			// should the chances of the sampled move be increased or decreased?

			switch (personality)
			{
			case MALICIOUS:
				probabilityGrad1 = (score2 > score1) * LEARNING_RATE;
				probabilityGrad2 = (score1 > score2) * LEARNING_RATE;
				break;
			case LOVER:
				probabilityGrad1 = (score2 < score1) * LEARNING_RATE;
				probabilityGrad2 = (score1 < score2) * LEARNING_RATE;
				break;
			case COOPERATIVE:
				probabilityGrad1 = (score1 > 0) * LEARNING_RATE;
				probabilityGrad2 = (score2 > 0) * LEARNING_RATE;
				break;
			case SUISIDAL:
				probabilityGrad1 = (score1 < 10) * LEARNING_RATE;
				probabilityGrad2 = (score2 < 10) * LEARNING_RATE;
				break;
			}

			// calculate gradient of the loss function
			cpuSoftmaxGradient(result, &probabilityGrad1, &sample1, gradient1, N);
			cpuSoftmaxGradient(result, &probabilityGrad2, &sample2, gradient2, N);

			// update bias
			for (int i = 0; i < N; i++)
				bias[i] += gradient1[i] + gradient2[i];
		}

		cout << "Result:\n";
		cout << "If both personalities are ";
		switch (personality)
		{
		case MALICIOUS:
			cout << "MALICIOUS";
			break;
		case LOVER:
			cout << "LOVER";
			break;
		case COOPERATIVE:
			cout << "COOPERATIVE";
			break;
		case SUISIDAL:
			cout << "SUISIDAL";
			break;
		}
		cout << ", the optimal strategy is to Snitch " << result[0] * 100 << "% of the time and Silent " << result[1] * 100 << "% of the time.\n\n";

		/*cout << "Bias: ";
		for (int i = 0; i < N; i++)
			cout << bias[i] << " ";
		cout << "\n";*/
	}

	return 0;
}
