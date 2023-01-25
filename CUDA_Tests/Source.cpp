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
using std::chrono::microseconds;
using std::cout;
using std::vector;
using std::sort;
using std::ceil;
using std::exp;
using std::ofstream;

class Random
{
public:
	Random(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
	}

	static uint32_t MakeSeed(uint32_t seed = 0)	// make seed from time and seed
	{
		uint32_t result = seed;
		result = Hash((uint8_t*)&result, sizeof(result), nanosecond());
		result = Hash((uint8_t*)&result, sizeof(result), microsecond());
		return result;
	}

	void Seed(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
	}

	uint32_t Ruint32()	// XORSHIFT128+
	{
		uint64_t a = state[0];
		uint64_t b = state[1];
		state[0] = b;
		a ^= a << 23;
		state[1] = a ^ b ^ (a >> 18) ^ (b >> 5);
		return uint32_t((state[1] + b) >> 16);
	}

	float Rfloat(float min = 0, float max = 1) { return min + (max - min) * Ruint32() * 2.3283064371e-10; }

	static uint32_t Hash(const uint8_t* key, size_t len, uint32_t seed = 0)	// MurmurHash3
	{
		uint32_t h = seed;
		uint32_t k;
		for (size_t i = len >> 2; i; i--) {
			memcpy(&k, key, sizeof(uint32_t));
			key += sizeof(uint32_t);
			h ^= murmur_32_scramble(k);
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}
		k = 0;
		for (size_t i = len & 3; i; i--) {
			k <<= 8;
			k |= key[i - 1];
		}
		h ^= murmur_32_scramble(k);
		h ^= len;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}

private:
	uint64_t state[2];

	static uint32_t murmur_32_scramble(uint32_t k) {
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		return k;
	}

	static uint32_t nanosecond() { return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
	static uint32_t microsecond() { return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
};

namespace GlobalVars
{
	Random random(Random::MakeSeed(0));
}

class Player
{
public:
	uint32_t move()
	{
		return GlobalVars::random.Ruint32() % 9;
	}
};

class TicTacToe
{
public:
	bool player;
	float board[9];
	Player players[2];
	uint32_t runningState;
	uint32_t moveCount;

	void Run()
	{
		player = GlobalVars::random.Ruint32() & 1;
		memset(board, 0, sizeof(board));
		runningState = 0;
		moveCount = 0;

		while (runningState == 0)
		{
			player = !player;
			uint32_t action = players[player].move();
			if (board[action] == 0)
			{
				board[action] = (player << 1) - 1.0f;
				moveCount++;

				for (uint32_t i = 0; i < 3; i++)
				{
					runningState |= board[i] && board[i] == board[i + 3] && board[i] == board[i + 6];
				}
				for (uint32_t i = 0; i < 9; i += 3)
				{
					runningState |= board[i] && board[i] == board[i + 1] && board[i] == board[i + 2];
				}
				runningState |= board[0] && board[0] == board[4] && board[0] == board[8];
				runningState |= board[2] && board[2] == board[4] && board[2] == board[6];
				runningState |= (moveCount == 9) << 1;
			}
		}

		if (runningState & 1)
			cout << "Winner: " << (player << 1) - 1.0f << "\n";
		else if (runningState & 2)
			cout << "Draw\n";

		for (uint32_t i = 0; i < 3; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				cout << board[i * 3 + j] << " ";
			}
			cout << "\n";
		}
	}
};

int main()
{
	TicTacToe game;
	game.Run();

	return 0;
}