#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

uint32_t lcg_parkmiller(uint32_t seed)
{
	return seed * 279470273u % 0xfffffffb;
}

class Example : public olc::PixelGameEngine
{
public:
	uint64_t seed;
	
	Example()
	{
		sAppName = "Example";
	}
	
	bool OnUserCreate() override
	{
		seed = 0;
		rng();
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		//seed += ScreenHeight() * ScreenWidth();
		seed++;
		rng();
		return true;
	}

	uint32_t murmur_32_scramble(uint32_t k) {
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		return k;
	}

	void rng()
	{
		uint64_t subsequence = 0;
		for (int x = 0; x < ScreenWidth(); x++)
			for (int y = 0; y < ScreenHeight(); y++)
			{
				uint32_t h = murmur_32_scramble(subsequence);
				h = (h << 13) | (h >> 19);
				h = h * 5 + 0xe6546b64;
				h ^= murmur_32_scramble(seed);
				h ^= h >> 16;
				h *= 0x85ebca6b;
				h ^= h >> 13;
				h *= 0xc2b2ae35;
				h ^= h >> 16;
				
				subsequence++;
				Draw(x, y, olc::PixelF(h & 0x1, h & 0x1, h & 0x1));
			}
	}
};

int main()
{
	Example demo;
	if (demo.Construct(1440, 810, 1, 1))
		demo.Start();
	return 0;
}