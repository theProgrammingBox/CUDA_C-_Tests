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

	void rng()
	{
		uint64_t subsequence = 0;
		for (int x = 0; x < ScreenWidth(); x++)
			for (int y = 0; y < ScreenHeight(); y++)
			{
				uint64_t tmp = (subsequence ^ 0xe120fc15) * 0x4a39b70d;
				tmp = (tmp >> 32 ^ tmp ^ (seed ^ 0xe120fc15) * 0x4a39b70d) * 0x12fad5c9;
				tmp = (tmp >> 32 ^ tmp) * 0x12fad5c9;
				tmp ^= tmp >> 32;
				subsequence++;
				Draw(x, y, olc::PixelF(tmp & 0x1, tmp & 0x1, tmp & 0x1));
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