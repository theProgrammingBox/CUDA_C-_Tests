#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

class Example : public olc::PixelGameEngine
{
public:
	uint64_t seed2;
	
	Example()
	{
		sAppName = "Example";
	}
	
	bool OnUserCreate() override
	{
		seed2 = 0;
		rng();
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		seed2++;
		rng();
		return true;
	}

	void rng()
	{
		uint64_t seed1 = 0;
		for (int x = 0; x < ScreenWidth(); x++)
			for (int y = 0; y < ScreenHeight(); y++)
			{
				uint32_t h = seed1 * 0xcc9e2d51;
				h ^= (h << 15) | (h >> 17);
				h *= 0x1b873593;
				h ^= (h << 13) | (h >> 19);
				h ^= seed2 * 0xcc9e2d51;
				h ^= (h << 15) | (h >> 17);
				h *= 0x1b873593;
				h ^= (h << 13) | (h >> 19);
				
				seed1++;
				Draw(x, y, olc::PixelF(h & 0xff, h >> 8 & 0xff, h >> 16 & 0xff));
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