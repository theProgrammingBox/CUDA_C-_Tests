#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

uint32_t Lehmer32(uint64_t seed, uint64_t subsequence = 0)
{
	uint64_t tmp = (seed + 0xe120fc15) * 0x4a39b70d;
	tmp = (tmp >> 32 ^ tmp) * 0x12fad5c9;
	return tmp >> 32 ^ tmp;
}

class Example : public olc::PixelGameEngine
{
public:
	Example()
	{
		sAppName = "Example";
	}
	
	bool OnUserCreate() override
	{
		float ratio = 0;
		for (int x = 0; x < ScreenWidth(); x++)
			for (int y = 0; y < ScreenHeight(); y++)
			{
				bool bPixel = Lehmer32(x * ScreenHeight() + y) & 3;
				ratio += bPixel;
				Draw(x, y, bPixel ? olc::BLACK : olc::WHITE);
			}
		printf("Ratio: %f\n", ratio / (ScreenWidth() * ScreenHeight()));
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(1440, 810, 1, 1))
		demo.Start();
	return 0;
}