#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

olc::Pixel hash(uint32_t idx, uint32_t seed, uint32_t offset)
{
	seed ^= idx;
	seed *= 0xBAC57D37;
	seed ^= seed >> 15;
	seed ^= offset;
	seed *= 0x24F66AC9;
	seed ^= seed >> 17;

	const float color1 = (seed & 0xFF) * 0.00390625f;
	const float color2 = (seed & 0xFF00) * 0.00001525878f;
	const float color3 = (seed & 0xFF0000) * 0.0000000596046448f;
	const float color4 = (seed & 0xFF000000) * 0.00000000023283064365f;

	return olc::PixelF(color1, color2, color3, color4);
}

class Example : public olc::PixelGameEngine
{
public:
	uint32_t seed;
	uint32_t offset;

	void render()
	{
		for (uint32_t x = 0; x < ScreenWidth(); x++)
			for (uint32_t y = 0; y < ScreenHeight(); y++)
				Draw(x, y, hash(y * ScreenWidth() + x, seed, offset));
	}

	bool OnUserCreate() override
	{
		seed = 0;
		offset = 0;

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		seed++;
		offset += 3;
		render();

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