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

	float const color = seed * 0.00000000023283064365386962890625f;
	return olc::PixelF(color, color, color);
}

class Example : public olc::PixelGameEngine
{
public:
	uint32_t seed;
	uint32_t offset;

	void render()
	{
		for (int x = 0; x < ScreenWidth(); x++)
			for (int y = 0; y < ScreenHeight(); y++)
			{
				uint32_t idx = y * ScreenWidth() + x;
				Draw(x, y, hash(idx, seed, offset));
			}
	}

	bool OnUserCreate() override
	{
		seed = 0;
		offset = 0;

		render();

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetKey(olc::Key::SPACE).bPressed)
		{
			seed++;
			offset += 3;

			render();
		}
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