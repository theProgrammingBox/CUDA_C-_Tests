#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

typedef unsigned int u32;

void xorshift(u32& x)
{
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
}

olc::Pixel hash(u32 idx, u32 seed, u32 offset)
{
	seed ^= idx;
	seed ^= seed << 13;
	seed ^= offset;
	seed *= 0xBAC57D37;
	seed ^= seed >> 17;
	seed *= 0x24F66AC9;

	const float color1 = (seed & 0xFF) * 0.00390625f;
	const float color2 = (seed & 0xFF00) * 0.00001525878f;
	const float color3 = (seed & 0xFF0000) * 0.0000000596046448f;
	const float color4 = (seed & 0xFF000000) * 0.00000000023283064365f;

	return olc::PixelF(color1, color2, color3, color4);
}

class Example : public olc::PixelGameEngine
{
public:
	u32 seed;
	u32 offset;

	void render()
	{
		for (u32 x = 0; x < ScreenWidth(); x++)
			for (u32 y = 0; y < ScreenHeight(); y++)
				Draw(x, y, hash(y * ScreenWidth() + x, seed, offset));
	}

	bool OnUserCreate() override
	{
		seed = time(NULL);
		xorshift(seed);
		xorshift(seed);

		offset = seed;
		xorshift(offset);

		printf("Seed: %u\n", seed);
		printf("Offset: %u\n", offset);

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