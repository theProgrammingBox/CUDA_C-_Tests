#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

typedef uint32_t u32;
typedef int8_t i8;
typedef float f32;

void Lehmer32(u32& x)
{
	x *= 0xBAC57D37;
	x ^= x >> 16;
	x *= 0x24F66AC9;
	x ^= x >> 16;
}

void FourF32Rands(u32 idx, u32 seed1, u32 seed2, f32* fourF32s)
{
	idx ^= seed1;
	Lehmer32(idx);
	idx ^= seed2;

	fourF32s[0] = i8(idx & 0xFF) * 0.0078125f;
	fourF32s[1] = i8(idx >> 8 & 0xFF) * 0.0078125f;
	fourF32s[2] = i8(idx >> 16 & 0xFF) * 0.0078125f;
	fourF32s[3] = i8(idx >> 24) * 0.0078125f;
}

class Example : public olc::PixelGameEngine
{
public:
	u32 seed1;
	u32 seed2;
	f32 fourF32s[4];

	void seed()
	{
		seed1 = time(NULL) ^ 0xE621B963;
		Lehmer32(seed1);
		seed2 = seed1 ^ 0x6053653F;
		Lehmer32(seed2);
	}

	void render()
	{
		for (u32 x = 0; x < ScreenWidth(); x++)
		{
			for (u32 y = 0; y < ScreenHeight(); y++)
			{
				FourF32Rands(y * ScreenWidth() + x, seed1, seed2, fourF32s);
				Draw(x, y, olc::PixelF(fourF32s[0], fourF32s[1], fourF32s[2], fourF32s[3]));
			}
		}
	}

	void refresh()
	{
		Lehmer32(seed1);
		Lehmer32(seed2);
		render();
	}

	bool OnUserCreate() override
	{
		seed();
		refresh();

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetKey(olc::Key::SPACE).bPressed)
			refresh();

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