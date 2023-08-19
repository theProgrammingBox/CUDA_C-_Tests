#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

uint32_t XXH_readLE32(const void* memPtr)
{
	const uint8_t* bytePtr = (const uint8_t*)memPtr;
	return bytePtr[0]
		| ((uint32_t)bytePtr[1] << 8)
		| ((uint32_t)bytePtr[2] << 16)
		| ((uint32_t)bytePtr[3] << 24);
}

uint64_t XXH_readLE64(const void* memPtr)
{
	const uint8_t* bytePtr = (const uint8_t*)memPtr;
	return bytePtr[0]
		| ((uint64_t)bytePtr[1] << 8)
		| ((uint64_t)bytePtr[2] << 16)
		| ((uint64_t)bytePtr[3] << 24)
		| ((uint64_t)bytePtr[4] << 32)
		| ((uint64_t)bytePtr[5] << 40)
		| ((uint64_t)bytePtr[6] << 48)
		| ((uint64_t)bytePtr[7] << 56);
}

static uint64_t XXH3_rrmxmx(uint64_t h64, uint64_t len)
{
	h64 ^= _rotl64(h64, 49) ^ _rotl64(h64, 24);
	h64 *= 0x9FB21C651E98DF25ULL;
	h64 ^= (h64 >> 35) + 8;
	h64 *= 0x9FB21C651E98DF25ULL;
	return h64 ^ (h64 >> 28);
}

olc::Pixel hash(const uint8_t* idx, uint64_t seed, const uint8_t* offset)
{
	seed ^= (uint64_t)_byteswap_ulong((uint32_t)seed) << 32;
	uint32_t const input1 = XXH_readLE32(idx);
	uint32_t const input2 = XXH_readLE32(idx + 4);
	uint64_t const bitflip = (0x0E4125884092CA03ULL ^ XXH_readLE64(offset)) - seed;
	uint64_t const input64 = input2 + (((uint64_t)input1) << 32);
	uint64_t const keyed = input64 ^ bitflip;
	uint64_t output = XXH3_rrmxmx(keyed, 8);

	float color = (float)output / (float)UINT64_MAX * 255.0f;
	return olc::PixelF(color, color, color);
}

class Example : public olc::PixelGameEngine
{
public:
	uint64_t seed;
	uint64_t offset;

	void render()
	{
		for (int x = 0; x < ScreenWidth(); x++)
			for (int y = 0; y < ScreenHeight(); y++)
			{
				uint32_t idx = y * ScreenWidth() + x;
				Draw(x, y, hash((uint8_t*)&idx, seed, (uint8_t*)&offset));
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
		/*if (GetKey(olc::Key::SPACE).bPressed)
		{*/
			seed++;
			offset += 3;

			render();
		//}
		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(1920, 1080, 1, 1))
		demo.Start();
	return 0;
}