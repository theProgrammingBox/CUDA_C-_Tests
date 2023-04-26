#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

class Example : public olc::PixelGameEngine
{
public:
	uint32_t seed;
	uint32_t offset;

	Example()
	{
		sAppName = "Example";
	}

	bool OnUserCreate() override
	{
		seed = std::chrono::system_clock::now().time_since_epoch().count();
		offset = std::chrono::system_clock::now().time_since_epoch().count();
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		uint32_t idx = offset;
		for (int x = 0; x < ScreenWidth(); ++x)
			for (int y = 0; y < ScreenHeight(); ++y)
			{
				uint8_t* idx2 = (uint8_t*)&idx;
				uint8_t* offset2 = (uint8_t*)&offset;
				uint8_t* seed2 = (uint8_t*)&seed;

				uint8_t arr[4];
				arr[0] = (idx2[1] * offset2[3] ^ seed2[2] ^ 0xA8);
				arr[1] = (idx2[3] ^ offset2[1] * seed2[0] ^ 0x83);
				arr[2] = (idx2[0] * offset2[2] ^ seed2[3] ^ 0x59);
				arr[3] = (idx2[2] ^ offset2[0] * seed2[1] ^ 0x63);

				arr[0] ^= arr[1] * arr[2] >> 3;
				arr[1] ^= arr[3] * arr[0] >> 3;
				arr[2] ^= arr[0] * arr[3] >> 3;
				arr[3] ^= arr[2] * arr[1] >> 3;

				Draw(x, y, olc::PixelF(arr[0], arr[1], arr[2]));
				++idx;
			}
		seed += 0xA8835963;
		offset += 0x8B7A1B65;
		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(1536, 768, 1, 1))
		demo.Start();
	return 0;
}