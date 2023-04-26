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
				uint32_t hash = (idx + offset ^ seed) * 0xAA69E974;
				hash = (hash >> 13 ^ hash) * 0x8B7A1B65;
				Draw(x, y, olc::PixelF(hash & 0xff, (hash >> 8) & 0xff, (hash >> 16) & 0xff));
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