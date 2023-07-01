#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

class Example : public olc::PixelGameEngine
{
public:

	Example()
	{
		sAppName = "Example";
	}

	bool OnUserCreate() override
	{
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		float mousex = GetMouseX() * 2.0f / ScreenWidth();
		uint32_t bits = *(uint32_t*)&mousex;

		// Clear Screen
		Clear(olc::BLACK);

		DrawString(0, 0, std::to_string(mousex));

		// Draw 32 bits, white if bit is set, black if not
		for (int i = 0; i < 32; i++)
			DrawCircle((ScreenWidth() >> 1) + (i - 16) * 8, (ScreenHeight() >> 1), 4, (bits >> i) & 1 ? olc::WHITE : olc::BLACK);

		return true;
	}
};

float randf()
{
	return (float)rand() / (float)RAND_MAX;
}

int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	int sum = 0;
	for (int i = 0; i < 10000000; ++i)
		sum += abs(randf()) > 1;
		//sum += 

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printf("Elapsed time: %f\n", elapsed.count());

	return 0;

	Example demo;
	if (demo.Construct(1000, 500, 1, 1))
		demo.Start();
	return 0;
}