#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

// Override base class with your custom functionality
class Example : public olc::PixelGameEngine
{
public:
	const float discount = 0.99f;
	static const int samples = 200;
	float rewards[samples];
	int idx;

	Example()
	{
		sAppName = "Example";
	}

	bool OnUserCreate() override
	{
		memset(rewards, 0, sizeof(rewards));
		idx = 0;
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		// if up, add 1 reward to front, if down, add -1 reward to front
		if (GetKey(olc::Key::UP).bPressed)
		{
			rewards[idx] = 1;
		}
		else if (GetKey(olc::Key::DOWN).bPressed)
		{
			rewards[idx] = -1;
		}
		else
		{
			rewards[idx] = 0;
		}

		//cycle the arr
		idx++;
		idx *= idx >= samples;

		// calculate discount reward
		float discount_reward = 0;
		for (int i = samples; i--;)
		{
			idx--;
			idx += (idx < 0) * samples;
			discount_reward += rewards[idx] + discount * discount_reward;
			Draw(i, 120 - discount_reward * 100, olc::Pixel(255, 255, 255));
			printf("%f\n", discount_reward);
		}

		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(256, 240, 4, 4))
		demo.Start();
	return 0;
}