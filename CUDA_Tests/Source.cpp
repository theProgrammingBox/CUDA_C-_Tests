#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

// Override base class with your custom functionality
class Example : public olc::PixelGameEngine
{
public:
	const float discount = 0.99f;
	static const int samples = 1800;
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
		// clear screen
		Clear(olc::BLACK);

		// if up, add 1 reward to front, if down, add -1 reward to front
		if (GetKey(olc::Key::UP).bHeld)
		{
			rewards[idx] = 1;
		}
		else if (GetKey(olc::Key::DOWN).bHeld)
		{
			rewards[idx] = -1;
		}
		else
		{
			rewards[idx] = 0;
		}

		// calculate discount reward
		float discount_reward = rewards[idx];

		//cycle the arr
		idx--;
		idx += (idx < 0) * samples;

		for (int i = samples; i--;)
		{
			idx++;
			idx *= idx < samples;

			discount_reward = rewards[idx] + discount * discount_reward;
			// red if negative, green if positive, color gradient
			// red 255 - 255 - 0
			// green 0 - 255 - 255
			const float limit = 200.0f;
			float red = 255 * std::max(-discount_reward / limit, 0.0f);
			float green = 255 * std::max(discount_reward / limit, 0.0f);
			olc::Pixel p = olc::Pixel(red, green, 0);
			Draw(i, 400 - discount_reward, p);
			//printf("%f\n", discount_reward);
		}


		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(1800, 800, 1, 1))
		demo.Start();
	return 0;
}