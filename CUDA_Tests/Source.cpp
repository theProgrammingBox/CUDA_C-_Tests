#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

// Override base class with your custom functionality
class Example : public olc::PixelGameEngine
{
public:
	const float reward = 1;
	const float discount = 0.99f;
	const float limit = reward / (1 - discount);
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
		Clear(olc::BLACK);
		
		if (GetKey(olc::Key::UP).bHeld)
		{
			rewards[idx] = reward;
		}
		else if (GetKey(olc::Key::DOWN).bHeld)
		{
			rewards[idx] = -reward;
		}
		else
		{
			rewards[idx] = 0;
		}
		
		float discount_reward = 0;
		for (int i = samples; i--;)
		{

			discount_reward = rewards[idx] + discount * discount_reward;
			
			int red = 255 * std::min(std::max(-discount_reward / limit + 1.0f, 0.0f), 1.0f);
			int green = 255 * std::min(std::max(discount_reward / limit + 1.0f, 0.0f), 1.0f);
			int blue = 255 * std::max(1 - abs(discount_reward / limit), 0.0f);
			olc::Pixel color = olc::Pixel(red, green, blue);
			
			Draw(i, ScreenHeight() / 2 - discount_reward, color);
			
			idx++;
			idx *= idx < samples;
		}
		idx--;
		idx += (idx < 0) * samples;
		
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