#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

class Example : public olc::PixelGameEngine
{
public:
	const float reward = 1;
	const float discount = 0.99f;
	const float lamda = 0.95f;
	const float invLimit = (1 - discount) / reward;
	static const int samples = 1800;
	float rewards[samples];
	int idx;
	float middle;

	bool OnUserCreate() override
	{
		memset(rewards, 0, sizeof(rewards));
		idx = 0;
		middle = ScreenHeight() * 0.5f;
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		Clear(olc::BLACK);

		rewards[idx] = (GetKey(olc::Key::UP).bHeld - GetKey(olc::Key::DOWN).bHeld) * reward;

		float discount_reward = 0;
		float advantage = 0;
		float value = 0;
		for (int i = samples; i--;)
		{
			// test
			discount_reward = rewards[idx] + discount * discount_reward;
			advantage = rewards[idx] + discount * value - rewards[idx] + discount * lamda * advantage;
			value = rewards[idx];

			float norm = discount_reward * invLimit;
			float normAdv = advantage * invLimit;

			int red = 255 * std::max(std::min(1.0f - norm, 1.0f), 0.0f);
			int green = 255 * std::max(std::min(1.0f + norm, 1.0f), 0.0f);
			int blue = 255 * std::max(1.0f - abs(norm), 0.0f);
			Draw(i, middle - discount_reward, olc::Pixel(red, green, blue));

			red = 255 * std::max(std::min(1.0f - normAdv, 1.0f), 0.0f);
			green = 255 * std::max(std::min(1.0f + normAdv, 1.0f), 0.0f);
			blue = 255 * std::max(1.0f - abs(normAdv), 0.0f);
			Draw(i, middle - advantage, olc::Pixel(red, green, blue));

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