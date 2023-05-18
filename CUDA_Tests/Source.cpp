#include <iostream>
#include <fstream>

int main()
{
public:
	const float reward = 1;
	const float discount = 0.99f;
	const float invLimit = (1 - discount) / reward;
	static const int samples = 1800;
	float rewards[samples];
	int idx;
	float middle;

	bool OnUserCreate() override
	{
		std::ofstream file("data.bin", std::ios::binary);
		if (!file)
		{
			std::cerr << "Couldn't open file for writing.\n";
			return 1;
		}

		bool b = true;
		void* ptr = reinterpret_cast<void*>(0x12345678);  // Just an example
		
		for (int i = 0; i < 10; i++)
		{
			file.write(reinterpret_cast<const char*>(&b), sizeof(b));
			file.write(reinterpret_cast<const char*>(&ptr), sizeof(ptr));
		}
	}
	else
	{
		std::ifstream file("data.bin", std::ios::binary);
		if (!file) {
			std::cerr << "Couldn't open file for reading.\n";
			return 1;
		}

		rewards[idx] = (GetKey(olc::Key::UP).bHeld - GetKey(olc::Key::DOWN).bHeld) * reward;
		
		float discount_reward = 0;
		for (int i = samples; i--;)
		{
			discount_reward = rewards[idx] + discount * discount_reward;
			
			float norm = discount_reward * invLimit;
			int red = 255 * std::max(std::min(1.0f - norm, 1.0f), 0.0f);
			int green = 255 * std::max(std::min(1.0f + norm, 1.0f), 0.0f);
			int blue = 255 * std::max(1.0f - abs(norm), 0.0f);
			olc::Pixel color = olc::Pixel(red, green, blue);
			
			Draw(i, middle - discount_reward, color);
			
			idx++;
			idx *= idx < samples;
		}
		idx--;
		idx += (idx < 0) * samples;
		
		return true;
	}

    return 0;
}