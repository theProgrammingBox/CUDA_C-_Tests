#include <iostream>
#include <fstream>
#include <unordered_set>

int main()
{

	std::ofstream file("data.bin", std::ios::binary);
	if (!file)
	{
		std::cerr << "Couldn't open file for writing.\n";
		return 1;
	}

	bool b;
	float* ptr;

	for (int i = 0; i < 10; i++)
	{
		ptr = new float(i);

		b = 1;
		file.write(reinterpret_cast<const char*>(&b), sizeof(b));
		file.write(reinterpret_cast<const char*>(&ptr), sizeof(ptr));
		printf("Inserting address: %p\n", ptr);

		float discount_reward = 0;
		float advantage = 0;
		float value = 0;
		for (int i = samples; i--;)
		{
			discount_reward = rewards[idx] + discount * discount_reward;
			advantage = rewards[idx] + discount * value - rewards[idx] + discount * lamda * advantage;
			value = rewards[idx];

			float norm = discount_reward * invLimit;
			float normAdv = advantage * invLimit;

			int red = 255 * std::max(std::min(1.0f - norm, 1.0f), 0.0f);
			int green = 255 * std::max(std::min(1.0f + norm, 1.0f), 0.0f);
			int blue = 255 * std::max(1.0f - abs(norm), 0.0f);
			Draw(i, middle - discount_reward, olc::Pixel(red, green, blue));

	std::unordered_set<void*> addressSet;

	while (file2.read(reinterpret_cast<char*>(&b), sizeof(b)) &&
		file2.read(reinterpret_cast<char*>(&ptr), sizeof(ptr)))
	{
		if (b)
		{
			addressSet.insert(ptr);
			printf("Inserting address: %p\n", ptr);
		}
		else
		{
			if (addressSet.find(ptr) == addressSet.end())
			{
				printf("Double delete on address: %p\n", ptr);
			}
			else
			{
				addressSet.erase(ptr);
				printf("Erasing address: %p\n", ptr);
			}
		}
	}
	file2.close();
	printf("\n\n\n");

	if (addressSet.empty())
	{
		printf("Perfect\n");
	}
	else
	{
		printf("Leftover\n");
		for (auto& ptr : addressSet)
		{
			printf("Leftover address: %p\n", ptr);
		}
	}

	return 0;
}