﻿#include <iostream>
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

		b = 0;
		file.write(reinterpret_cast<const char*>(&b), sizeof(b));
		file.write(reinterpret_cast<const char*>(&ptr), sizeof(ptr));
		delete ptr;
		printf("Erasing address: %p\n", ptr);
	}
	file.close();
	printf("\n\n\n");

	std::ifstream file2("data.bin", std::ios::binary);
	if (!file2)
	{
		std::cerr << "Couldn't open file for reading.\n";
		return 1;
	}

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