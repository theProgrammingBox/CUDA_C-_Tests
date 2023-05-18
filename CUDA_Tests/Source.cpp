#include <iostream>
#include <fstream>

int main()
{
	if (false)
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

		bool b;
		void* ptr;
		
		while (file.read(reinterpret_cast<char*>(&b), sizeof(b)) &&
			file.read(reinterpret_cast<char*>(&ptr), sizeof(ptr)))
		{
			std::cout << "Bool: " << std::boolalpha << b << "\n";
			std::cout << "Address: " << ptr << "\n";
		}
	}

    return 0;
}