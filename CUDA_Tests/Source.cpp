#include <fstream>

using std::ofstream;
using std::ifstream;
using std::ios;

class IDK
{
public:
	uint32_t what = -3;
	
	uint32_t GetWhat()
	{
		return what;
	}
};

int main()
{
	ofstream file("test.txt", std::ios::out | std::ios::binary);
	IDK idk;
	idk.what = 5326;
	file.write((char*)&idk.GetWhat(), sizeof(IDK));
	file.close();
	
	ifstream file2("test.txt", std::ios::in | std::ios::binary);
	IDK idk2;
	file2.read((char*)&idk2.what, sizeof(IDK));
	file2.close();
	
	printf("%d", idk2.what);

	return 0;
}