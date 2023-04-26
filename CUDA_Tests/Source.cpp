#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

void r4_nor2_setup(int32_t kn[128], float fn[128], float wn[128])
{
	double dn = 3.442619855899;
	const double m1 = 2147483648.0;
	const double vn = 9.91256303526217E-03;
	double q = vn / exp(-0.5 * dn * dn);

	kn[0] = dn / q * m1;
	kn[1] = 0;
	wn[0] = q / m1;
	wn[127] = dn / m1;
	fn[0] = 1.0;
	fn[127] = exp(-0.5 * dn * dn);

	double tn;
	for (uint8_t i = 126; 1 <= i; i--)
	{
		tn = dn;
		dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
		kn[i + 1] = dn / tn * m1;
		fn[i] = exp(-0.5 * dn * dn);
		wn[i] = dn / m1;
	}

	return;
}

class Example : public olc::PixelGameEngine
{
public:
	uint32_t seed;
	uint32_t offset;

	Example()
	{
		sAppName = "Example";
	}

	bool OnUserCreate() override
	{
		seed = std::chrono::system_clock::now().time_since_epoch().count();
		offset = std::chrono::system_clock::now().time_since_epoch().count();


		int32_t kn2[128];
		float fn2[128];
		float wn2[128];
		r4_nor2_setup(kn2, fn2, wn2);

		// now printing the arrays in hex, format so that it can be copy pasted into the kernel
		printf("int32_t kn2[128] = {");
		for (int i = 0; i < 128; ++i)
		{
			if (i % 8 == 0)
				printf("\n\t");
			printf("0x%08x, ", kn2[i]);
		}
		printf("\n};\n\n");

		printf("float fn2[128] = {");
		for (int i = 0; i < 128; ++i)
		{
			if (i % 8 == 0)
				printf("\n\t");
			printf("0x%08x, ", *(int32_t*)&fn2[i]);
		}
		printf("\n};\n\n");

		printf("float wn2[128] = {");
		for (int i = 0; i < 128; ++i)
		{
			if (i % 8 == 0)
				printf("\n\t");
			printf("0x%08x, ", *(int32_t*)&wn2[i]);
		}
		printf("\n};\n\n");

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		uint32_t idx = offset;
		for (int x = 0; x < ScreenWidth(); ++x)
			for (int y = 0; y < ScreenHeight(); ++y)
			{
				uint16_t* idx2 = (uint16_t*)&idx;
				uint16_t* offset2 = (uint16_t*)&offset;
				uint16_t* seed2 = (uint16_t*)&seed;
				uint16_t h = (0xE558 ^ idx2[0] + idx2[1] ^ offset2[0] + offset2[1] + seed2[0] ^ seed2[1]) * 0x9E97;
				h = (h >> 7 ^ h) * 0x7A1B;
				Draw(x, y, olc::PixelF(((uint8_t*)&h)[0], ((uint8_t*)&h)[1], ((uint8_t*)&h)[0]));
				++idx;
			}
		seed += 0xA8835963;
		offset += 0x8B7A1B65;
		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(1536, 768, 1, 1))
		demo.Start();
	return 0;
}