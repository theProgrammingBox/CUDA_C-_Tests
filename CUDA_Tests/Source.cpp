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

		const float special1 = -1.1641532183e-10f;
		const float special2 = -22.1807097779f;
		const uint32_t idx = 1;
		uint32_t seed1 = 14554362;
		uint32_t seed2 = 142343152;

		seed1 = (0xE558D374 ^ idx + seed1 ^ seed2) * 0xAA69E974;
		seed1 = (seed1 >> 13 ^ seed1) * 0x8B7A1B65;

		uint16_t u1 = ((uint16_t*)&seed1)[0];
		uint16_t u2 = ((uint16_t*)&seed1)[1];

		uint32_t s = u1 * u1 + u2 * u2;

		while (s >= 0x10000 || s == 0)
		{
			seed1 = (seed1 >> 13 ^ seed1) * 0x8B7A1B65;
			u1 = ((uint16_t*)&seed1)[0];
			u2 = ((uint16_t*)&seed1)[1];
			s = u1 * u1 + u2 * u2;
		}
		printf("%f\n", u1 / 65535.0f);
		printf("%f\n", u2 / 65535.0f);

		printf("%u\n", u1);
		printf("%u\n", u2);
		printf("%u\n", s);

		const float r = special1 * s / (logf(s) + special2);
		printf("%f\n", r);
		printf("%f\n", special1 * s);
		printf("%f\n", logf(s) + special2);

		uint32_t i = 0x5F1FFFF9 - (*(uint32_t*)&r >> 1);
		float tmp = *(float*)&i;
		tmp *= 0.0000107414589386f * (2.38924456f - r * tmp * tmp);

		printf("%f\n", tmp * u1);
		printf("%f\n", tmp * u2);

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