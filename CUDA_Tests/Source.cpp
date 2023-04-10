#include <iostream>
#include <chrono>

void randomNormalSetup(uint32_t kn[128], float fn[128], float wn[128])
{
    double dn = 3.442619855899;
    const double m1 = 2147483648.0;
    const double vn = 9.91256303526217e-03;
    double q = vn / exp(-0.5 * dn * dn);

    kn[0] = dn / q * m1;
    kn[1] = 0;

    wn[0] = q / m1;
    wn[127] = dn / m1;

    fn[0] = 1.0;
    fn[127] = (float)(exp(-0.5 * dn * dn));

    double tn = 3.442619855899;
    for (uint8_t i = 126; 1 <= i; i--)
    {
        dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
        kn[i + 1] = dn / tn * m1;
        tn = dn;
        fn[i] = exp(-0.5 * dn * dn);
        wn[i] = dn / m1;
    }
}

float randomNormal(uint32_t& seed, const uint32_t kn[128], const float fn[128], const float wn[128])
{
    uint32_t tempSeed = seed;
    seed = (seed ^ (seed << 13));
    seed = (seed ^ (seed >> 17));
    seed = (seed ^ (seed << 5));
    int32_t randomInt = tempSeed + seed;
    uint32_t sevenBits = randomInt & 127;

    if (randomInt < kn[sevenBits] || randomInt & 0x80000000 && ~randomInt + 1 < kn[sevenBits])
        return wn[sevenBits] * randomInt;

    float x, y;
    for (;;)
    {
        if (sevenBits == 0)
        {
            for (;;)
            {
                tempSeed = seed;
                seed = (seed ^ (seed << 13));
                seed = (seed ^ (seed >> 17));
                seed = (seed ^ (seed << 5));
                x = -0.2904764f * log((tempSeed + seed) * 2.32830643654e-10f);

                tempSeed = seed;
                seed = (seed ^ (seed << 13));
                seed = (seed ^ (seed >> 17));
                seed = (seed ^ (seed << 5));
                y = -log((tempSeed + seed) * 2.32830643654e-10f);

                if (x * x <= y + y)
                {
                    if (randomInt & 0x80000000)
                        return -3.442620f - x;
                    return 3.442620f + x;
                }
            }
        }

        x = wn[sevenBits] * randomInt;
        tempSeed = seed;
        seed = (seed ^ (seed << 13));
        seed = (seed ^ (seed >> 17));
        seed = (seed ^ (seed << 5));
        if (fn[sevenBits] + (tempSeed + seed) * 2.32830643654e-10f * (fn[sevenBits - 1] - fn[sevenBits]) < exp(-0.5f * x * x))
            return x;

        tempSeed = seed;
        seed = (seed ^ (seed << 13));
        seed = (seed ^ (seed >> 17));
        seed = (seed ^ (seed << 5));
        randomInt = tempSeed + seed;
        sevenBits = randomInt & 127;
        if (randomInt < kn[sevenBits] || randomInt & 0x80000000 && ~randomInt + 1 < kn[sevenBits])
            return wn[sevenBits] * randomInt;
    }
}

int main()
{
	uint32_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	uint32_t kn[128];
	float fn[128];
	float wn[128];

	randomNormalSetup(kn, fn, wn);

    // histogram
    const uint32_t bins = 56;
	const uint32_t samples = 1000000;
    uint32_t hist[bins];
	float min = -3.0f;
	float max = 3.0f;
	float bin_width = (max - min) / bins;
    const float scale = 1000.0f / samples;

    
	memset(hist, 0, sizeof(hist));
    
	for (uint32_t i = 0; i < samples; i++)
	{
		float value = randomNormal(seed, kn, fn, wn);
		uint32_t bin = (uint32_t)((value - min) / bin_width);
		if (bin < bins)
		{
			hist[bin]++;
		}
	}
    
	for (uint32_t i = 0; i < bins; i++)
	{
		for (uint32_t j = scale * hist[i]; j--;)
		{
			printf("*");
		}
		printf("\n");
	}

	return 0;
}