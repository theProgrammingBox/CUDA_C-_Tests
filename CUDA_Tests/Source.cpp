#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

class Random
{
public:
	Random(uint32_t seed = 0x8a777e69) {
		double dn = 3.442619855899;
		const double m1 = 2147483648.0;
		const double vn = 9.91256303526217E-03;
		double q = vn / exp(-0.5 * dn * dn);

		kn[0] = (uint32_t)((dn / q) * m1);
		kn[1] = 0;

		wn[0] = (float)(q / m1);
		wn[127] = (float)(dn / m1);

		fn[0] = 1.0;
		fn[127] = (float)(exp(-0.5 * dn * dn));

		double tn = 3.442619855899;
		for (uint8_t i = 126; i--;)
		{
			dn = sqrt(-2.0 * log(vn / dn + exp(-0.5f * dn * dn)));
			kn[i + 1] = (uint32_t)((dn / tn) * m1);
			tn = dn;
			fn[i] = (float)(exp(-0.5 * dn * dn));
			wn[i] = (float)(dn / m1);
		}
	}

	float operator()()
	{
		int32_t hz = rand() << 24 ^ rand() << 16 ^ rand() << 8 ^ rand();
		uint8_t sevenBits = hz & 127;
		
		if (fabs(hz) < kn[sevenBits])
			return float(hz) * wn[sevenBits];
		
		uint32_t temp;
		uint32_t seed = hz;
		float x;
		float y;
		const float r = 3.442620f;

		for (;;)
		{
			if (sevenBits == 0)
			{
				do
				{
					temp = seed;
					seed = (seed ^ (seed << 13));
					seed = (seed ^ (seed >> 17));
					seed = (seed ^ (seed << 5));
					seed = temp + seed;
					x = -0.2904764f * log(seed * 2.32830643654e-10f);
					
					temp = seed;
					seed = (seed ^ (seed << 13));
					seed = (seed ^ (seed >> 17));
					seed = (seed ^ (seed << 5));
					seed = temp + seed;
					y = -log(seed * 2.32830643654e-10f);
				} while (x * x > y + y);

				x += r;
				if (hz <= 0)
					x = 0 - x;
				return x;
			}

			x = float(hz) * wn[sevenBits];

			temp = seed;
			seed = (seed ^ (seed << 13));
			seed = (seed ^ (seed >> 17));
			seed = (seed ^ (seed << 5));
			seed = temp + seed;
			if (fn[sevenBits] + seed * 2.32830643654e-10f * (fn[sevenBits - 1] - fn[sevenBits]) < exp(-0.5f * x * x))
				return x;

			temp = seed;
			seed = (seed ^ (seed << 13));
			seed = (seed ^ (seed >> 17));
			seed = (seed ^ (seed << 5));
			seed = temp + seed;
			hz = seed;
			sevenBits = (hz & 127);

			if (fabs(hz) < kn[sevenBits])
				return float(hz) * wn[sevenBits];
		}
	}

private:
	uint32_t kn[128];
	float fn[128];
	float wn[128];
};

int main() {
	srand(time(NULL));
	float mean = 0;
	float stddev = 1;
	int num_samples = 10000;
	int num_bins = 100;
	float bin_width = 0.1;

	std::vector<int> histogram(num_bins, 0);
	Random random;

	for (int i = 0; i < num_samples; ++i) {
		float random_number = random() * stddev + mean;
		int bin_index = static_cast<int>((random_number - mean + (num_bins * 0.5f) * bin_width) / bin_width);

		if (bin_index >= 0 && bin_index < num_bins) {
			histogram[bin_index]++;
		}
	}

	for (int i = 0; i < num_bins; ++i) {
		std::cout << i << ": ";
		for (int j = 0; j < histogram[i] * 1000.0f / num_samples; ++j) {
			std::cout << "*";
		}
		std::cout << std::endl;
	}

	return 0;
}
