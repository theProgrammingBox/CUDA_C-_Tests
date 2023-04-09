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

class NormalRandomAtHome
{
public:
	NormalRandomAtHome(uint32_t seed = 0x8a777e69) {
		double dn = 3.442619855899;
		const double m1 = 2147483648.0;
		const double vn = 9.91256303526217E-03;
		double q = vn / exp(-0.5 * dn * dn);

		wn[0] = (float)(q / m1);
		wn[127] = (float)(dn / m1);
		
		for (uint8_t i = 126; i--;)
		{
			dn = sqrt(-2.0 * log(vn / dn + exp(-0.5f * dn * dn)));
			wn[i] = (float)(dn / m1);
		}
	}

	float operator()(int32_t seed)
	{
		return seed * wn[seed & 127];
	}

private:
	float wn[128];
};

float rand_normal(float mean, float stddev) {
	float u1 = rand() / (float)RAND_MAX;
	float u2 = rand() / (float)RAND_MAX;

	float z0 = sqrt(-2 * log(u1)) * cos(6.28318530718f * u2);
	return mean + z0 * stddev;
}

int main() {
	srand(time(NULL));
	float mean = 0;
	float stddev = 1;
	int num_samples = 10000;
	int num_bins = 100;
	float bin_width = 0.1;

	std::vector<int> histogram(num_bins, 0);
	NormalRandomAtHome random;

	for (int i = 0; i < num_samples; ++i) {
		float random_number = random(rand() << 24 ^ rand() << 16 ^ rand() << 8 ^ rand()) * stddev + mean;
		//float random_number = rand_normal(mean, stddev);
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
