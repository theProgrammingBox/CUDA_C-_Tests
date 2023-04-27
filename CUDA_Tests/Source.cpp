#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

#define NUM_SAMPLES 40000000
#define NUM_BINS 100

float uniform_random_range() {
	const float scale = 1.0f / RAND_MAX;
    float r = rand() | 1;
    return r * scale;
}

//double marsaglia_polar_s() {
//    double u, v, s;
//    do {
//        u = uniform_random_range(-1.0, 1.0);
//        v = uniform_random_range(-1.0, 1.0);
//        s = u * u + v * v;
//    } while (s >= 1.0 || s == 0.0);
//    double factor = sqrt(-2.0 * log(s) / s);
//    return u * factor * 0.08f + 0.5f;
//}

double box_muller() {
	double u = uniform_random_range();
	double v = uniform_random_range();
	double factor = sqrt(-2.0 * log(u));
	return cos(6.28318530718 * v) * factor * 0.08f + 0.5f;
}

double custom() {
    float a = sqrtf(-logf(uniform_random_range()));
    float b = sqrtf(-logf(uniform_random_range()));
    
    return ((a - b) * 1.54) * 0.08 + 0.5;
}

int main() {
    srand(time(NULL));

    uint32_t histogram1[NUM_BINS] = { 0 };
	uint32_t histogram2[NUM_BINS] = { 0 };
    float scale = (float)NUM_BINS / NUM_SAMPLES * 18;

    double s;
    int bin_index;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        s = box_muller();
        bin_index = (int)(s * NUM_BINS);
        histogram1[bin_index]++;
        
        s = custom();
		bin_index = (int)(s * NUM_BINS);
        histogram2[bin_index]++;
    }

    printf("Histogram of s values:\n");
    printf("Bin\tFrequency\n");

    float diff;
    for (int i = 0; i < NUM_BINS; i++) {
        printf("%u\t", i);
		diff = (float)histogram2[i] / histogram1[i] * 100;
		printf("%f%%\t", diff);
        for (int j = 0; j < int(histogram2[i] * scale + 1); j++) {
			printf("*");
		}
        printf("\n");
    }

    return 0;
}
