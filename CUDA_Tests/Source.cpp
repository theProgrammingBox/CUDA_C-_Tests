#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

#define NUM_SAMPLES 1000000
#define NUM_BINS 100

double uniform_random_range(double min, double max) {
    double u = (double)rand() / (double)RAND_MAX;
    return min + u * (max - min);
}

double marsaglia_polar_s() {
    double u, v, s;
    /*do {
        u = uniform_random_range(-1.0, 1.0);
        v = uniform_random_range(-1.0, 1.0);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    double factor = sqrt(-2.0 * log(s) / s);
    return u * factor * 0.1f + 0.5f;*/

    /*s = uniform_random_range(0.0, 1.0);
    double factor = sqrt(-2.0 * log(s) / s);
    return uniform_random_range(-1.0, 1.0) * factor * 0.1f + 0.5f;*/
    do {
        u = uniform_random_range(-1.0, 1.0);
        v = uniform_random_range(-1.0, 1.0);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    return u * 0.5 + 0.5;
}

int main() {
    srand(time(NULL));

    uint32_t histogram[NUM_BINS] = { 0 };
    float scale = (float)NUM_BINS / NUM_SAMPLES * 20;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        double s = marsaglia_polar_s();
        int bin_index = (int)(s * NUM_BINS);
        if (bin_index < NUM_BINS && bin_index >= 0) {
            histogram[bin_index]++;
		}
    }

    printf("Histogram of s values:\n");
    printf("Bin\tFrequency\n");

    for (int i = 0; i < NUM_BINS; i++) {
        for (int j = 0; j < histogram[i] * scale; j++) {
			printf("*");
		}
        		printf("\n");
    }

    return 0;
}
