#include <iostream>

int main()
{
    float weight = 2;
    float bias = 1;
    /*while(true)
    {*/
        for (int i = 16; i--;)
        {
            float input = (float)rand() / RAND_MAX;
            float output = input * weight + bias;
            float move = std::max(0.0f, std::min(3.0f, floor(output)));
            printf("%f %f %f\n", input, output, move);
        }
    /*}*/

    return 0;
}