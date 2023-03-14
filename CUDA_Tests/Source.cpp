#include <iostream>
#include "Random.h"

uint32_t LowLevelI32Add(uint32_t a, uint32_t b)
{
	uint32_t carry;
	while (b)
	{
		carry = a & b;
		a ^= b;
		b = carry << 1;
	}
	return a;
}

uint32_t LowLevelI32Neg(uint32_t a)
{
	return LowLevelI32Add(~a, 1);
}

uint32_t LowLevelI32Mul(uint32_t x, uint32_t y)
{
	uint32_t result = 0;
	while (y)
	{
		if (y & 1) result = LowLevelI32Add(result, x);
		y >>= 1;
		x <<= 1;
	}
	return result;
}

uint8_t LowLevelI8Add(uint8_t a, uint8_t b)
{
	uint8_t carry;
	while (b)
	{
		carry = a & b;
		a ^= b;
		b = carry << 1;
	}
	return a;
}

float LowLevelf32Add(float a, float b)
{
	uint32_t aI = *(uint32_t*)&a;
	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (aI >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n\n");
	uint32_t bI = *(uint32_t*)&b;
	uint32_t mantA = aI & 0x7FFFFF;
	uint32_t mantB = bI & 0x7FFFFF;
	int8_t expA = aI >> 23 & 0xFF;
	int8_t expB = bI >> 23 & 0xFF;
	printf("%d\n", expA);
	for (int32_t i = 7; i >= 0; i--)
	{
		printf("%d", (expA >> i) & 1);
	}
	printf("\n\n");

	int8_t resultExp = LowLevelI8Add(expA, expB);
	uint32_t resultMant = LowLevelI32Mul(mantA, mantB);
	uint32_t resultSign = (aI ^ bI) & 0x80000000;

	while (resultMant & 0xFF800000)
	{
		resultMant >>= 1;
		LowLevelI32Add(resultExp, 1);
	}

	/*printf("%d\n", resultExp);
	resultExp = LowLevelI8Add(resultExp, 127);
	printf("%d\n", resultExp);
	for (int32_t i = 7; i >= 0; i--)
	{
		printf("%d", (resultExp >> i) & 1);
	}
	printf("\n\n");*/
	/*if (resultExp > 255)
	{
		resultExp = 255;
		resultMant = 0;
	}*/

	uint32_t result = resultMant | resultSign;// | (resultExp << 23);

	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (result >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n\n");
	
	return *(float*)&result;
}

uint32_t main()
{
	int8_t a = -110 - 129;
	for (int32_t i = 7; i >= 0; i--)
	{
		printf("%d", (a >> i) & 1);
	}
	printf("\n");
	/*Random random(Random::MakeSeed(275));
	for (uint32_t itr = 10; itr--;)
	{
		int32_t a = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t b = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t c = a * b;
		int32_t d = LowLevelI32Mul(a, b);
		if (c != d) printf("%d * %d = %d\n%d * %d = %d\n\n", a, b, c, a, b, d);

		int32_t e = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t f = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t g = e + f;
		int32_t h = LowLevelI32Add(e, f);
		if (g != h) printf("%d + %d = %d\n%d + %d = %d\n\n", e, f, g, e, f, h);

		int32_t i = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t j = i * -1;
		int32_t k = LowLevelI32Neg(i);
		if (j != k) printf("%d * -1 = %d\n%d * -1 = %d\n\n", i, j, i, k);

		float l = random.Rfloat(-10, 10);
		float m = random.Rfloat(-10, 10);
		float n = l * m;
		float o = LowLevelf32Add(l, m);
		if (n != o) printf("%f + %f = %f\n%f + %f = %f\n\n", l, m, n, l, m, o);
	}*/
	return 0;
}