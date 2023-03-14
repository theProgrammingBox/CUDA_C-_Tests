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

float LowLevelf32Mul(float a, float b)
{
	float ans = a * b;
	uint32_t answer = *(uint32_t*)&ans;
	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (answer >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");
	
	uint32_t aI = *(uint32_t*)&a;
	uint32_t bI = *(uint32_t*)&b;
	/*unsigned int mantissa_bits = (float_bits & 0x7FFFFF) | 0x800000;
	int mantissa_shift = 150 - ((float_bits >> 23) & 0xFF);
	unsigned long long mantissa = (unsigned long long)mantissa_bits << mantissa_shift;*/
	uint32_t mantA = aI & 0x7FFFFF;
	uint32_t mantB = bI & 0x7FFFFF;
	uint32_t expA = aI >> 23 & 0xFF;
	uint32_t expB = bI >> 23 & 0xFF;

	uint32_t resultExp = LowLevelI32Add(expA, expB);
	uint32_t resultMant = LowLevelI32Mul(mantA, mantB);
	uint32_t resultSign = (aI ^ bI) & 0x80000000;
	
	/*for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (resultExp >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");*/

	while (resultMant & 0xFF800000)
	{
		resultMant >>= 1;
		LowLevelI32Add(resultExp, 1);
	}

	// Round the result
	if (resultMant & 0x400000)
	{
		LowLevelI32Add(resultMant, 0x800000);
	}

	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (resultMant >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");
	
	resultExp = LowLevelI8Add(resultExp, -127);

	uint32_t result = (resultExp) << 23;
	result |= resultMant | resultSign;

	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (result >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");
	
	return *(float*)&result;
}

uint32_t main()
{
	Random random(Random::MakeSeed(275));
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
		float o = LowLevelf32Mul(l, m);
		if (n != o) printf("%f * %f = %f\n%f * %f = %f\n\n", l, m, n, l, m, o);
	}
	
	return 0;
}