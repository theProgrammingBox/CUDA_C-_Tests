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

uint64_t LowLevelI64Add(uint64_t a, uint64_t b)
{
	uint64_t carry;
	while (b)
	{
		carry = a & b;
		a ^= b;
		b = carry << 1;
	}
	return a;
}

uint64_t LowLevelI64Mul(uint64_t x, uint64_t y)
{
	uint64_t result = 0;
	while (y)
	{
		if (y & 1) result = LowLevelI64Add(result, x);
		y >>= 1;
		x <<= 1;
	}
	return result;
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

	uint64_t mantA = (aI & 0x7FFFFF) | 0x800000;
	uint64_t mantB = (bI & 0x7FFFFF) | 0x800000;

	uint32_t expA = aI >> 23 & 0xFF;
	uint32_t expB = bI >> 23 & 0xFF;

	uint64_t resultMant = LowLevelI64Mul(mantA, mantB) >> 23;
	uint32_t resultExp = LowLevelI32Add(LowLevelI32Add(expA, expB), -127);
	
	while (resultMant & 0xffffffffff800000)
	{
		resultMant >>= 1;
		resultExp = LowLevelI32Add(resultExp, 1);
	}

	while (!(resultMant & 0x800000))
	{
		resultMant <<= 1;
		resultExp = LowLevelI32Add(resultExp, -1);
	}
	
	uint32_t result = (aI ^ bI) & 0x80000000 | resultExp << 23 | (resultMant & 0x7FFFFF);

	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (result >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");
	
	return *(float*)&result;
}

float LowLevelf32Add(float a, float b)
{
	
	return 0;
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

		float l = random.Rfloat(-100, 100);
		float m = random.Rfloat(-100, 100);
		float n = l * m;
		float o = LowLevelf32Mul(l, m);
		if (n != o) printf("%f * %f = %f\n%f * %f = %f\n\n", l, m, n, l, m, o);
	}
	
	return 0;
}