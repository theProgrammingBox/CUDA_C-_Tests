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

/*float LowLevelf32Add(float a, float b)
{
	uint32_t aI = *(uint32_t*)&a;
	uint32_t bI = *(uint32_t*)&b;
	bool signA = aI & 0x80000000;
	bool signB = bI & 0x80000000;
	uint32_t expA = (aI >> 23) & 0xFF;
	uint32_t expB = (bI >> 23) & 0xFF;
	uint32_t mantA = aI & 0x7FFFFF;
	uint32_t mantB = bI & 0x7FFFFF;

	
	return *(float*)& resultI;
}*/

uint32_t main()
{
	Random random(Random::MakeSeed(275));
	for (uint32_t itr = 10; itr--;)
	{
		int32_t a = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t b = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t c = a * b;
		int32_t d = LowLevelI32Mul(a, b);
		if (c != d)
		printf("%d * %d = %d\n%d * %d = %d\n\n", a, b, c, a, b, d);

		int32_t e = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t f = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t g = e + f;
		int32_t h = LowLevelI32Add(e, f);
		if (g != h)
		printf("%d + %d = %d\n%d + %d = %d\n\n", e, f, g, e, f, h);

		int32_t i = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t j = i * -1;
		int32_t k = LowLevelI32Neg(i);
		if (j != k)
		printf("%d * -1 = %d\n%d * -1 = %d\n\n", i, j, i, k);
	}
	return 0;
}