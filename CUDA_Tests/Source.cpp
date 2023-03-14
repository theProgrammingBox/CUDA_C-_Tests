#include <iostream>
#include "Random.h"

uint32_t LowLevelI32Add(uint32_t a, uint32_t b)
{
	uint32_t carry;
	while (b)
	{
		carry = a & b;
		a = a ^ b;
		b = carry << 1;
	}
	return a;
}

uint32_t LowLevelI32Abs(uint32_t a)
{
	return LowLevelI32Add(~a, 1);
}

uint32_t LowLevelI32Mul(uint32_t a, uint32_t b)
{
	bool aPos = a & 0x80000000;
	bool bPos = b & 0x80000000;
	if (aPos) a = LowLevelI32Abs(a);
	if (bPos) b = LowLevelI32Abs(b);
	uint32_t result = 0;
	while (b > 0)
	{
		if (b & 1) result = LowLevelI32Add(result, a);
		b >>= 1;
		a <<= 1;
	}
	if (aPos ^ bPos) result = LowLevelI32Abs(result);
	return result;
}

/*float LowLevelf32Add(float a, float b)
{
	uint32_t aI = *(uint32_t*)&a;
	uint32_t bI = *(uint32_t*)&b;
	bool signA = aI & 0x80000000;
	bool signB = bI & 0x80000000;
	
	
	return *(float*)&resultI;
}*/

uint32_t main()
{
	Random random(Random::MakeSeed(275));
	for (uint32_t i = 10; i--;)
	{
		int32_t a = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t b = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t c = a * b;
		int32_t d = LowLevelI32Mul(a, b);

		int32_t e = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t f = (int32_t)random.Ruint32() * 0.0000001f;
		int32_t g = e + f;
		int32_t h = LowLevelI32Add(e, f);
		if (c ^ d)
			printf("%d * %d = %d\n%d * %d = %d\n\n", a, b, c, a, b, d);
		if (e ^ f)
			printf("%d + %d = %d\n%d + %d = %d\n\n", e, f, g, e, f, h);
	}
	return 0;
}