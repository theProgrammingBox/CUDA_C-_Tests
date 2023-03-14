#include <iostream>
#include "Random.h"

int32_t LowLevelI32Add(int32_t a, int32_t b)
{
	int32_t carry;
	while (b)
	{
		carry = a & b;
		a = a ^ b;
		b = carry << 1;
	}
	return a;
}

int32_t LowLevelI32Abs(int32_t a)
{
	return LowLevelI32Add(~a, 1);
}

int32_t LowLevelI32Mul(int32_t a, int32_t b)
{
	bool aPos = a & 0x80000000;
	bool bPos = b & 0x80000000;
	if (aPos) a = LowLevelI32Abs(a);
	if (bPos) b = LowLevelI32Abs(b);
	int32_t result = 0;
	while (b > 0)
	{
		if (b & 1)
			result = LowLevelI32Add(result, a);
		b >>= 1;
		a <<= 1;
	}
	if (aPos ^ bPos)
		result = LowLevelI32Abs(result);
	return result;
}

int32_t main()
{
	Random random(Random::MakeSeed());
	for (uint32_t i = 1000; i--;)
	{
		int32_t a = random.Ruint32();
		int32_t b = random.Ruint32();
		int32_t c = a * b;
		int32_t d = LowLevelI32Mul(a, b);
		int32_t e = a + b;
		int32_t f = LowLevelI32Add(a, b);
		if (c ^ d)
			printf("%d * %d = %d, %d * %d = %d\n", a, b, c, a, b, d);
		if (e ^ f)
			printf("%d + %d = %d, %d + %d = %d\n", a, b, e, a, b, f);
	}
	return 0;
}