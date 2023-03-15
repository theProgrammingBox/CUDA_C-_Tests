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
	uint32_t aI = *(uint32_t*)&a;
	uint32_t bI = *(uint32_t*)&b;
	
	uint64_t mantA = aI & 0x7FFFFF | 0x800000;
	uint64_t mantB = bI & 0x7FFFFF | 0x800000;

	uint32_t expA = aI >> 23 & 0xFF;
	uint32_t expB = bI >> 23 & 0xFF;

	uint64_t resultMant = LowLevelI64Mul(mantA, mantB);
	uint32_t resultExp = LowLevelI32Add(LowLevelI32Add(expA, expB), -127);
	bool lastBit = resultMant & 0x400000;
	resultMant >>= 23;
	
	/*do
	{*/
		while (resultMant & 0xffffffffff000000)
		{
			lastBit = resultMant & 1;
			resultMant >>= 1;
			resultExp = LowLevelI32Add(resultExp, 1);
		}
		resultMant = LowLevelI64Add(resultMant, lastBit);
	//} while (resultMant & 0xffffffffff000000);
	
	while (!(resultMant & 0x800000))
	{
		resultMant <<= 1;
		resultExp = LowLevelI32Add(resultExp, -1);
	}
	
	uint32_t result = (aI ^ bI) & 0x80000000 | resultExp << 23 | (resultMant & 0x7FFFFF);

	float ans = a * b;
	uint32_t answer = *(uint32_t*)&ans;
	if (answer ^ result)
	{
		for (int32_t i = 31; i >= 0; i--)
		{
			printf("%d", (answer >> i) & 1);
			if (i == 31 || i == 23) printf(" ");
		}
		printf("\n");

		for (int32_t i = 31; i >= 0; i--)
		{
			printf("%d", (result >> i) & 1);
			if (i == 31 || i == 23) printf(" ");
		}
		printf("\n\n");
	}
	
	return *(float*)&result;
}

float LowLevelf32Add(float a, float b)
{
	return 0;
}

uint32_t main()
{
	Random random(Random::MakeSeed(275));
	for (uint32_t itr = 100000000000; itr--;)
	{
		float x = random.Rfloat(-100, 100);
		float y = random.Rfloat(-100, 100);
		LowLevelf32Mul(x, y);
	}
	
	return 0;
}