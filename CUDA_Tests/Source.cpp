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

uint32_t LowLevelI32Flip(uint32_t x)
{
	return LowLevelI32Add(~x, 1);
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

float LowLevelf32Mul(float x, float y)
{
	uint32_t I32x = *(uint32_t*)&x;
	uint32_t I32y = *(uint32_t*)&y;
	
	uint64_t resultMantissa = LowLevelI64Mul((uint64_t)I32x & 0x7FFFFF | 0x800000, (uint64_t)I32y & 0x7FFFFF | 0x800000);
	uint32_t resultExponent = LowLevelI32Add(LowLevelI32Add(I32x >> 23 & 0xFF, I32y >> 23 & 0xFF), -127);
	bool roundUp = resultMantissa & 0x400000;
	resultMantissa >>= 23;
	
	while (resultMantissa & 0xffffffffff000000)
	{
		roundUp = resultMantissa & 1;
		resultMantissa >>= 1;
		resultExponent = LowLevelI32Add(resultExponent, 1);
	}
	resultMantissa = LowLevelI64Add(resultMantissa, roundUp);
	
	while (resultMantissa && !(resultMantissa & 0x800000))
	{
		resultMantissa <<= 1;
		resultExponent = LowLevelI32Add(resultExponent, -1);
	}
	
	uint32_t result = (I32x ^ I32y) & 0x80000000 | resultExponent << 23 | (resultMantissa & 0x7FFFFF);
	
	return *(float*)&result;
}

float LowLevelf32Add(float x, float y)
{
	uint32_t I32a = *(uint32_t*)&x;
	uint32_t I32b = *(uint32_t*)&y;
	
	uint32_t xExponent = I32a >> 23 & 0xFF;
	uint32_t yExponent = I32b >> 23 & 0xFF;

	uint32_t xMantissa = I32a & 0x7FFFFF | 0x800000;
	uint32_t yMantissa = I32b & 0x7FFFFF | 0x800000;

	if (I32a & 0x80000000)
		xMantissa = LowLevelI32Flip(xMantissa);
	if (I32b & 0x80000000)
		yMantissa = LowLevelI32Flip(yMantissa);

	if (y > x)	// change
	{
		std::swap(I32a, I32b);
		std::swap(xExponent, yExponent);
		std::swap(xMantissa, yMantissa);
	}

	for (;;)
	{
		if (xExponent <= yExponent) break;
		xMantissa >>= 1;
		xExponent = LowLevelI32Add(xExponent, 1);
		if (xExponent <= yExponent) break;
		yMantissa <<= 1;
		yExponent = LowLevelI32Add(yExponent, -1);
	}

	uint32_t resultMantissa = LowLevelI32Add(xMantissa, yMantissa);
	uint32_t resultExponent = xExponent;
	uint32_t resultSign = 0;

	if (resultMantissa & 0x80000000)
	{
		resultSign = 0x80000000;
		LowLevelI32Flip(resultMantissa);
	}
	
	while (resultMantissa & 0xff000000)
	{
		resultMantissa >>= 1;
		resultExponent = LowLevelI32Add(resultExponent, 1);
	}

	while (resultMantissa && !(resultMantissa & 0x800000))
	{
		resultMantissa <<= 1;
		resultExponent = LowLevelI32Add(resultExponent, -1);
	}
	
	uint32_t result = resultSign | resultExponent << 23 | (resultMantissa & 0x7FFFFF);
	float answer = x + y;
	uint32_t U32a = *(uint32_t*)&answer;

	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (U32a >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");

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
	for (uint32_t i = 10; i--;)
	{
		float x = random.Rfloat(-100, 100);
		float y = random.Rfloat(-100, 100);
		//printf("%f * %f = %f\n", x, y, LowLevelf32Mul(x, y));
		printf("%f + %f = %f\n", x, y, LowLevelf32Add(x, y));
		printf("%f + %f = %f\n\n", x, y, x + y);
		/*int32_t x = (int32_t)random.Ruint32() * 0.00000001f;
		printf("%d = %d\n", x, LowLevelI32Flip(x));*/
	}
	
	return 0;
}