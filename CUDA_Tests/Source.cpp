#include <iostream>

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

int32_t LowLevelI32Mul(int32_t a, int32_t b)
{
	bool aPos = a & 0x80000000;
	bool bPos = b & 0x80000000;
	if (aPos) a = LowLevelI32Add(~a, 1);
	if (bPos) b = LowLevelI32Add(~b, 1);
	int32_t result = 0;
	while (b > 0)
	{
		if (b & 1)
			result = LowLevelI32Add(result, a);
		b >>= 1;
		a <<= 1;
	}
	if (aPos ^ bPos)
		result = LowLevelI32Add(~result, 1);
	return result;
}

int32_t main()
{
	printf("LowLevelI32Add(5, 7) = %d\n", LowLevelI32Add(5, 7));
	printf("LowLevelI32Add(5, -7) = %d\n", LowLevelI32Add(5, -7));
	printf("multiply_bitewise(5, 7) = %d\n", LowLevelI32Mul(5, 7));
	printf("multiply_bitewise(5, -7) = %d\n", LowLevelI32Mul(5, -7));
	return 0;
}