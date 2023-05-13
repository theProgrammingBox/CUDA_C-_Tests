#include <iostream>
#include <assert.h>

struct Param4D
{
	uint32_t height;
	uint32_t width;
	uint32_t channels;
	uint32_t batches;

	Param4D(uint32_t height = 1, uint32_t width = 1, uint32_t channels = 1, uint32_t batches = 1)
	{
		assert(height > 0 && width > 0 && channels > 0 && batches > 0);
		this->height = height;
		this->width = width;
		this->channels = channels;
		this->batches = batches;
	}

	Param4D(const Param4D* other)
	{
		this->height = other->height;
		this->width = other->width;
		this->channels = other->channels;
		this->batches = other->batches;
	}

	void Print() const
	{
		printf("(%u, %u, %u, %u)\n", height, width, channels, batches);
	}
};

int main()
{
	Param4D param = { 64 };
	Param4D* parameter = new Param4D(param);
	Param4D* parameter2 = new Param4D(parameter);
	parameter->Print();
}