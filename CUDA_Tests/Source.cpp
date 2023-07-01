#include <iostream>
#include <vector>

struct Layer
{
	int size;
	float* arr;

	Layer(int size) : size(size)
	{
		arr = new float[size];
	}

	~Layer()
	{
		delete[] arr;
	}
};

struct SomeLayer : Layer
{
	SomeLayer(int size) : Layer(size)
	{
		for (int i = 0; i < size; i++)
		{
			arr[i] = 0.0f;
		}
	}
};

struct Network
{
	std::vector<Layer*> layers;

	~Network()
	{
		for (auto layer : layers)
		{
			delete layer;
		}
	}

	void AddLayer(Layer* layer)
	{
		layers.emplace_back(layer);
	}
};

int main()
{
	Network net;
	net.AddLayer(new SomeLayer(10));

	return 0;
}