//#include <cublas_v2.h>
//#include <curand.h>
#include <iostream>
#include <vector>
#include <chrono>

using std::cout;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

uint32_t seed = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();

uint32_t random()
{
	// xorshift32
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

int main()
{
	/*float* arr;
	arr = new float[100];
	arr[10] = 2435;
	float* placeHolder = arr;
	cout << placeHolder[10] << "\n";
	delete[] arr;
	cout << placeHolder[10] << "\n";*/

	
	class Balls
	{
	public:
		struct ball
		{
			float* arr;

			ball()
			{
				arr = new float[100];
				arr[10] = 2435;
			}
		};
		
		vector<ball> balls;
		
		Balls()
		{
		}

		void addBall()
		{
			balls.push_back(ball());
		}

		void clear()
		{
			for (int i = 0; i < balls.size(); i++)
			{
				delete[] balls[i].arr;
			}
			balls.clear();
		}

		~Balls()
		{
			for (int i = 0; i < balls.size(); i++)
			{
				delete[] balls[i].arr;
			}
		}
	};

	float* placeHolder;
	while (true)
	{
		for (int i = 0; i == 0; i++)
		{
			Balls balls;
			balls.addBall();
			placeHolder = balls.balls[0].arr;
			cout << placeHolder[10] << "\n";
		}
		cout << placeHolder[10] << "\n";
	}

	return 0;
}