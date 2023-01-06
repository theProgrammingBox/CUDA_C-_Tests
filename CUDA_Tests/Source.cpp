#include <chrono>
#include <functional>
#include <iostream>

template <typename Func, typename... Args>
double measure_time(Func&& func, Args&&... args) {
    // Use the high-resolution clock to measure the time elapsed
    auto start = std::chrono::high_resolution_clock::now();
    std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in seconds and return it
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    return elapsed.count();
}

void test_function(uint32_t size)
{
	uint32_t x = 0;
	for (uint32_t i = 0; i < size; i++)
		x = (x << 1) - 1;
}

void test_function2(uint32_t size)
{
	uint32_t i = size;
	uint32_t x = 0;
	while (i--)
		x = (x << 1) - 1;
}

void test_function3(uint32_t size)
{
	uint32_t i, x;
	for (i = size, x = 0; i--; x = (x << 1) - 1);
}

int main() {
	auto time1 = 0.0;
	auto time2 = 0.0;
	auto time3 = 0.0;
	
	for (int i = 0; i < 10; ++i)
		time2 += measure_time(test_function, 100000000);
	for (int i = 0; i < 10; ++i)
		time1 += measure_time(test_function2, 100000000);
	for (int i = 0; i < 10; ++i)
		time3 += measure_time(test_function3, 100000000);
	
	std::cout << time1 / 10 << " seconds" << std::endl;
	std::cout << time2 / 10 << " seconds" << std::endl;
	std::cout << time3 / 10 << " seconds" << std::endl;

    return 0;
}