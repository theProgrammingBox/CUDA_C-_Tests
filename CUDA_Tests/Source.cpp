#include <iostream>

class Test
{
public:
	Test()
	{
		name = new std::string("Test");
	}
	virtual ~Test()
	{
		delete name;
	}

	void Print()
	{
		printf("%s\n", name->c_str());
	}
	
	std::string* name;
};

class Test2 : public Test
{
public:
	Test2()
	{
		name = new std::string("Test2");
		num = new std::string("123");
	}
	
	virtual ~Test2()
	{
		delete num;
	}

	void Print()
	{
		Test::Print();
		printf("%s\n", num->c_str());
	}
	
	std::string* num;
};

int main()
{
	Test2* test2 = new Test2();
	test2->Print();
	std::string* name = test2->name;
	delete test2;

	printf("Has name been deleted by abstract: %s\n", name->c_str());

	return 0;
}