#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

class Example : public olc::PixelGameEngine
{
public:
	olc::vf2d pos;
	olc::vf2d vel;
	//olc::vf2d mean;
	float learningRate;
	
	Example()
	{
		sAppName = "Example";
	}
	
	bool OnUserCreate() override
	{
		pos = { ScreenWidth() * 0.5f, ScreenHeight() * 0.5f };
		vel = { 0.0f, 0.0f };
		//mean = { 0.0f, 0.0f };
		learningRate = 1.0f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		olc::vf2d vel = { 0.0f, 0.0f };
		if (GetKey(olc::Key::W).bHeld)
			vel.y -= 1.0f;
		if (GetKey(olc::Key::S).bHeld)
			vel.y += 1.0f;
		if (GetKey(olc::Key::A).bHeld)
			vel.x -= 1.0f;
		if (GetKey(olc::Key::D).bHeld)
			vel.x += 1.0f;
		
		//mean = mean + (vel - mean);
		pos += vel * learningRate;
		
		Clear(olc::BLACK);
		DrawCircle(pos, 10, olc::WHITE);
		
		return true;
	}
};

int main()
{
	Example demo;
	if (demo.Construct(1280, 720, 1, 1))
		demo.Start();
	return 0;
}