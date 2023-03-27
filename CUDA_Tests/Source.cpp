#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <vector>
#include <cmath>

class Circle {
public:
	olc::vf2d pos;
    float radius;
	Circle(float x, float y, float r) : pos(x, y), radius(r) {}
};

class ScrollAndPanProgram : public olc::PixelGameEngine
{
public:
    ScrollAndPanProgram() 
    {
        sAppName = "Scroll and Pan Program";
    }
    
    bool OnUserCreate() override
    {
		zoomLevel = 1.0f;
		offset = { 0.0f, 0.0f };
		halfScreen = { ScreenWidth() * 0.5f, ScreenHeight() * 0.5f };
        
		circles.emplace_back(100, 100, 10);
		circles.emplace_back(-200, 200, 20);
		circles.emplace_back(300, -100, 15);
		circles.emplace_back(-150, -250, 25);

		return true;
    }
    
    bool OnUserUpdate(float fElapsedTime) override
    {
		Controls(fElapsedTime);
		Render();
        
        return true;
    }
    
private:
	static constexpr float zoomSpeed = 4.0f;
    static constexpr float panSpeed = 400.0f;

    float zoomLevel;
	olc::vf2d offset;
	olc::vf2d halfScreen;
    std::vector<Circle> circles;

    void Controls(float fElapsedTime)
    {
        if (GetKey(olc::Key::Q).bHeld) zoomLevel *= std::pow(zoomSpeed, fElapsedTime);
        if (GetKey(olc::Key::E).bHeld) zoomLevel *= std::pow(zoomSpeed, -fElapsedTime);
		if (GetKey(olc::Key::A).bHeld) offset.x += panSpeed * fElapsedTime / zoomLevel;
		if (GetKey(olc::Key::D).bHeld) offset.x -= panSpeed * fElapsedTime / zoomLevel;
		if (GetKey(olc::Key::W).bHeld) offset.y += panSpeed * fElapsedTime / zoomLevel;
		if (GetKey(olc::Key::S).bHeld) offset.y -= panSpeed * fElapsedTime / zoomLevel;
    }

    void Render()
    {
		Clear(olc::BLACK);
        for (auto& circle : circles)
        {
			olc::vf2d pos = (circle.pos + offset) * zoomLevel + halfScreen;
			float radius = circle.radius * zoomLevel;
			if (pos.x + radius > 0 && pos.x - radius < ScreenWidth() && pos.y + radius > 0 && pos.y - radius < ScreenHeight())
				DrawCircle(pos, radius, olc::WHITE);
        }
    }
};

int main()
{
    ScrollAndPanProgram program;
    if (program.Construct(800, 600, 1, 1))
    {
        program.Start();
    }
    return 0;
}
