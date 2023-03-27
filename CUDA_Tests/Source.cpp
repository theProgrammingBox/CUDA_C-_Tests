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
		lastMousePos = GetMousePos();
        
		circles.emplace_back(100, 100, 10);
		circles.emplace_back(-200, 200, 20);
		circles.emplace_back(300, -100, 15);
		circles.emplace_back(-150, -250, 25);

		return true;
    }
    
    bool OnUserUpdate(float fElapsedTime) override
    {
		Controls();
		Render();
        
        return true;
    }
    
private:
	static constexpr float zoomSpeed = 2.0f;
    static constexpr float panSpeed = 400.0f;

    float zoomLevel;
	olc::vf2d offset;
	olc::vf2d halfScreen;
	olc::vf2d lastMousePos;
    std::vector<Circle> circles;

    void Controls()
    {
		if (GetMouseWheel() > 0) zoomLevel *= std::pow(zoomSpeed, -0.1f);
		if (GetMouseWheel() < 0) zoomLevel *= std::pow(zoomSpeed, 0.1f);
		olc::vf2d delta = GetMousePos() - lastMousePos;
        if (GetMouse(1).bHeld) offset += delta / zoomLevel;
        lastMousePos = GetMousePos();
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
        
		olc::vf2d stringPos = (olc::vf2d(10, 10) + offset) * zoomLevel + halfScreen;
        float stringScale = 1.0f * zoomLevel;
		DrawStringDecal(stringPos, "Zoom: " + std::to_string(zoomLevel), olc::WHITE, { stringScale, stringScale });
    }
};

int main()
{
    ScrollAndPanProgram program;
	if (program.Construct(1440, 810, 1, 1))
    {
        program.Start();
    }
    return 0;
}
