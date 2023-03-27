#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <vector>
#include <memory>

class Element
{
public:
	olc::PixelGameEngine* pge;
	Element(olc::PixelGameEngine* p) : pge(p) {}
	
	virtual void Render(const olc::vf2d& position, const olc::vf2d& size, const float& zoomLevel, const olc::vf2d& offset, const olc::vf2d& halfScreen) = 0;
};

class HueSlider : public Element
{
public:
	float hue;

	HueSlider(olc::PixelGameEngine* p) : Element(p), hue(0.0f) {}

	void Render(const olc::vf2d& position, const olc::vf2d& size, const float& zoomLevel, const olc::vf2d& offset, const olc::vf2d& halfScreen)
	{
		pge->FillRectDecal((position + offset) * zoomLevel + halfScreen, size * zoomLevel, olc::PixelF(hue, 1.0f, 1.0f));
	}
};

class Form
{
public:
	olc::PixelGameEngine* pge;
	olc::vf2d position;
	olc::vf2d size;
	std::string formName;
    uint32_t numberOfElements;
	std::unique_ptr<std::unique_ptr<Element>[]> element;

	Form(olc::PixelGameEngine* p, uint32_t numberOfElements = 1) : pge(p), position(olc::vf2d(0, 0)), size(olc::vf2d(100, 100)), formName("Form"), numberOfElements(numberOfElements)
	{
		element = std::make_unique<std::unique_ptr<Element>[]>(numberOfElements);

		for (uint32_t i = 0; i < numberOfElements; i++)
			element[i] = std::make_unique<HueSlider>(pge);
	}

	void Render(const float& zoomLevel, const olc::vf2d& offset, const olc::vf2d& halfScreen)
	{
		pge->FillRectDecal((position + offset) * zoomLevel + halfScreen, size * zoomLevel, olc::WHITE);
	}
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
        
		forms.push_back(std::make_shared<Form>(this, 1));

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
	std::vector<std::shared_ptr<Form>> forms;

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
		for (auto& form : forms)
		{
			form->Render(zoomLevel, offset, halfScreen);
		}
        
		olc::vf2d stringPos = (olc::vf2d(100, 100) + offset) * zoomLevel + halfScreen;
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
