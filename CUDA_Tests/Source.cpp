#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

float InvSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

olc::vf2d InvSqrt(olc::vf2d vec)
{
	return olc::vf2d(InvSqrt(vec.x), InvSqrt(vec.y));
}

class Example : public olc::PixelGameEngine
{
public:
	olc::vf2d parameters;
	olc::vf2d gradientMomentum;
	olc::vf2d gradientMean;
	olc::vf2d gradientVariance;
	float learningRate;
	float momentumBeta;
	float meanBeta;
	float varianceBeta;
	float decayedMeanBeta;
	float decayedVarianceBeta;
	
	Example()
	{
		sAppName = "Example";
	}
	
	bool OnUserCreate() override
	{
		parameters = { ScreenWidth() * 0.5f, ScreenHeight() * 0.5f };
		gradientMomentum = { 0.0f, 0.0f };
		gradientMean = { 0.0f, 0.0f };
		gradientVariance = { 0.0f, 0.0f };

		learningRate = 1.0f;
		momentumBeta = 0.9f;
		meanBeta = 0.9f;
		varianceBeta = 0.999f;
		decayedMeanBeta = 1.0f;
		decayedVarianceBeta = 1.0f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		olc::vf2d gradient = { (float)GetMouseX() - parameters.x, (float)GetMouseY() - parameters.y };
		if (GetKey(olc::Key::W).bHeld)
			gradient.y -= 0.2f;
		if (GetKey(olc::Key::S).bHeld)
			gradient.y += 0.2f;
		if (GetKey(olc::Key::A).bHeld)
			gradient.x -= 0.2f;
		if (GetKey(olc::Key::D).bHeld)
			gradient.x += 0.2f;
		
		{	// SGD
			//parameters += gradient * learningRate;
		}
		{	// momentum
			//gradientMomentum = momentumBeta * gradientMomentum + (1.0f - momentumBeta) * gradient;
			//parameters += gradientMomentum * learningRate;
		}
		{	// Adam
			decayedMeanBeta *= meanBeta;
			decayedVarianceBeta *= varianceBeta;
			gradientMean = meanBeta * gradientMean + (1.0f - meanBeta) * gradient;
			gradientVariance = varianceBeta * gradientVariance + (1.0f - varianceBeta) * gradient * gradient;
			olc::vf2d correctedMean = gradientMean / (1.0f - decayedMeanBeta);
			olc::vf2d correctedVarience = gradientVariance / (1.0f - decayedVarianceBeta);
			parameters += correctedMean * InvSqrt(correctedVarience) * learningRate;
		}
		
		Clear(olc::BLACK);
		DrawCircle(parameters, 10, olc::WHITE);
		
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