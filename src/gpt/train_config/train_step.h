#pragma once


struct TTrainingStep
{
    float Rate = 0.01f;
    float L2Reg = 0;
    float Beta1 = 0;
    float Weight0 = 1;
    float Weight1 = 0;
    float DispDecay = 0.9f;

    TTrainingStep() {}
    TTrainingStep(float rate) : Rate(rate) {}
    void ScaleRate(float x)
    {
        Rate *= x;
    }
    void DisableMovingAverage()
    {
        Weight0 = 1;
        Weight1 = 0;
        Beta1 = 0;
        DispDecay = 0;
    }
};


#define GetShrinkMult(rate, l2reg) (1 - (rate) * (l2reg))
