#pragma once


struct TMaxGainResult
{
    float Gain = 0;
    float Mult = 0;
};

TMaxGainResult CalcMaxGain(const TVector<float> &gainArr);
