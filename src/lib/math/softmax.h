#pragma once


void CalcSoftmax(float *data, float valScale, yint sz);

inline void Softmax(TVector<float> *p)
{
    CalcSoftmax(p->data(), 1, YSize(*p));
}


inline float LogisticFloat(float x)
{
    if (x < -20) {
        return 0;
    } else if (x > 20) {
        return 1;
    } else {
        return 1 / (1 + expf(-x));
    }
}
