#include "proximity.h"


void Shrink(float lambda, TVector<float> *p)
{
    for (float &x : *p) {
        if (x > 0) {
            x = Max<float>(0, x - lambda);
        } else {
            x = Min<float>(0, x + lambda);
        }
    }
}


void ShrinkToPrev(const TVector<float> &prevVec, float lambda, TVector<float> *p)
{
    yint sz = YSize(prevVec);
    Y_ASSERT(sz == YSize(*p));
    for (yint k = 0; k < sz; ++k) {
        float prev = prevVec[k];
        float &dst = (*p)[k];
        if (dst > prev + lambda) {
            dst -= lambda;
        } else if (dst < prev - lambda) {
            dst += lambda;
        } else {
            dst = prev;
        }
    }
}
