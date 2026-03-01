#include "softmax.h"


void CalcSoftmax(float *data, float valScale, yint sz)
{
    float maxVal = -1e10f;
    for (yint k = 0; k < sz; ++k) {
        maxVal = Max(maxVal, data[k]);
    }
    float sum = 0;
    for (yint k = 0; k < sz; ++k) {
        data[k] = expf(data[k] - maxVal);
        sum += data[k];
    }
    for (yint k = 0; k < sz; ++k) {
        data[k] /= sum;
    }
}
