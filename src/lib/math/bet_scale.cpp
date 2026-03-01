#include "bet_scale.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
static float CalcGain(const TVector<float> &gainArr, float mult)
{
    float res = 0;
    for (yint t = 0; t < YSize(gainArr); ++t) {
        float dx = gainArr[t];
        float delta = 1 + dx * mult;
        if (delta <= 0) {
            return -1e10f;
        }
        res += log(delta);
    }
    return res;
}

TMaxGainResult CalcMaxGain(const TVector<float> &gainArr)
{
    float minMult = 0;
    float minScore = -1e10f;
    float maxMult = 10000000;
    float maxScore = -1e10f;
    float midMult = (minMult + maxMult);
    float midScore = CalcGain(gainArr, midMult);
    while (midScore <= -1e10f) {
        midMult *= 0.5;
        midScore = CalcGain(gainArr, midMult);
    }
    for (yint iter = 0; iter < 50; ++iter) {
        if ((maxMult - midMult) > (midMult - minMult)) {
            float chk = (maxMult + midMult) * 0.5;
            float chkScore = CalcGain(gainArr, chk);
            if (chkScore > midScore) {
                minMult = midMult;
                minScore = midScore;
                midMult = chk;
                midScore = chkScore;
            } else {
                maxMult = chk;
                maxScore = chkScore;
            }
        } else {
            float chk = (minMult + midMult) * 0.5;
            float chkScore = CalcGain(gainArr, chk);
            if (chkScore > midScore) {
                maxMult = midMult;
                maxScore = midScore;
                midMult = chk;
                midScore = chkScore;
            } else {
                minMult = chk;
                minScore = chkScore;
            }
        }
    }
    TMaxGainResult res;
    res.Mult = midMult;
    res.Gain = midScore;
    return res;
}


