#include "rope.h"


struct TRopeComputer
{
    TVector<float> FreqArr;

    TRopeComputer(yint width)
    {
        //float theta = 10000;
        float theta = 500000;
        for (yint k = 0; k < width / 2; ++k) {
            FreqArr.push_back(exp(-log(theta) * (k * 2. / width)));
        }
    }
    void Compute(yint t, yint width, float *buf)
    {
        for (yint k = 0; k < width; k += 2) {
            float angle = t * FreqArr[k / 2];
            buf[k] = cos(angle);
            buf[k + 1] = sin(angle);
        }
    }
};



void FillRopeBuf(TArray2D<float> *pRopeBuf, yint width, yint ropeLen)
{
    TRopeComputer rope(width);
    pRopeBuf->SetSizes(width, ropeLen);
    for (yint t = 0; t < ropeLen; ++t) {
        rope.Compute(t, width, pRopeBuf->GetRow(t));
    }
}


void FillRopeBuf(TArray2D<float> *pRopeBuf, yint width, const TVector<yint> tArr)
{
    TRopeComputer rope(width);
    yint sz = YSize(tArr);
    pRopeBuf->SetSizes(width, sz);
    for (yint y = 0; y < sz; ++y) {
        rope.Compute(tArr[y], width, pRopeBuf->GetRow(y));
    }
}


// reference ropeBuf application
template <class TFunc>
void ReferenceApplyRope(yint t, yint dim, yint headSize, const TArray2D<float> &ropeBuf, float ropeRotateDir, TFunc getElem)
{
    for (yint offset = 0; offset < dim; offset += headSize) {
        for (int k = 0; k < headSize; k += 2) {
            float &v0 = *getElem(offset + k);
            float &v1 = *getElem(offset + k + 1);
            float cosValue = ropeBuf[t][k];
            float sinValue = ropeBuf[t][k + 1] * ropeRotateDir;
            float r0 = v0 * cosValue - v1 * sinValue;
            float r1 = v0 * sinValue + v1 * cosValue;
            v0 = r0;
            v1 = r1;
        }
    }
}

void ApplyRope(const TArray2D<float> &ropeBuf, float ropeRotateDir, yint t, TVector<float> *p)
{
    yint dim = YSize(*p);
    yint headSize = ropeBuf.GetXSize();
    ReferenceApplyRope(t, dim, headSize, ropeBuf, ropeRotateDir, [&](yint k) { return &(*p)[k]; });
}

void ApplyRope(const TArray2D<float> &ropeBuf, float ropeRotateDir, TArray2D<float> *p)
{
    yint dim = p->GetXSize();
    yint headSize = ropeBuf.GetXSize();
    yint len = p->GetYSize();
    Y_ASSERT(len <= ropeBuf.GetYSize());
    for (yint t = 0; t < len; ++t) {
        ReferenceApplyRope(t, dim, headSize, ropeBuf, ropeRotateDir, [&](yint k) { return &(*p)[t][k]; });
    }
}
