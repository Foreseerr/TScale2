#include "cpu_util.h"
#include <gpt/model_params/model_dim.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
// resArr = kqv @ vecArr
TArray2D<float> MulForward(const TArray2D<float> &vecArr, const TArray2D<float> &kqv)
{
    yint len = vecArr.GetYSize();
    yint dim = vecArr.GetXSize();
    yint rDim = kqv.GetYSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    TArray2D<float> resArr;
    resArr.SetSizes(rDim, len);
    float normScale = CalcDotScale(dim);
    for (yint t = 0; t < len; ++t) {
        for (yint k = 0; k < rDim; ++k) {
            float res = 0;
            for (yint x = 0; x < dim; ++x) {
                res += vecArr[t][x] * kqv[k][x];
            }
            resArr[t][k] = res * normScale;
        }
    }
    return resArr;
}


void MulBackwardWithAccum(TArray2D<float> *pVecArrGrad, const TArray2D<float> &kqv, const TArray2D<float> &resArrGrad)
{
    yint len = resArrGrad.GetYSize();
    yint dim = kqv.GetXSize();
    yint rDim = resArrGrad.GetXSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    float normScale = CalcDotScale(dim);
    for (yint t = 0; t < len; ++t) {
        for (yint x = 0; x < dim; ++x) {
            float res = 0;
            for (yint k = 0; k < rDim; ++k) {
                res += resArrGrad[t][k] * kqv[k][x];
            }
            (*pVecArrGrad)[t][x] += res * normScale;
        }
    }
}


void SumRankOne(const TArray2D<float> &vecArr, TArray2D<float> *pDelta, const TArray2D<float> &resArrGrad)
{
    yint len = vecArr.GetYSize();
    yint dim = vecArr.GetXSize();
    yint rDim = resArrGrad.GetXSize();
    Y_ASSERT(len == resArrGrad.GetYSize());
    pDelta->SetSizes(dim, rDim);
    pDelta->FillZero();
    for (yint k = 0; k < rDim; ++k) {
        for (yint x = 0; x < dim; ++x) {
            float res = 0;
            for (yint t = 0; t < len; ++t) {
                res += resArrGrad[t][k] * vecArr[t][x];
            }
            (*pDelta)[k][x] += res;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
TArray2D<float> NormalizeState(const TArray2D<float> &state, yint headCount)
{
    yint len = state.GetYSize();
    yint dim = state.GetXSize();
    yint headDim = dim / headCount;
    TArray2D<float> res;
    res.SetSizes(dim, len);
    for (yint t = 0; t < len; ++t) {
        for (yint h = 0; h < headCount; ++h) {
            yint offset = h * headDim;
            float sum2 = 0;
            for (yint x = 0; x < headDim; ++x) {
                sum2 += Sqr(state[t][offset + x]);
            }
            if (sum2 == 0) {
                for (yint x = 0; x < headDim; ++x) {
                    res[t][offset + x] = 0;
                }
            } else {
                float scale = sqrt(headDim / sum2);
                for (yint x = 0; x < headDim; ++x) {
                    res[t][offset + x] = state[t][offset + x] * scale;
                }
            }
        }
    }
    return res;
}


void NormalizeStateBackward(const TArray2D<float> &state, yint headCount, const TArray2D<float> &dNormState, TArray2D<float> *pGrad)
{
    yint len = state.GetYSize();
    yint dim = state.GetXSize();
    yint headDim = dim / headCount;
    pGrad->SetSizes(dim, len);
    for (yint t = 0; t < len; ++t) {
        for (yint h = 0; h < headCount; ++h) {
            yint offset = h * headDim;
            float sum2 = 0;
            float dp = 0;
            for (yint x = 0; x < headDim; ++x) {
                float src = state[t][offset + x];
                float grad = dNormState[t][offset + x];
                sum2 += Sqr(src);
                dp += src * grad;
            }
            if (sum2 == 0) {
                for (yint x = 0; x < headDim; ++x) {
                    (*pGrad)[t][offset + x] = 0;
                }
            } else {
                float sigma = dp / sum2;
                float scale = sqrt(headDim / sum2);
                for (yint x = 0; x < headDim; ++x) {
                    float src = state[t][offset + x];
                    float grad = dNormState[t][offset + x];
                    (*pGrad)[t][offset + x] = scale * (grad - src * sigma);
                }
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
inline float CpuComputeLolu(float x)
{
    float arg = x * LOLU_SCALE;
    if (arg < -20) {
        return 0;
    } else if (arg > 20) {
        return 1;
    } else {
        float w = expf(arg);
        return w / (1 + w);
    }
}


void LoLU(const TArray2D<float> &gate, const TArray2D<float> &vals, TArray2D<float> *p)
{
    yint xSize = vals.GetXSize();
    yint ySize = vals.GetYSize();
    Y_ASSERT(xSize == gate.GetXSize());
    Y_ASSERT(ySize == gate.GetYSize());
    p->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            float mult = CpuComputeLolu(gate[y][x]);
            (*p)[y][x] = mult * vals[y][x];
        }
    }
}


void BackpropLoLU(const TArray2D<float> &gate, const TArray2D<float> &vals, const TArray2D<float> &grad, TArray2D<float> *gradGate, TArray2D<float> *gradVals)
{
    yint xSize = vals.GetXSize();
    yint ySize = vals.GetYSize();
    Y_ASSERT(xSize == gate.GetXSize());
    Y_ASSERT(ySize == gate.GetYSize());
    Y_ASSERT(xSize == grad.GetXSize());
    Y_ASSERT(ySize == grad.GetYSize());
    gradGate->SetSizes(xSize, ySize);
    gradVals->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            float mult = CpuComputeLolu(gate[y][x]);
            (*gradGate)[y][x] = grad[y][x] * vals[y][x] * mult * (1 - mult) * LOLU_SCALE;
            (*gradVals)[y][x] = grad[y][x] * mult;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void CrossShuffleForward(TArray2D<float> *p, const TVector<int> &fwdShuffle)
{
    yint ySize = p->GetYSize();
    yint xSize = p->GetXSize();
    TArray2D<float> tmp;
    tmp.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        yint srcY = fwdShuffle[y];
        memcpy(tmp.GetRow(y), p->GetRow(srcY), sizeof(tmp[0][0]) * xSize);
    }
    p->Swap(tmp);
}


void CrossShuffleBackward(TArray2D<float> *p, const TVector<int> &fwdShuffle)
{
    yint ySize = p->GetYSize();
    yint xSize = p->GetXSize();
    TArray2D<float> tmp;
    tmp.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        yint srcY = fwdShuffle[y];
        memcpy(tmp.GetRow(srcY), p->GetRow(y), sizeof(tmp[0][0]) * xSize);
    }
    p->Swap(tmp);
}
