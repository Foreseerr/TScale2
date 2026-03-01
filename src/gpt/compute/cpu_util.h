#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void PrintArr(int t, const TArray2D<T> &arr)
{
    for (int k = 0; k < arr.GetXSize(); ++k) {
        printf("cpu vec[%g] = %g\n", k * 1., arr[t][k] * 1.);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
template <class T1, class T2>
void CopyMatrix(TVector<TVector<T1>> *p, const TArray2D<T2> &src)
{
    p->resize(src.GetYSize());
    for (yint y = 0; y < src.GetYSize(); ++y) {
        (*p)[y].resize(src.GetXSize());
        for (yint x = 0; x < src.GetXSize(); ++x) {
            (*p)[y][x] = src[y][x];
        }
    }
}


template <class T1, class T2>
void InitDeltaMatrix(TArray2D<T1> *p, const TArray2D<T2> &src)
{
    p->SetSizes(src.GetXSize(), src.GetYSize());
    p->FillZero();
}


template <class T1, class T2, class T3>
void AddScaledMatrix(TArray2D<T1> *p, const TArray2D<T2> &src, T3 scale)
{
    yint xSize = src.GetXSize();
    yint ySize = src.GetYSize();
    Y_ASSERT(p->GetXSize() == xSize);
    Y_ASSERT(p->GetYSize() == ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] += src[y][x] * scale;
        }
    }
}


template <class T1, class T2>
void ScaleMatrix(TArray2D<T1> *p, T2 scale)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] *= scale;
        }
    }
}


template <class T>
void MatrixTestNan(const TArray2D<T> &arr)
{
    yint xSize = arr.GetXSize();
    yint ySize = arr.GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            float val = arr[y][x];
            Y_ASSERT(!isnan(val) && isfinite(val));
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// linear algebra

// resArr = kqv @ vecArr
TArray2D<float> MulForward(const TArray2D<float> &vecArr, const TArray2D<float> &kqv);
void MulBackwardWithAccum(TArray2D<float> *pVecArrGrad, const TArray2D<float> &kqv, const TArray2D<float> &resArrGrad);
void SumRankOne(const TArray2D<float> &vecArr, TArray2D<float> *pDelta, const TArray2D<float> &resArrGrad);

template <class T>
void MatmulBackprop(T transform, TArray2D<float> &src, TArray2D<float> &dstGrad, bool updateMatrix, TArray2D<float> *pSrcGrad)
{
    MulBackwardWithAccum(pSrcGrad, GetData(transform), dstGrad);
    if (updateMatrix) {
        TArray2D<float> delta;
        SumRankOne(src, &delta, dstGrad);
        transform->GetHostCompute()->HostApplyDelta(delta);
    }
}

template <class T>
void MatmulBackpropNoScale(T transform, TArray2D<float> &src, TArray2D<float> &dstGrad, bool updateMatrix, TArray2D<float> *pSrcGrad)
{
    MulBackwardWithAccum(pSrcGrad, GetDataNoScale(transform), dstGrad);
    if (updateMatrix) {
        TArray2D<float> delta;
        SumRankOne(src, &delta, dstGrad);
        transform->GetHostCompute()->HostApplyDelta(delta);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
TArray2D<float> NormalizeState(const TArray2D<float> &state, yint headCount);
void NormalizeStateBackward(const TArray2D<float> &state, yint headCount, const TArray2D<float> &dNormState, TArray2D<float> *pGrad);

void LoLU(const TArray2D<float> &gate, const TArray2D<float> &vals, TArray2D<float> *p);
void BackpropLoLU(const TArray2D<float> &gate, const TArray2D<float> &vals, const TArray2D<float> &grad, TArray2D<float> *gradGate, TArray2D<float> *gradVals);


///////////////////////////////////////////////////////////////////////////////////////////////////
void CrossShuffleForward(TArray2D<float> *p, const TVector<int> &fwdShuffle);
void CrossShuffleBackward(TArray2D<float> *p, const TVector<int> &fwdShuffle);
