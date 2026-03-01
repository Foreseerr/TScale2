#pragma once
#include "delta.h"
#include <gpt/model_params/model_matrix.h>
#include <gpt/train_config/train_step.h>
#include <lib/cuda/cuda_arrays.h>


template <class T>
inline void CopyHost(NCuda::TCuda2DArray<T> *p, const THost2DPtr<T> &src)
{
    Y_VERIFY(p->GetXSize() == src.GetXSize() && p->GetYSize() == src.GetYSize());
    PutHost(p, src);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class THostModelMatrix
{
    NCuda::TCuda2DArray<float> Matr;
    NCuda::TCuda2DArray<float> AvrgDelta1; // single EMA is good enough
    float SumWeight = 0;
    TVector<float> RowDisp;
    float Sum2 = 0;
    float Sparsity = 0;

public:
    void Create(yint xSize, yint ySize, float sparsity)
    {
        Matr.AllocateHost(xSize, ySize);
        AvrgDelta1.AllocateHost(xSize, ySize);
        Sparsity = sparsity;
    }
    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    yint GetRowDispSize() const { return YSize(RowDisp); }

    // direct data access
    const THost2DPtr<float> GetWeights() const { return Matr.GetHostPtr(); }
    const TVector<float> &GetRowDisp() const { return RowDisp; }
    float GetSumWeight() const { return SumWeight; }
    float GetSum2() const { return Sum2; }
    float GetSparsity() const { return Sparsity; }
    NCuda::TCuda2DArray<float> &GetCudaMatr() { return Matr; }
    NCuda::TCuda2DArray<float> &GetCudaGrad() { return AvrgDelta1; }
    void SetRowDisp(const TVector<float> &rowDisp, float sumWeight)
    {
        RowDisp = rowDisp;
        SumWeight = sumWeight;
    }
    void OnDataUpdate() { Sum2 = CalcMatrixSum2(Matr.GetHostPtr()); }

    // Set/Get ops
    void GetData(TModelMatrix *p) const //
    {
        p->Create(Matr.GetHostPtr(), AvrgDelta1.GetHostPtr(), SumWeight, RowDisp, Sparsity);
    }
    void SetData(const TModelMatrix &data)
    {
        CopyHost(&Matr, data.GetMatrix().GetHostPtr());
        CopyHost(&AvrgDelta1, data.GetGrad1().GetHostPtr());
        RowDisp = data.GetRowDisp();
        SumWeight = data.GetSumWeight();
        Sparsity = data.GetSparsity();
        OnDataUpdate();
    }

    // delta ops
    void AddDelta(const TModelMatrixHalfDelta &delta, const TTrainingStep &step);
    bool AddBitDelta(const TModelMatrixBitDelta &bitDelta, const TTrainingStep &step);
    void CompressDelta(const TModelMatrixHalfDelta &delta, TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail);
};
