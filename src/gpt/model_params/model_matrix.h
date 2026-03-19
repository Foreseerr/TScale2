#pragma once
#include <gpt/model_params/sse_utils.h>


void PackMatrix(TBufferedStream &f, const THost2DPtr<float> &matr);
void AddPackedMatrix(THost2DPtr<float> matr, TBufferedStream &f, float scale);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelRowDisp
{
    TVector<float> RowDisp;
    float SumWeight = 0;
    SAVELOAD(RowDisp, SumWeight);

public:
    void AddScaled(TModelRowDisp &arg, float scale)
    {
        if (RowDisp.empty()) {
            *this = arg;
            Scale(scale);
            return;
        }
        Y_VERIFY(YSize(RowDisp) == YSize(arg.RowDisp));
        for (yint k = 0; k < YSize(RowDisp); ++k) {
            RowDisp[k] += arg.RowDisp[k] * scale;
        }
        SumWeight += arg.SumWeight * scale;
    }
    void Scale(float scale)
    {
        for (float &val : RowDisp) {
            val *= scale;
        }
        SumWeight *= scale;
    }
    void Clear()
    {
        RowDisp.resize(0);
        SumWeight = 0;
    }
    void AddMatrixRowDisp(const TVector<float> &mmRowDisp, float mmSumWeight)
    {
        float mult = (mmSumWeight > 0) ? 1 / mmSumWeight : 0;
        for (float val : mmRowDisp) {
            RowDisp.push_back(val * mult);
        }
        SumWeight = Max<float>(SumWeight, mmSumWeight);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EModelMatrixReset {
    MM_RESET_GRAD = 1,
    MM_RESET_ROW_DISP = 2,
    MM_RESET_GRAD_AND_ROW_DISP = 3,
};


class TModelMatrix
{
    TArray2D<float> Matr;
    TArray2D<float> Grad1;
    float SumWeight = 0;
    TVector<float> RowDisp;
    float Sparsity = 0;

public:
    SAVELOAD(Matr, Grad1, SumWeight, RowDisp, Sparsity);

private:
    void ClearRowDisp()
    {
        SumWeight = 0;
        ClearPodArray(&RowDisp, YSize(RowDisp));
    }

public:
    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    const TVector<float> &GetRowDisp() const { return RowDisp; }
    float GetSumWeight() const { return SumWeight; }
    float GetSparsity() const { return Sparsity; }
    TArray2D<float> &GetMatrix() { return Matr; }
    TArray2D<float> &GetGrad1() { return Grad1; }
    const TArray2D<float> &GetMatrix() const { return Matr; }
    const TArray2D<float> &GetGrad1() const { return Grad1; }

    void SetRowDisp(float sumWeight, const TVector<float> &rowDisp)
    {
        SumWeight = sumWeight;
        RowDisp = rowDisp;
    }

    // set ops
    void SetMatrix(const TArray2D<float> &data)
    {
        Y_ASSERT(Matr.GetXSize() == data.GetXSize());
        Y_ASSERT(Matr.GetYSize() == data.GetYSize());
        Matr = data;
        Grad1.FillZero();
        ClearRowDisp();
    }

    void Create(const THost2DPtr<float> &data, const THost2DPtr<float> &grad1, float sumWeight, const TVector<float> &rowDisp, float sparsity)
    {
        Matr.Assign(data);
        Grad1.Assign(grad1);
        SumWeight = sumWeight;
        RowDisp = rowDisp;
        Sparsity = sparsity;
    }

    // resize
    void SetSizes(yint xSize, yint ySize, yint rowDispSize, float sparsity)
    {
        Matr.SetSizes(xSize, ySize);
        Matr.FillZero();
        Grad1.SetSizes(xSize, ySize);
        Grad1.FillZero();
        SumWeight = 0;
        ClearPodArray(&RowDisp, rowDispSize);
        Sparsity = sparsity;
    }
    void SetSizesPerMatrixDisp(yint xSize, yint ySize, float sparsity) { SetSizes(xSize, ySize, 1, sparsity); }
    void SetSizesPerRowDisp(yint xSize, yint ySize, float sparsity) { SetSizes(xSize, ySize, ySize, sparsity); }

    // reset grad
    void ResetGrad(EModelMatrixReset rr)
    {
        if (rr & MM_RESET_GRAD) {
            Grad1.FillZero();
        }
        if (rr & MM_RESET_ROW_DISP) {
            SumWeight = 0;
            ClearPodArray(&RowDisp, YSize(RowDisp));
        }
    }
    void ScaleGrad(float x)
    {
        ScaleMatrixAligned(Grad1.GetHostPtr(), x);
    }
};
