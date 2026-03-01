#pragma once
#include "model_dim.h"
#include "model_matrix.h"
#include <lib/random/xrng.h>


struct TModelParams
{
    struct TAttentionMatrices
    {
        TVector<TModelMatrix> MatrArr;
        SAVELOAD(MatrArr);
    };
    TModelDescr ModelDescr;
    TVector<TAttentionMatrices> LayerArr;
    TVector<TModelMatrix> MatrArr;
    TVector<float> Bias;
    SAVELOAD(ModelDescr, LayerArr, MatrArr, Bias);

    TModelDescr GetModelDescr() const
    {
        Y_ASSERT(ModelDescr.Dims.Dim == MatrArr[MP_MODEL_EMBED].GetXSize());
        Y_ASSERT(ModelDescr.LabelCount == MatrArr[MP_MODEL_EMBED].GetYSize());
        Y_ASSERT(ModelDescr.OutputTokenCount == MatrArr[MP_MODEL_FINAL].GetYSize());
        Y_ASSERT(YSize(ModelDescr.LayerArr) == YSize(LayerArr));
        return ModelDescr;
    }
    template <class T>
    void ForEachModelMatrix(T func)
    {
        for (TModelMatrix &mm : MatrArr) {
            func(mm);
        }
        for (TAttentionMatrices &att : LayerArr) {
            for (TModelMatrix &mm : att.MatrArr) {
                func(mm);
            }
        }
    }
    void ResetGrad(EModelMatrixReset rr)
    {
        ForEachModelMatrix([&](TModelMatrix &mm) { mm.ResetGrad(rr); });
    }
    void ScaleGrad(float x)
    {
        ForEachModelMatrix([&](TModelMatrix &mm) { mm.ScaleGrad(x); });
    }
    bool IsEmpty() const { return Bias.empty(); }
};


struct TModelParamsHolder : public TThrRefBase
{
    TModelParams Params;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void InitModel(TModelParams *pParams, TXRng &rng, const TModelDescr &modelDescr, ECombinerInit combinerInit, const TVector<float> &biasArr);
void ReplaceHead(TModelParams *pParams, TXRng &rng, const TVector<float> &biasArr);

yint CountModelSize(const TModelParams &params);
yint CountActiveModelSize(const TModelParams &params);

void BackgroundSaveModel(TPtrArg<TModelParamsHolder> params, const TString &fname);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAllModelMatrices
{
    TVector<TArray2D<float>> MatrArr;
    SAVELOAD(MatrArr);

public:
    TAllModelMatrices() { MatrArr.reserve(10000); }
    void Clear()
    {
        MatrArr.resize(0);
    }
    bool IsEmpty() const
    {
        return MatrArr.empty();
    }
    void AddMatrix(yint xSize, yint ySize)
    {
        TArray2D<float> &mm = *MatrArr.insert(MatrArr.end());
        mm.SetSizes(xSize, ySize);
        mm.FillZero();
    }
    void FillZero()
    {
        for (auto &mm : MatrArr) {
            mm.FillZero();
        }
    }
    void AddScaled(const TAllModelMatrices &arg, float scale)
    {
        Y_VERIFY(YSize(MatrArr) == YSize(arg.MatrArr));
        for (yint k = 0; k < YSize(MatrArr); ++k) {
            AddScaledMatrixAligned(&MatrArr[k], arg.MatrArr[k], scale);
        }
    }
    void Scale(float scale)
    {
        for (yint k = 0; k < YSize(MatrArr); ++k) {
            ScaleMatrixAligned(MatrArr[k].GetHostPtr(), scale);
        }
    }
};


void InitZero(TAllModelMatrices *p, const TModelDescr &descr);
void GetMatrices(TAllModelMatrices *p, const TModelParams &params);
void SetMatrices(TModelParams *p, const TAllModelMatrices &params);
void AddMatrices(TModelParams *p, const TAllModelMatrices &params, float scale);
void GetGradient(TAllModelMatrices *p, const TModelParams &params);
void SetGradient(TModelParams *p, const TAllModelMatrices &params);


///////////////////////////////////////////////////////////////////////////////////////////////////
void PackMatrices(TBufferedStream &f, const TModelParams &params);
void PackMatrices(TBufferedStream &f, const TAllModelMatrices &params);
void AddPackedMatrices(TAllModelMatrices *p, TBufferedStream &f, float scale);
void AddPackedMatrices(TModelParams *p, TBufferedStream &f, float scale);


///////////////////////////////////////////////////////////////////////////////////////////////////
void GetRowDisp(TModelRowDisp *p, const TModelParams &params);
void SetRowDisp(TModelParams *p, const TModelRowDisp &rd);
