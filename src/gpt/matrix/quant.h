#pragma once
#include "matrix_scale.h"
#include "host_matrix.h"
#include <gpt/model_params/model_dim.h>
#include <immintrin.h>
#include <lib/cuda/cuda_arrays.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
void ConvertToFastMatrixFloat(i8 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant);
void ConvertToFastMatrixFloat(half *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant);
void ConvertToFastMatrixFloat(e4m3 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant);


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace NCuda
{

template <class TMatrixFloat>
class TQuantizedMatrix 
{
    TCuda2DArray<TMatrixFloat> FastHost;
    EModelMatrixQuant Quantization;
    float DiscrScale = 0;
    TIntrusivePtr<THostModelMatrixScale> MatrixScale;
    yint MatrixScaleIndex = 0;

private:
    void SetDiscrScale(float discrScale) { MatrixScale->SetScale(MatrixScaleIndex, discrScale); }
    float GetMatrixScaleValue() const { return MatrixScale->GetScale(MatrixScaleIndex); }

public:
    void Create(yint xSize, yint ySize, EModelMatrixQuant quant, float discrScale, TPtrArg<THostModelMatrixScale> pScale)
    {
        FastHost.AllocateHost(xSize, ySize);
        Quantization = quant;
        DiscrScale = discrScale;
        MatrixScale = pScale.Get();
        MatrixScaleIndex = pScale->AddScale(0);
    }

    void Convert(const THostModelMatrix &matr)
    {
        yint xSize = matr.GetXSize();
        yint ySize = matr.GetYSize();
        Y_ASSERT(xSize == FastHost.GetXSize());
        Y_ASSERT(ySize == FastHost.GetYSize());

        float sko = sqrt(matr.GetSum2() / (xSize * ySize));
        float discrScale = sko * DiscrScale;
        __m256 mult = _mm256_set1_ps((sko == 0) ? 0 : (1 / discrScale));

        SetDiscrScale(sko);

        TMemoryBlob fastMem = FastHost.GetHostMem();
        const THost2DPtr<float> weights = matr.GetWeights();
        for (yint y = 0; y < ySize; ++y) {
            TMatrixFloat *dst = fastMem.GetElementAddress<TMatrixFloat>(0, y);
            const float *src = weights.GetRow(y);
            ConvertToFastMatrixFloat(dst, src, mult, xSize, Quantization);
        }
    }

    void GetFastFloatData(TArray2D<float> *p, bool noScale) const
    {
        yint xSize = FastHost.GetXSize();
        yint ySize = FastHost.GetYSize();
        THost2DPtr<TMatrixFloat> src = FastHost.GetHostPtr();
        float scale = DiscrScale;
        if (!noScale) {
            scale *= GetMatrixScaleValue();
        }
        p->SetSizes(xSize, ySize);
        for (yint y = 0; y < ySize; ++y) {
            for (yint x = 0; x < xSize; ++x) {
                (*p)[y][x] = float(src[y][x]) * scale;
            }
        }
    }

    yint GetMatrixScaleIndex() const { return MatrixScaleIndex; }
    TCuda2DArray<TMatrixFloat> &GetFastHost() { return FastHost; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
float ConvertMatrixToFastFloat(float matrixDiscrScale, const TArray2D<float> matr, EModelMatrixQuant quant, TCuda2DArray<TMatrixFloat> *p)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();

    float sum2 = CalcMatrixSum2(matr);
    float sko = sqrt(sum2 / (xSize * ySize));
    float discrScale = sko * matrixDiscrScale;
    __m256 mult = _mm256_set1_ps((sko == 0) ? 0 : (1 / discrScale));

    TMemoryBlob fastMem = p->GetHostMem();
    for (yint y = 0; y < ySize; ++y) {
        TMatrixFloat *dst = fastMem.GetElementAddress<TMatrixFloat>(0, y);
        const float *src = matr.GetRow(y);
        ConvertToFastMatrixFloat(dst, src, mult, xSize, quant);
    }
    return sko;
}


}
