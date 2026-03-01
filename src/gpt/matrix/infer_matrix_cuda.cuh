#pragma once
#include "matrix_scale.h"
#include <gpt/matrix/base_cuda.cuh>
#include <gpt/matrix/base_host.h>
#include <gpt/matrix/quant.h>
#include <gpt/model_params/model_dim.h>
#include <lib/cuda/cuda_arrays.h>


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
class TCudaInferModelMatrix : public TThrRefBase
{
    TIntrusivePtr<THostModelMatrixScale> HostMatrixScale;
    TIntrusivePtr<TCudaModelMatrixScale> CudaMatrixScale;
    TCuda2DArray<TMatrixFloat> FastDevice;
    yint ScaleIndex = 0;

public:
    TCudaInferModelMatrix(TStream &stream, TIntrusivePtr<THostModelMatrixScale> pHostMatrixScale,
        TIntrusivePtr<TCudaModelMatrixScale> pCudaMatrixScale, const TModelMatrix &matr, EModelMatrixQuant quant, EModelMatrixMemory mmm)
        : HostMatrixScale(pHostMatrixScale), CudaMatrixScale(pCudaMatrixScale)
    {
        yint xSize = matr.GetXSize();
        yint ySize = matr.GetYSize();
        if (mmm == MM_MEM_HOST) {
            FastDevice.AllocateHost(xSize, ySize);
        } else {
            FastDevice.Allocate(xSize, ySize);
        }
        float sko = ConvertMatrixToFastFloat(MODEL_DISCR_SCALE, matr.GetMatrix(), quant, &FastDevice);
        ScaleIndex = HostMatrixScale->AddScale(sko);
        if (mmm == MM_MEM_HOST) {
            return;
        }
        FastDevice.CopyToDevice(stream);
    }

    yint GetXSize() const { return FastDevice.GetXSize(); }
    yint GetYSize() const { return FastDevice.GetYSize(); }
    yint GetLocalXSize() const { return FastDevice.GetXSize(); }
    yint GetLocalYSize() const { return FastDevice.GetYSize(); }
    TCuda2DArray<TMatrixFloat> &GetFast() { return FastDevice; }
    TCudaPOD<float> GetScale() const { return CudaMatrixScale->GetElement(ScaleIndex); }
};
}
