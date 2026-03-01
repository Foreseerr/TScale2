#pragma once
#include <lib/cuda/cuda_arrays.h>

namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
class THostModelMatrixScale : public TThrRefBase
{
    TCudaVector<float> MatrixScaleHost;
    int IndexCount = 0;

public:
    THostModelMatrixScale(yint sz);
    yint GetSize() { return MatrixScaleHost.GetSize(); }
    void SetScale(yint index, float val);
    float GetScale(yint index);
    int AddScale(float val);
    TCudaVector<float> &GetMatrixScaleHost() { return MatrixScaleHost; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaModelMatrixScale : public TThrRefBase
{
    TIntrusivePtr<THostModelMatrixScale> HostMatrixScale;
    TCudaVector<float> DeviceMatrixScale;

public:
    TCudaModelMatrixScale(TPtrArg<THostModelMatrixScale> pScale);
    //TIntrusivePtr<THostModelMatrixScale> GetHostMatrixScale() { return HostMatrixScale; }
    void CopyToDevice(TStream &stream)
    {
        Y_ASSERT(DeviceMatrixScale.GetSize() == HostMatrixScale->GetSize());
        TVector<float> scaleArr;
        GetAllData(HostMatrixScale->GetMatrixScaleHost(), &scaleArr);
        Put(stream, &DeviceMatrixScale, scaleArr);
    }
    template <class TGraph>
    void CopyToDevice(TPtrArg<TGraph> c)
    {
        c->KernelCopy(&DeviceMatrixScale, HostMatrixScale->GetMatrixScaleHost());
    }
    TCudaPOD<float> GetElement(yint idx) const { return DeviceMatrixScale.GetElement(idx); }
};

}
