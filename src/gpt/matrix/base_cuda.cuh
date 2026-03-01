#pragma once
#include "base_host.h"
#include "delta_cuda.h"
#include <lib/cuda/cuda_arrays.h>

namespace NCuda
{
class TGraph;
struct TCudaPackedDeltaMatrix;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ICopyModelParamsToHost : public TThrRefBase
{
    virtual void CopyModelParamsToHost() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
class ICudaModelMatrixBase : public ICopyModelParamsToHost
{
protected:
    yint DeviceId = 0;

public:
    ICudaModelMatrixBase(yint deviceId) : DeviceId(deviceId) {}

    // basic
    virtual yint GetLocalXSize() const = 0;
    virtual yint GetLocalYSize() const = 0;
    virtual void CopyToDevice(TPtrArg<TGraph> c) = 0;
    virtual void AddDelta(TPtrArg<TGraph> c, TCuda2DArray<float> &delta, EBackpropMode bm) = 0;
    virtual void AddDelta(TPtrArg<TGraph> c, TCudaPackedDeltaMatrix &delta, EBackpropMode bm) = 0;

    // host delta manipulation
    virtual TCudaPackedDeltaMatrix &GetHostDelta() = 0;
    virtual void AddHostDelta(TPtrArg<TGraph> c, EBackpropMode bm) = 0;
    virtual void AllowDelayedUpdates(TPtrArg<TGraph> c) = 0;

    // data
    virtual TCuda2DArray<TMatrixFloat> &GetFast() = 0;
    virtual TCudaPOD<float> GetScale() const = 0;
};

}
