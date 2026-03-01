#pragma once
#include "base_host.h"
#include "acpu_base.h"
#include "delta_cuda.h"
#include "host_matrix.h"
#include "quant.h"
#include <lib/cuda/cuda_arrays.h>


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline void CopyValue(TCudaPOD<T> *p, const TCudaPOD<T> &src)
{
    *p->GetDevicePtr() = *src.GetDevicePtr();
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// simulation model params quantization
template <class TMatrixFloat>
class TAcpuHostModelMatrix : public IModelMatrixBase<TMatrixFloat>, public TMatrixBaseOps, public IModelMatrixHostCompute
{
    using IModelMatrixBase<TMatrixFloat>::Matr;

    struct TDeviceData : public TThrRefBase
    {
        TCudaPackedDeltaMatrix Delta;
        TCudaPOD<int> CudaLaunchFlag;
        TCudaPOD<int> CudaAllowDelayedFlag;

        TDeviceData() : CudaLaunchFlag(0, null_ptr_arg, 0), CudaAllowDelayedFlag(0, null_ptr_arg, 0) {}
    };

private:
    TQuantizedMatrix<TMatrixFloat> Quant;
    TVector<TIntrusivePtr<TDeviceData>> DeviceArr;
    TModelMatrixHalfDelta SumDelta;
    volatile bool HasDelta = false;
    TModelMatrixBitDelta BitDelta;
    bool DelayGradient = false;
    TCudaPOD<int> CurrentIteration;
    EModelMatrixMemory MMM;

private:
    // base functions
    void AttachOp(TCudaPOD<int> currentIteration, const TVector<TCudaPOD<int>> &cudaDeltaFlags,
        const TVector<TCudaPOD<int>> &cudaAllowDelayedFlags) override
    {
        CurrentIteration = currentIteration;
        Y_ASSERT(YSize(cudaDeltaFlags) == YSize(DeviceArr));
        for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
            TDeviceData &dev = *DeviceArr[deviceId];
            Y_ASSERT(dev.CudaLaunchFlag.GetOwner() == 0);
            Y_ASSERT(dev.CudaAllowDelayedFlag.GetOwner() == 0);
            dev.CudaLaunchFlag = cudaDeltaFlags[deviceId];
            dev.CudaAllowDelayedFlag = cudaAllowDelayedFlags[deviceId];
        }
    }

    bool IsDelayGradient() const override { return DelayGradient; }
    void AddDeviceToSumDelta(yint deviceId) override
    {
        THostPackedDeltaPtr delta = DeviceArr[deviceId]->Delta.GetDelta();
        if (HasDelta) {
            Add(&SumDelta, delta);
        } else {
            Copy(&SumDelta, delta);
        }
        HasDelta = true;
    }
    void AddDelta(const TTrainingStep &step) override
    {
        Y_ASSERT(HasDelta);
        Matr.AddDelta(SumDelta, step);
        Convert();
        HasDelta = false;
    }

    bool AddBitDelta(const TTrainingStep &step) override { return Matr.AddBitDelta(BitDelta, step); }

    void Convert() override { Quant.Convert(Matr); }

    // IBitDelta
    yint GetBitDeltaXSize() const override { return IModelMatrixBase<TMatrixFloat>::GetXSize(); }
    yint GetBitDeltaYSize() const override { return IModelMatrixBase<TMatrixFloat>::GetYSize(); }
    TModelMatrixBitDelta &GetBitDelta() override { return BitDelta; }
    void ExtractDelta(TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail) override
    {
        Matr.CompressDelta(SumDelta, pBitDelta, pDeltaTail);
        HasDelta = false;
    }

public:
    TAcpuHostModelMatrix() : CurrentIteration(0, null_ptr_arg, 0) {}

    void Create(yint deviceCount, TPtrArg<THostModelMatrixScale> pScale, float discrScale, const TModelMatrix &data,
        EModelMatrixQuant quant, EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm)
    {
        IModelMatrixBase<TMatrixFloat>::Create(data);
        yint xSize = data.GetXSize();
        yint ySize = data.GetYSize();
        SumDelta.Init(xSize, ySize);
        DelayGradient = (delayGrad == MM_DELAY_GRADIENT);
        MMM = mmm;
        DeviceArr.resize(deviceCount);
        for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
            DeviceArr[deviceId] = new TDeviceData;
            TDeviceData &dev = *DeviceArr[deviceId];
            dev.Delta.AllocateHost(xSize, ySize);
        }
        Quant.Create(xSize, ySize, quant, discrScale, pScale);
        Convert();
    }

    EModelMatrixMemory GetMemoryType() const { return MMM; }

    // packed delta access
    TCudaPackedDeltaMatrix &GetHostDelta(yint deviceId) { return DeviceArr[deviceId]->Delta; }
    TCudaPOD<int> GetLaunchFlag(yint deviceId) const { return DeviceArr[deviceId]->CudaLaunchFlag; }
    TCudaPOD<int> GetAllowDelayedFlag(yint deviceId) const { return DeviceArr[deviceId]->CudaAllowDelayedFlag; }

    // quant access
    TCuda2DArray<TMatrixFloat> &GetFastHost() { return Quant.GetFastHost(); }
    yint GetMatrixScaleIndex() const { return Quant.GetMatrixScaleIndex(); }

    // cpu transformer, we expect these functions are used with cpu transformer only, so we write to cuda flags from cpu (does not work if
    // cpu&gpu write there concurrently)
    IModelMatrixHostCompute *GetHostCompute() override { return this; }
    void GetFastFloatData(TArray2D<float> *p, bool noScale) const override { Quant.GetFastFloatData(p, noScale); }
    void HostApplyDelta(const TArray2D<float> &data) override
    {
        Y_VERIFY(YSize(DeviceArr) == 1);
        THostPackedDeltaPtr delta = DeviceArr[0]->Delta.GetDelta();
        Compress(&delta, data);
        CopyValue(&DeviceArr[0]->CudaLaunchFlag, CurrentIteration);
    }
    void HostAllowDelayedUpdates() override
    {
        Y_VERIFY(YSize(DeviceArr) == 1);
        CopyValue(&DeviceArr[0]->CudaAllowDelayedFlag, CurrentIteration);
    }
};

}
