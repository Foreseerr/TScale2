#pragma once
#include "acpu_cuda.cuh"
#include "acpu_process.h"
#include "base_host.h"
#include "matrix_scale.h"


using namespace NCuda;

class TCpuModelOps : public IModelOps
{
    TIntrusivePtr<TAcpuDeviceInfo> DeviceInfo;
    TIntrusivePtr<TCPUMatrixAdd> MatrixAdd;

private:
    template <class TMatrixFloat>
    TIntrusivePtr<IModelMatrixBase<TMatrixFloat>> CreateModelMatrixImpl(
        const TModelMatrix &data, EModelMatrixQuant quant, EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm)
    {
        TIntrusivePtr<TAcpuModelMatrix<TMatrixFloat>> res = new TAcpuModelMatrix<TMatrixFloat>(DeviceInfo);
        res->Create(MatrixAdd->GetDeviceCount(), DeviceInfo->HostMatrixScale, MODEL_DISCR_SCALE, data, quant, delayGrad, mmm);
        MatrixAdd->AddMatrix(PtrArg(res));
        return res;
    }

    TIntrusivePtr<IModelMatrixBase<i8>> CreateModelMatrix(i8 *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) override
    {
        (void)tps;
        return CreateModelMatrixImpl<i8>(data, quant, delayGrad, mmm);
    }
    TIntrusivePtr<IModelMatrixBase<e4m3>> CreateModelMatrix(e4m3 *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) override
    {
        (void)tps;
        return CreateModelMatrixImpl<e4m3>(data, quant, delayGrad, mmm);
    }
    TIntrusivePtr<IModelMatrixBase<half>> CreateModelMatrix(half *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) override
    {
        (void)tps;
        return CreateModelMatrixImpl<half>(data, quant, delayGrad, mmm);
    }

    void LaunchWorkers() override { MatrixAdd->LaunchWorkers(); }

    void InitDevice(yint deviceId, TStream &stream, TPtrArg<NCuda::TCudaMemoryAllocator> cudaMemllocator) override
    {
        DeviceInfo->InitDevice(deviceId, stream);
    }

    void InitFwdPass(yint deviceId, TPtrArg<TGraph> c, bool copyModelToDevice) override
    {
        if (copyModelToDevice) {
            DeviceInfo->DeviceArr[deviceId]->CudaMatrixScale->CopyToDevice(c);
        }
    }

    void InitBwdPass(yint deviceId, TPtrArg<TGraph> c) override
    {
        DeviceInfo->AssignIterCounter(deviceId, c, MatrixAdd->GetCurrentIteration());
    }

    TPtrArg<NCuda::TMultiDeviceBuffers> GetMultiBuffers(yint deviceGroupId) { return DeviceInfo->MultiBufferArr[deviceGroupId]; }
    yint GetDeviceCount() override { return MatrixAdd->GetDeviceCount(); }
    yint GetDeviceGroupCount() { return GetDeviceCount(); }
    bool IsHostMasterModel() override { return true; }

    EBackpropMode StartIteration(const TTrainingStep &step, EAddToModel addToModel) override
    {
        MatrixAdd->StartIteration(step, addToModel);
        return BM_NONE;
    }

    yint GetBackpropModeCount() override { return 1; }
    void WaitActiveCompute() override { MatrixAdd->WaitActiveCompute(); }
    void WaitDelayedCompute() override { MatrixAdd->WaitDelayedCompute(); }
    void ConvertMatrices() override { MatrixAdd->ConvertMatrices(); }
    void CopyModelParamsToHost() override {}

public:
    TCpuModelOps(yint deviceCount, yint maxMatrixCount, IMMDeltaHookGen *deltaHookGen)
    {
        DeviceInfo = new TAcpuDeviceInfo(deviceCount, maxMatrixCount);
        MatrixAdd = new TCPUMatrixAdd(deviceCount, maxMatrixCount, deltaHookGen);
    }
};
