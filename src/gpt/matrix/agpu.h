#pragma once
#include "agpu_cuda.cuh"
#include "base_host.h"
#include "matrix_scale.h"
#include <lib/ib/ib_reducer.h>


using namespace NCuda;

class TCudaModelOps : public IModelOps
{
    TIntrusivePtr<TAgpuDeviceInfo> DeviceInfo;
    TIntrusivePtr<TAgpuAllCudaMatrices> AllMatrixArr;
    TCudaVector<TAddGradientKernelParams> HostAGParams;
    bool NeedCopyFromDevice = false;
    bool HasGradientData = false;
    yint MoeExpertCount = 0;

private:
    template <class TMatrixFloat>
    TIntrusivePtr<IModelMatrixBase<TMatrixFloat>> CreateModelMatrixImpl(const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps)
    {
        Y_VERIFY(quant == MM_QUANT_NONE);
        (void)delayGrad;
        TIntrusivePtr<TAgpuModelMatrix<TMatrixFloat>> res = new TAgpuModelMatrix<TMatrixFloat>(DeviceInfo, AllMatrixArr, MoeExpertCount, mmm, tps);
        res->Create(data);
        return res;
    }

    TIntrusivePtr<IModelMatrixBase<i8>> CreateModelMatrix(i8 *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) override
    {
        return CreateModelMatrixImpl<i8>(data, quant, delayGrad, mmm, tps);
    }
    TIntrusivePtr<IModelMatrixBase<e4m3>> CreateModelMatrix(e4m3 *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) override
    {
        return CreateModelMatrixImpl<e4m3>(data, quant, delayGrad, mmm, tps);
    }
    TIntrusivePtr<IModelMatrixBase<half>> CreateModelMatrix(half *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) override
    {
        return CreateModelMatrixImpl<half>(data, quant, delayGrad, mmm, tps);
    }

    void LaunchWorkers() override {}

    void InitDevice(yint deviceId, TStream &stream, TPtrArg<TCudaMemoryAllocator> cudaMemAllocator) override
    {
        DeviceInfo->InitDevice(deviceId, stream, cudaMemAllocator);
    }

    void InitFwdPass(yint deviceId, TPtrArg<TGraph> c, bool copyModelToDevice) override {}

    void InitBwdPass(yint deviceId, TPtrArg<TGraph> c) override
    {
        TAgpuDeviceInfo::TDeviceCtx &ctx = DeviceInfo->GetDeviceCtx(deviceId);
        c->KernelCopy(&ctx.CudaAGParams, HostAGParams);
        ctx.ResetDepChains();
    }

    TPtrArg<NCuda::TMultiDeviceBuffers> GetMultiBuffers(yint deviceGroupId) { return DeviceInfo->GetDeviceGroupMultiBuffers(deviceGroupId); }
    yint GetDeviceCount() override { return DeviceInfo->GetDeviceCount(); }
    yint GetDeviceGroupCount() override { return DeviceInfo->GetDeviceGroupCount(); }
    bool IsHostMasterModel() override { return false; }

    ETensorParallelSplit GetTensorSplit(ELayerType layerType, int matrId) override
    {
        if (DeviceInfo->GetTP() > 1) {
            if (layerType == MLT_MOE) {
                if (matrId == MP_MOE_SELECT) {
                    return TPS_COPY;
                }
                if (matrId == MP_MOE_CONTRACT) {
                    return TPS_ROW_MOE;
                }
            }
            return TPS_ROW;
        } else {
            return TPS_COPY;
        }
    }

    ETensorParallelSplit GetTensorSplit(int matrId) override
    {
        if (DeviceInfo->GetTP() > 1 && matrId == MP_MODEL_EMBED) {
            return TPS_COLUMN;
        }
        return TPS_COPY;
    }

    EBackpropMode StartIteration(const TTrainingStep &step, EAddToModel addToModel) override
    {
        TAddGradientKernelParams agParams;
        agParams.DispDecay = step.DispDecay;
        agParams.Beta1 = step.Beta1;
        agParams.Weight0 = step.Weight0;
        agParams.Weight1 = step.Weight1;
        agParams.L2Reg = step.L2Reg;
        agParams.StepMult = step.Rate;
        TVector<TAddGradientKernelParams> agArr;
        agArr.push_back(agParams);
        PutHost(&HostAGParams, agArr);
        
        NeedCopyFromDevice = true;
        // determine backprop mode
        yint bm = BM_NONE;
        if (HasGradientData) {
            bm |= BM_GRAD_ADD;
        }
        if (addToModel == GRADIENT_APPLY) {
            bm |= BM_GRAD_APPLY;
            HasGradientData = false;
        } else {
            HasGradientData = true;
        }
        return EBackpropMode(bm);
    }

    yint GetBackpropModeCount() override { return BM_COUNT; }

    void WaitActiveCompute() override {}

    void WaitDelayedCompute() override {}

    void ConvertMatrices() override { NeedCopyFromDevice = false; }

    void CopyModelParamsToHost() override
    {
        if (NeedCopyFromDevice) {
            AllMatrixArr->CopyModelParamsToHost();
            NeedCopyFromDevice = false;
        }
    }

public:
    TCudaModelOps(yint deviceCount, yint tp, yint pp, yint moeExpertCount, TPtrArg<NNet::INetReducer> ibReducer)
        : MoeExpertCount(moeExpertCount)
    {
        DeviceInfo = new TAgpuDeviceInfo(deviceCount, tp, pp, ibReducer);
        AllMatrixArr = new TAgpuAllCudaMatrices();
        HostAGParams.AllocateHost(1);
    }
};
