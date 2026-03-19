#include <util/pch.h>
#define KERNEL_UNIT "model/"
#include "layer_att.cuh"
#include "layer_att_3lin.cuh"
#include "layer_ffn.cuh"
#include "layer_moe.cuh"
#include "model.h"
#include "cpu_transformer.h"
#include "gpu_transformer.cuh"
#include <gpt/matrix/acpu.h>
#include <gpt/matrix/agpu.h>
#include <lib/cuda/cuda_arrays.h>


using namespace NCuda;


///////////////////////////////////////////////////////////////////////////////////////////////////
TModelSplit SplitModel(yint depth, yint tp, yint pp, yint microBatchCount)
{
    if (microBatchCount > pp) {
        DebugPrintf("MB <= PP supported\n");
        Y_VERIFY(0);
    }
    TModelSplit res;
    res.TP = tp;
    res.PP = pp;
    res.MicroBatchCount = microBatchCount;
    if (pp != 1) {
        yint passId = 0;
        res.LocEmbed = TModelLayerLocation(passId, 0);
        yint loc = 0;
        yint w = 0;
        for (yint d = 0; d < depth; ++d) {
            res.LocLayers.push_back(TModelLayerLocation(passId, loc));
            if (++w == 4) {
                w = 0;
                if (++loc == pp) {
                    loc = 0;
                    ++passId;
                }
            }
        }
        if (passId == 0) {
            loc = pp - 1;
        }
        res.LocFinal = TModelLayerLocation(passId, loc);
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TModelStorage : public IModelStorage
{
    TModelDescr ModelDescr;
    TIntrusivePtr<IModelMatrixBase<TFastModelFloat>> FinalLayer;
    TIntrusivePtr<IModelMatrixBase<TEmbedFloat>> Embedding;
    TVector<TIntrusivePtr<TLayerBase>> LayerArr;
    TIntrusivePtr<TModelVector> Bias;

private:
    TIntrusivePtr<IModelMatrixBase<TEmbedFloat>> GetEmbedding() override { return Embedding; }
    TVector<TIntrusivePtr<TLayerBase>> &GetLayers() override { return LayerArr; }
    TIntrusivePtr<IModelMatrixBase<TFastModelFloat>> GetFinalLayer() override { return FinalLayer; }
    TPtrArg<TModelVector> GetBias() override { return Bias; }

public:
    TModelStorage(const TModelParams &modelParams, TPtrArg<IModelOps> modelOps)
    {
        ModelDescr = modelParams.ModelDescr;

        EModelMatrixQuant quant = GetQuant(ModelDescr);
        EModelMatrixDelayGradient delayGradient = MM_DELAY_GRADIENT;
        if (ModelDescr.HasFlag(MPF_SYNC_ALL_GRADIENTS)) {
            delayGradient = MM_SYNC_GRADIENT;
        }

        FinalLayer = modelOps->CreateModelMatrix( //
            (TFastModelFloat *)0, modelParams.MatrArr[MP_MODEL_FINAL], MM_QUANT_NONE, MM_SYNC_GRADIENT, MM_MEM_DEVICE,
            modelOps->GetTensorSplit(MP_MODEL_FINAL));
        Embedding = modelOps->CreateModelMatrix(//
            (TEmbedFloat *)0, modelParams.MatrArr[MP_MODEL_EMBED], MM_QUANT_NONE, delayGradient, MM_MEM_HOST,
            modelOps->GetTensorSplit(MP_MODEL_EMBED));
        LayerArr.resize(YSize(ModelDescr.LayerArr));
        for (yint d = 0; d < YSize(ModelDescr.LayerArr); ++d) {
            const TModelDescr::TLayerParams &layerDescr = ModelDescr.LayerArr[d];
            const TModelParams::TAttentionMatrices &modelParamsLayer = modelParams.LayerArr[d];
            yint count = YSize(modelParamsLayer.MatrArr);

            TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> matrArr;
            matrArr.resize(count);
            for (yint k = 0; k < count; ++k) {
                ETensorParallelSplit tps = modelOps->GetTensorSplit(layerDescr.LayerType, k);
                matrArr[k] = modelOps->CreateModelMatrix(//
                    (TFastModelFloat *)0, modelParamsLayer.MatrArr[k], quant, MM_SYNC_GRADIENT, MM_MEM_DEVICE, tps);
            }

            switch (layerDescr.LayerType) {
            case MLT_ATT:
                LayerArr[d] = CreateAttLayer(ModelDescr, matrArr, layerDescr.AlibiSlope, layerDescr.AttentionTypeId);
                break;
            case MLT_ATT_3LIN:
                LayerArr[d] = Create3LinAttLayer(ModelDescr, matrArr, layerDescr.AlibiSlope, layerDescr.AttentionTypeId);
                break;
            case MLT_FFN:
                LayerArr[d] = CreateFFLayer(ModelDescr, matrArr);
                break;
            case MLT_MOE:
                LayerArr[d] = CreateMoELayer(ModelDescr, matrArr);
                break;
            default:
                Y_VERIFY(0 && "unknown layer type");
            }
        }
        Bias = new TModelVector(modelParams.Bias);
        Y_VERIFY(YSize(Bias->Vec) == ModelDescr.OutputTokenCount);
    }

    void GetParams(TModelParams *p)
    {
        p->ModelDescr = ModelDescr;
        yint depth = YSize(LayerArr);
        p->LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr = LayerArr[d]->GetMatrArr();
            yint count = YSize(matrArr);
            p->LayerArr[d].MatrArr.resize(count);
            for (yint k = 0; k < count; ++k) {
                matrArr[k]->GetData(&p->LayerArr[d].MatrArr[k]);
            }
        }
        p->MatrArr.resize(MP_MODEL_COUNT);
        FinalLayer->GetData(&p->MatrArr[MP_MODEL_FINAL]);
        Embedding->GetData(&p->MatrArr[MP_MODEL_EMBED]);
        p->Bias = Bias->Vec;
    }

    void SetParams(const TModelParams &p)
    {
        Y_VERIFY(ModelDescr == p.GetModelDescr());
        yint depth = YSize(p.LayerArr);
        LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr = LayerArr[d]->GetMatrArr();
            yint count = YSize(matrArr);
            for (yint k = 0; k < count; ++k) {
                matrArr[k]->SetData(p.LayerArr[d].MatrArr[k]);
            }
        }
        FinalLayer->SetData(p.MatrArr[MP_MODEL_FINAL]);
        Embedding->SetData(p.MatrArr[MP_MODEL_EMBED]);
        Bias->Vec = p.Bias;
    }

    const TModelDescr &GetModelDescr() override { return ModelDescr; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TModel : public IModel
{
    TIntrusivePtr<IModelOps> ModelOps;
    TIntrusivePtr<TModelStorage> ModelStorage;
    TIntrusivePtr<IModelImpl> ModelImpl;

public:
    TModel(const TModelParams &modelParams, TPtrArg<IModelOps> modelOps) : ModelOps(modelOps.Get())
    {
        ModelStorage = new TModelStorage(modelParams, ModelOps);
        ModelOps->LaunchWorkers();
    }

    void SetImpl(TIntrusivePtr<IModelImpl> impl) { ModelImpl = impl.Get(); }

    const TModelDescr &GetModelDescr() override { return ModelStorage->GetModelDescr(); }
    TPtrArg<IModelOps> GetModelOps() override { return ModelOps; }
    TPtrArg<IModelStorage> GetModelStorage() override { return PtrArg(ModelStorage); }

    void GetParams(TModelParams *p) override
    {
        WaitAllCompute();
        ModelOps->CopyModelParamsToHost();
        ModelStorage->GetParams(p);
    }

    void SetParams(const TModelParams &p) override
    {
        Y_VERIFY(ModelStorage->GetModelDescr() == p.ModelDescr);
        WaitAllCompute();
        ModelStorage->SetParams(p);
        ModelOps->ConvertMatrices();
        ModelImpl->CopyModelParamsToDevice();
    }

    TIntrusivePtr<IModelInstance> GetInstance(yint modelInstanceId) override { return ModelImpl->GetInstance(modelInstanceId); }

    void Backprop(const TTrainingStep &step, EAddToModel addToModel) override{ ModelImpl->Backprop(step, addToModel); }
    void WaitAllCompute() override { ModelImpl->WaitAllCompute(); }
    float GetAvrgTrainErr() override { return ModelImpl->GetAvrgTrainErr(); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
inline yint GetMaxMatrixCount(const TModelParams &modelParams)
{
    return DivCeil(YSize(modelParams.ModelDescr.LayerArr) * MP_MAX_COUNT + MP_MODEL_COUNT, 32) * 32;
}

using namespace NCUDA_Transformer;

TIntrusivePtr<IModel> CreateCpuTransformer(const TModelParams &modelParams, yint nodeCount)
{
    yint deviceCount = 1;
    TIntrusivePtr<IModelOps> modelOps = new TCpuModelOps(deviceCount, GetMaxMatrixCount(modelParams), nullptr);
    TIntrusivePtr<TModel> model = new TModel(modelParams, modelOps);
    model->SetImpl(NCPU_Transformer::CreateContext(model->GetModelStorage(), modelOps, nodeCount));
    return model;
}

TIntrusivePtr<IModel> CreateNoSoftmaxTransformer(const TModelParams &modelParams, yint nodeCount)
{
    yint deviceCount = 1;
    TIntrusivePtr<IModelOps> modelOps = new TCpuModelOps(deviceCount, GetMaxMatrixCount(modelParams), nullptr);
    TIntrusivePtr<TModel> model = new TModel(modelParams, modelOps);
    model->SetImpl(CreateContextNoSoftmax(model->GetModelStorage(), modelOps, nodeCount));
    return model;
}


TIntrusivePtr<IModel> CreateHostMatrixTransformer(const TModelParams &modelParams, yint deviceCount, yint nodeCount)
{
    TIntrusivePtr<IModelOps> modelOps = new TCpuModelOps(deviceCount, GetMaxMatrixCount(modelParams), nullptr);
    TIntrusivePtr<TModel> model = new TModel(modelParams, modelOps);
    model->SetImpl(CreateContext(model->GetModelStorage(), modelOps, TModelSplit(), nodeCount));
    return model;
}


TIntrusivePtr<IModel> CreateCustomLossTransformer(
    const TModelParams &modelParams, yint deviceCount, yint nodeCount, TPtrArg<ICustomLoss> customLoss)
{
    TIntrusivePtr<IModelOps> modelOps;
    if (CudaCanEnablePeerAccess()) {
        // use CUDA gradient accumulation
        modelOps = new TCudaModelOps(deviceCount, 1, 1, 0, null_ptr_arg);
    } else {
        // fallback to CPU gradient aggregation for non peer access enabled configurations
        modelOps = new TCpuModelOps(deviceCount, GetMaxMatrixCount(modelParams), nullptr);
    }
    TIntrusivePtr<TModel> model = new TModel(modelParams, modelOps);
    model->SetImpl(CreateWithCustomLoss(model->GetModelStorage(), modelOps, nodeCount, customLoss));
    return model;
}


TIntrusivePtr<IModel> CreateLocalTransformer(const TModelParams &modelParams, yint deviceCount, yint nodeCount, const TModelSplit &msplit)
{
    yint moeExpertCount = modelParams.GetModelDescr().Dims.GetMoeExpertCount();
    TIntrusivePtr<IModelOps> modelOps = new TCudaModelOps(deviceCount, msplit.TP, msplit.PP, moeExpertCount, null_ptr_arg);
    TIntrusivePtr<TModel> model = new TModel(modelParams, modelOps);
    model->SetImpl(CreateContext(model->GetModelStorage(), modelOps, msplit, nodeCount));
    return model;
}


TIntrusivePtr<IModel> CreateOneBitNetTransformer(
    const TModelParams &modelParams, yint deviceCount, yint nodeCount, IMMDeltaHookGen *deltaHookGen)
{
    TIntrusivePtr<IModelOps> modelOps = new TCpuModelOps(deviceCount, GetMaxMatrixCount(modelParams), deltaHookGen);
    TIntrusivePtr<TModel> model = new TModel(modelParams, modelOps);
    model->SetImpl(CreateContext(model->GetModelStorage(), modelOps, TModelSplit(), nodeCount));
    return model;
}


TIntrusivePtr<IModel> CreateClusterTransformer(const TModelParams &modelParams, yint deviceCount, yint nodeCount, yint myRank,
    yint rankCount, const TModelSplit &msplit, TPtrArg<NNet::INetReducer> netReducer)
{
    yint moeExpertCount = modelParams.GetModelDescr().Dims.GetMoeExpertCount();
    TIntrusivePtr<IModelOps> modelOps = new TCudaModelOps(deviceCount, msplit.TP, msplit.PP, moeExpertCount, netReducer);
    TIntrusivePtr<TModel> model = new TModel(modelParams, modelOps);
    model->SetImpl(CreateContext(model->GetModelStorage(), modelOps, msplit, nodeCount));
    return model;
}
