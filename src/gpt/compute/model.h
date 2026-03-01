#pragma once
#include <gpt/att/nodes_batch.h>
#include <gpt/model_params/model_params.h>
#include <gpt/matrix/base_host.h>
#include <gpt/matrix/matrix_scale.h>
#include "layer.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IModelStorage : public TThrRefBase
{
    // retrieve param storage
    virtual const TModelDescr &GetModelDescr() = 0;
    virtual TIntrusivePtr<IModelMatrixBase<TEmbedFloat>> GetEmbedding() = 0;
    virtual TVector<TIntrusivePtr<TLayerBase>> &GetLayers() = 0;
    virtual TIntrusivePtr<IModelMatrixBase<TFastModelFloat>> GetFinalLayer() = 0;
    virtual TPtrArg<TModelVector> GetBias() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IModelInstance : virtual public TThrRefBase
{
    enum EInitType {
        INIT_BACKPROP,
        INIT_FWD,
    };
    virtual const TModelDescr &GetModelDescr() = 0;

    virtual void WaitInstanceOpLaunch() = 0; // used in custom loss, upon return we can modify input host buffers
    virtual TBatchNodes &GetNodes(yint microBatchId) = 0;
    virtual TArray2D<float> &GetSampleEmbedVectors(yint microBatchId) = 0;
    virtual void Init(EInitType initType) = 0;

    virtual void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) = 0;
    virtual void ComputeFragmentPredictions(TVector<float> *pPrediction) = 0;
    virtual float ComputeScore() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IModel : public TThrRefBase
{
    virtual const TModelDescr &GetModelDescr() = 0;
    virtual TPtrArg<IModelOps> GetModelOps() = 0;
    virtual TPtrArg<IModelStorage> GetModelStorage() = 0;

    virtual void GetParams(TModelParams *p) = 0;
    virtual void SetParams(const TModelParams &p) = 0;

    virtual TIntrusivePtr<IModelInstance> GetInstance(yint modelInstanceId) = 0;

    virtual void Backprop(const TTrainingStep &step, EAddToModel addToModel) = 0;
    virtual void WaitAllCompute() = 0;
    virtual float GetAvrgTrainErr() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IModelImpl : virtual public TThrRefBase
{
    virtual TIntrusivePtr<IModelInstance> GetInstance(yint modelInstanceId) = 0;
    virtual void Backprop(const TTrainingStep &step, EAddToModel addToModel) = 0;
    virtual void WaitAllCompute() = 0;
    virtual void CopyModelParamsToDevice() = 0;
    virtual float GetAvrgTrainErr() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelLayerLocation
{
    yint PassId = 0;
    yint Loc = 0;

    TModelLayerLocation() {}
    TModelLayerLocation(yint passId, yint loc) : PassId(passId), Loc(loc) {}
};
inline bool operator==(const TModelLayerLocation &a, const TModelLayerLocation &b)
{
    return a.PassId == b.PassId && a.Loc == b.Loc;
}
inline bool operator!=(const TModelLayerLocation &a, const TModelLayerLocation &b)
{
    return !(a == b);
}


struct TModelSplit
{
    yint TP = 1;
    yint PP = 1;
    yint MicroBatchCount = 1;
    // layer to device group binding
    TModelLayerLocation LocEmbed;
    TModelLayerLocation LocFinal;
    TVector<TModelLayerLocation> LocLayers;
    SAVELOAD(TP, PP, MicroBatchCount, LocEmbed, LocFinal, LocLayers);

    TModelLayerLocation GetLayerLoc(yint d) const { return LocLayers.empty() ? TModelLayerLocation() : LocLayers[d]; }
    yint GetPassCount() const { return LocFinal.PassId + 1; }
    yint GetLocationCount() const
    {
        yint res = Max<yint>(LocEmbed.Loc, LocFinal.Loc);
        for (const TModelLayerLocation &ml : LocLayers) {
            res = Max<yint>(res, ml.Loc);
        }
        return res + 1;
    }
};

TModelSplit SplitModel(yint depth, yint tp, yint pp, yint microBatchCount);


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace NCUDA_Transformer
{
struct ICustomLoss;
}
using NCUDA_Transformer::ICustomLoss;

struct IMMDeltaHookGen;

namespace NNet
{
struct INetReducer;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<IModel> CreateCpuTransformer(const TModelParams &modelParams, yint nodeCount);

TIntrusivePtr<IModel> CreateNoSoftmaxTransformer(const TModelParams &modelParams, yint nodeCount);

TIntrusivePtr<IModel> CreateHostMatrixTransformer(const TModelParams &modelParams, yint deviceCount, yint nodeCount);

TIntrusivePtr<IModel> CreateCustomLossTransformer(
    const TModelParams &modelParams, yint deviceCount, yint nodeCount, TPtrArg<ICustomLoss> customLoss);

TIntrusivePtr<IModel> CreateLocalTransformer(const TModelParams &modelParams, yint deviceCount, yint nodeCount, const TModelSplit &msplit);

TIntrusivePtr<IModel> CreateOneBitNetTransformer(
    const TModelParams &modelParams, yint deviceCount, yint nodeCount, IMMDeltaHookGen *deltaHookGen);

TIntrusivePtr<IModel> CreateClusterTransformer(const TModelParams &modelParams, yint deviceCount, yint nodeCount, yint myRank,
    yint rankCount, const TModelSplit &msplit, TPtrArg<NNet::INetReducer> netReducer);
