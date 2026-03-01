#include "cpu_transformer.h"
#include "cpu_util.h"
#include "layer.h"
#include <lib/math/linear.h>


namespace NCPU_Transformer
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// 
static void SoftMax(const TArray2D<float> &vecArr, TVector<TVector<float>> *pPrediction, const TVector<float> &bias)
{
    yint len = vecArr.GetYSize();
    yint dim = YSize(bias);
    Y_ASSERT(vecArr.GetXSize() == dim);
    pPrediction->resize(len);
    for (yint t = 0; t < len; ++t) {
        TVector<float> &dst = (*pPrediction)[t];
        dst.resize(dim);
        float maxVal = 0;
        for (yint k = 0; k < dim; ++k) {
            maxVal = fmaxf(maxVal, vecArr[t][k]);
        }
        double sumWeight = 0;
        for (yint k = 0; k < dim; ++k) {
            float val = vecArr[t][k];
            float w = exp2(val + bias[k] - maxVal);
            dst[k] = w;
            sumWeight += w;
        }
        float scale = 1 / sumWeight;
        for (yint k = 0; k < dim; ++k) {
            dst[k] *= scale;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
struct TFragmentStates
{
    TArray2D<float> State;

    void SetLength(yint len, yint dim)
    {
        State.SetSizes(dim, len);
        State.FillZero();
    }
    void Clear()
    {
        State.FillZero();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
//
class TComputeContext : public IModelImpl, public IModelInstance
{
    TIntrusivePtr<IModelOps> ModelOps;
    TIntrusivePtr<IModelStorage> ModelStorage;
    TVector<TIntrusivePtr<TLayerBase>> LayerArr;
    TCommonDataCPU CommonData;
    TVector<TFragmentStates> AllStates;
    TVector<TLabelIndex> LabelArr;
    TVector<ui32> LabelPtr;
    TVector<TNodeTarget> KeepTarget;
    yint MaxNodeCount = 0;
    TArray2D<float> FinalNormState;
    TBatchNodes Nodes;
    TArray2D<float> SampleEmbedVectors;

private:
    void ComputeEmbedding(const TArray2D<float> &labelEmbed)
    {
        TModelDescr modelDescr = GetModelDescr();
        int dim = modelDescr.Dims.Dim;
        yint len = YSize(LabelPtr) - 1;

        AllStates[0].State.FillZero();
        for (yint t = 0; t < len; ++t) {
            for (yint k = LabelPtr[t], kFinish = LabelPtr[t + 1]; k < kFinish; ++k) {
                yint label = LabelArr[k];
                float w = (label & LABEL_NEGATIVE) ? -1 : 1;
                label &= LABEL_MASK;
                for (yint x = 0; x < dim; ++x) {
                    AllStates[0].State[t][x] += labelEmbed[label][x] * w;
                }
            }
        }
        if (modelDescr.HasFlag(MPF_SAMPLE_EMBED_VECTORS)) {
            for (yint t = 0; t < len; ++t) {
                for (yint x = 0; x < dim; ++x) {
                    AllStates[0].State[t][x] += SampleEmbedVectors[t][x];
                }
            }
        }
    }

    void ComputeForward(TVector<TVector<float>> *pPrediction)
    {
        TModelDescr modelDescr = ModelStorage->GetModelDescr();
        int dim = modelDescr.Dims.Dim;

        auto labelEmbed = ModelStorage->GetEmbedding();
        auto finalLayer = ModelStorage->GetFinalLayer();

        // embedding
        ComputeEmbedding(GetData(labelEmbed));
        labelEmbed->GetHostCompute()->HostAllowDelayedUpdates();

        // apply layers
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            LayerArr[d]->ComputeForward(CommonData, AllStates[d].State, &AllStates[d + 1].State);
        }

        FinalNormState = NormalizeState(AllStates.back().State, dim / STATE_NORM_TILE);

        if (pPrediction) {
            TArray2D<float> predictionArr = MulForward(FinalNormState, GetData(finalLayer));
            ScaleMatrix(&predictionArr, CalcFinalLayerMult());
            SoftMax(predictionArr, pPrediction, ModelStorage->GetBias()->Vec);
        }
    }

    void ComputeBackprop(const TTrainingStep &step, EAddToModel addToModel)
    {
        TModelDescr modelDescr = ModelStorage->GetModelDescr();
        yint len = YSize(LabelPtr) - 1;
        int dim = modelDescr.Dims.Dim;

        auto labelEmbed = ModelStorage->GetEmbedding();
        auto finalLayer = ModelStorage->GetFinalLayer();

        TVector<TVector<float>> predArr;
        ComputeForward(&predArr);
        Y_ASSERT(YSize(predArr) == len);

        ModelOps->WaitActiveCompute();
        ModelOps->StartIteration(step, addToModel);

        TFragmentStates grad;
        grad.SetLength(len, dim);
        {
            // final soft max gradient
            TArray2D<float> gradArr;
            gradArr.SetSizes(modelDescr.OutputTokenCount, len);
            gradArr.FillZero();
            for (const TNodeTarget &nt : KeepTarget) {
                for (yint q = 0; q < modelDescr.OutputTokenCount; ++q) {
                    gradArr[nt.Node][q] += -predArr[nt.Node][q];
                }
                gradArr[nt.Node][nt.TargetId] += 1;
            }

            // can be omitted, gradient scale does not change anything due to gradient normalization
            ScaleMatrix(&gradArr, CalcFinalLayerMult() * LOG2);

            TArray2D<float> normStateGrad;
            InitDeltaMatrix(&normStateGrad, FinalNormState);
            MulBackwardWithAccum(&normStateGrad, GetData(finalLayer), gradArr);

            if (!modelDescr.HasFlag(MPF_DISABLE_TUNE_FINAL_LAYER)) {
                // modify final layer
                TArray2D<float> deltaFinalLayer;
                SumRankOne(FinalNormState, &deltaFinalLayer, gradArr);
                finalLayer->GetHostCompute()->HostApplyDelta(deltaFinalLayer);
            }

            NormalizeStateBackward(AllStates.back().State, dim / STATE_NORM_TILE, normStateGrad, &grad.State);
        }

        // modify layers
        for (yint d = YSize(LayerArr) - 1; d >= 0; --d) {
            LayerArr[d]->ComputeBackward(CommonData, AllStates[d].State, &grad.State);
            // can normalize grad.state, all deltas are normalized anyway
        }

        // modify embedding
        if (!modelDescr.HasFlag(MPF_DISABLE_TUNE_EMBED)) {
            TArray2D<float> deltaLabel;
            deltaLabel.SetSizes(dim, modelDescr.LabelCount);
            deltaLabel.FillZero();
            for (yint t = 0; t < len; ++t) {
                for (yint k = LabelPtr[t], kFinish = LabelPtr[t + 1]; k < kFinish; ++k) {
                    yint label = LabelArr[k];
                    float w = (label & LABEL_NEGATIVE) ? -1 : 1;
                    label &= LABEL_MASK;
                    for (yint x = 0; x < dim; ++x) {
                        deltaLabel[label][x] += grad.State[t][x] * w;
                    }
                }
            }
            labelEmbed->GetHostCompute()->HostApplyDelta(deltaLabel);
        }
    }

public:
    TComputeContext(TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount)
        : ModelOps(modelOps), ModelStorage(modelStorage), MaxNodeCount(nodeCount)
    {
        TModelDescr modelDescr = ModelStorage->GetModelDescr();
        LayerArr = ModelStorage->GetLayers();
        CommonData.Init(modelDescr, nodeCount);
    }

    // IModelInstance
    const TModelDescr &GetModelDescr() override { return ModelStorage->GetModelDescr(); }

    void WaitInstanceOpLaunch() override {}
    TBatchNodes &GetNodes(yint microBatchId) override
    {
        Y_ASSERT(microBatchId == 0);
        return Nodes;
    }
    TArray2D<float> &GetSampleEmbedVectors(yint microBatchId) override
    {
        Y_ASSERT(microBatchId == 0);
        return SampleEmbedVectors;
    }

    void Init(EInitType initType) override
    {
        (void)initType;
        TModelDescr modelDescr = ModelStorage->GetModelDescr();
        yint len = Nodes.GetNodeCount();
        Y_ASSERT(len <= MaxNodeCount);
        yint depth = YSize(LayerArr);
        AllStates.resize(depth + 1);
        for (yint k = 0; k < YSize(AllStates); ++k) {
            AllStates[k].SetLength(len, modelDescr.Dims.Dim);
        }
        LabelArr = Nodes.Labels.LabelArr;
        LabelPtr = Nodes.Labels.LabelPtr;
        KeepTarget = Nodes.Target;
        CommonData.InitAttention(Nodes);
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) override
    {
        ModelOps->WaitDelayedCompute();
        ComputeForward(pPrediction);
    }
    
    void ComputeFragmentPredictions(TVector<float> *pPrediction) override
    {
        TVector<TVector<float>> prediction;
        ComputeFragmentPredictions(&prediction);
        ClearPodArray(pPrediction, YSize(prediction));
        for (const TNodeTarget &nt : KeepTarget) {
            (*pPrediction)[nt.Node] = prediction[nt.Node][nt.TargetId];
        }
    }
    
    float ComputeScore() override
    {
        TVector<TVector<float>> prediction;
        ComputeFragmentPredictions(&prediction);
        float sum = 0;
        yint count = 0;
        for (const TNodeTarget &nt : KeepTarget) {
            sum += log(prediction[nt.Node][nt.TargetId]);
            count += 1;
        }
        return sum / count;
    }

    // IModelImpl
    TIntrusivePtr<IModelInstance> GetInstance(yint modelInstanceId) override
    {
        Y_ASSERT(modelInstanceId == 0);
        return this;
    }
    void Backprop(const TTrainingStep &step, EAddToModel addToModel) override { ComputeBackprop(step, addToModel); }
    void WaitAllCompute() override { ModelOps->WaitDelayedCompute(); }
    void CopyModelParamsToDevice() override {}
    float GetAvrgTrainErr() override { return 0; } // not implemented
};

TIntrusivePtr<IModelImpl> CreateContext(TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount)
{
    return new TComputeContext(modelStorage, modelOps, nodeCount);
}
}
