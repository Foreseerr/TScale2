#pragma once
#include "custom_loss.cuh"
#include "gpu_final_kernels.cuh"
#include "cfg_precision.h"
#include "layer_cuda_base.cuh"


namespace NCUDA_Transformer
{
enum ETargetType {
    TARGET_TOKEN,
    TARGET_NO_SOFTMAX,
    TARGET_CUSTOM,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
//
class TFinalLayerWindows
{
public:
    struct TWin : public TThrRefBase
    {
        int Offset = 0;
        yint MaxLen = 0;
        TKernelParameter<int> Len;
        TKernelParameterRoundUp<int> LenRound;
        TKernelEnable Enabled;
        bool IsEnabled = true;

        TWin(int offset, yint maxLen) : Offset(offset), MaxLen(maxLen), Len(), LenRound(RoundUp(Len, MM_TILE)) {}
        void SetLen(yint totalLen)
        {
            yint len = Min<yint>(totalLen - Offset, MaxLen);
            Len.Set(len);
            Enabled.SetEnabled(true);
            IsEnabled = true;
        }
        void Disable()
        {
            Enabled.SetEnabled(false);
            IsEnabled = false;
        }
    };

private:
    TVector<TIntrusivePtr<TWin>> WindowArr;
    yint MaxWindowLen = 0;

public:
    void Create(yint maxWindowCount, yint maxWindowLen)
    {
        MaxWindowLen = maxWindowLen;
        WindowArr.resize(maxWindowCount);
        for (yint w = 0; w < maxWindowCount; ++w) {
            WindowArr[w] = new TWin(w * maxWindowLen, maxWindowLen);
        }
    }

    void Init(yint len)
    {
        yint currentWindowCount = DivCeil(len, MaxWindowLen);
        for (yint w = 0; w < currentWindowCount; ++w) {
            WindowArr[w]->SetLen(len);
        }
        for (yint w = currentWindowCount; w < YSize(WindowArr); ++w) {
            WindowArr[w]->Disable();
        }
    }

    TWin &GetWindow(yint w) { return *WindowArr[w]; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TFianlLayer
{
    struct TMicroBatch : public TThrRefBase
    {
        TFinalLayerWindows FinalWindows;
        TIntrusivePtr<TFragmentStates> FinalStates;
        TCudaVector<int> TargetArr;
        yint TargetNodeCount = 0;
    };

    bool IsInitialized = false;
    yint DeviceId = 0;
    TModelDescr ModelDescr;
    yint MaxLen = 0;
    yint FinalMaxLen = 0;
    yint WindowCount = 0;
    ETargetType TargetType = TARGET_TOKEN;
    TIntrusivePtr<ICustomLoss> CustomLoss;
    TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>> Final;
    TIntrusivePtr<TCudaMemoryPool> PoolFinal;
    TCudaVector<float> Bias;
    TCuda2DArray<float> DeltaFinalLayer;
    TCuda2DArray<half> LogitBuf;
    TCuda2DArray<float> LogitBufRowTileLogSum;
    TCuda2DArray<half> LogitBufHost;
    TCudaVector<float> TargetResProbArr;
    TCudaVector<float> SumScore;
    TCudaVector<float> SumTrainErr;
    TIntrusivePtr<TGraph> SumScoreComputer;
    TIntrusivePtr<TGraph> FinalLayerComputer;
    TVector<TIntrusivePtr<TGraph>> FinalLayerWindowComputerArr;
    TVector<TIntrusivePtr<TMicroBatch>> MBatchArr;


private:
    yint GetFinalLayerSizeRounded() const { return RoundUp(ModelDescr.OutputTokenCount, MM_TILE); }

    void CopyLogitBufToHost(TPtrArg<TGraph> c) { c->KernelCopy(&LogitBufHost, LogitBuf); }

    void AddFinalLayerWindow(TPtrArg<TGraph> c, yint microBatchId, TFinalLayerWindows::TWin &window)
    {
        TModelDims &dims = ModelDescr.Dims;
        int stateDim = dims.Dim;
        int flSize = GetFinalLayerSizeRounded();
        int outputCount = ModelDescr.OutputTokenCount;
        TCudaPOD<float> scaleFinalLayer = Final->GetScale();
        TMicroBatch &mb = *MBatchArr[microBatchId];

        auto finalStateNormalized = mb.FinalStates->GetFull().MakeFragment(0, window.Offset);

        float normScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;
        float flScale = CalcFinalLayerMult() * normScale;
        int computeLogit = (TargetType == TARGET_TOKEN) ? 1 : 0;
        TKernelOp *pFinalOp = 0;
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            // using half precision sum here impairs quality beyond repair
            pFinalOp = &MulForwardFp16<TStoreFinalLayerLogits>(c, &window, stateDim, flSize, finalStateNormalized, PtrArg(Final), &LogitBuf);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            pFinalOp = &MulForwardFp8<TStoreFinalLayerLogits>(c, &window, stateDim, flSize, finalStateNormalized, PtrArg(Final), &LogitBuf);
        } else if (FWD_MATMUL_TYPE == MATMUL_INT8) {
            pFinalOp = &MulForwardInt8<TStoreFinalLayerLogits>(c, &window, stateDim, flSize, finalStateNormalized, PtrArg(Final), &LogitBuf);
        }
        (*pFinalOp).Read(scaleFinalLayer, flScale, outputCount, Bias, computeLogit).Write(&LogitBufRowTileLogSum);
        pFinalOp->Link(window.Enabled);

        if (TargetType == TARGET_TOKEN) {
            CudaCall(c, SumLogWeight).Grid(MMTiles(window.Len)).Read(MMTiles(flSize)).Write(&LogitBufRowTileLogSum).Link(window.Enabled);
        }
    }


    TIntrusivePtr<TGraph> CreateSumScoreGraph(yint microBatchId)
    {
        TMicroBatch &mb = *MBatchArr[microBatchId];
        TIntrusivePtr<TGraph> c = new TGraph;
        c->SetMemPool(PoolFinal);
        c->ClearMem(&SumScore);
        for (yint w = 0; w < WindowCount; ++w) {
            TFinalLayerWindows::TWin &window = mb.FinalWindows.GetWindow(w);
            // apply final layer on fragment
            AddFinalLayerWindow(c, microBatchId, window);
            // compute score
            CudaCall(c, ComputeLossKernel)
                .Read(window.Offset, window.Len, LogitBuf, LogitBufRowTileLogSum, mb.TargetArr)
                .Write(&SumScore)
                .Link(window.Enabled);
        }
        return c;
    }


public:
    TFianlLayer(yint deviceId, const TModelDescr &modelDescr, ETargetType targetType, TPtrArg<ICustomLoss> pLoss)
        : DeviceId(deviceId), ModelDescr(modelDescr), TargetType(targetType), CustomLoss(pLoss)
    {
    }

    void Init(yint microBatchCount, yint maxLen)
    {
        MaxLen = maxLen;
        yint maxSamplesPerFinalWindow = 4096; // tradeoff perf vs memory consumption
        if (TargetType == TARGET_CUSTOM) {
            maxSamplesPerFinalWindow = maxLen; // disable final layer windows
        }
        FinalMaxLen = Min<yint>(maxLen, maxSamplesPerFinalWindow);
        WindowCount = DivCeil(maxLen, FinalMaxLen);
        // separate micro batches
        MBatchArr.resize(microBatchCount);
        for (yint mbId = 0; mbId < microBatchCount; ++mbId) {
            MBatchArr[mbId] = new TMicroBatch;
            TMicroBatch &mb = *MBatchArr[mbId];
            mb.FinalWindows.Create(WindowCount, maxSamplesPerFinalWindow);
        }
    }

    void AllocateCuda(TPtrArg<TCudaMemoryPool> poolFinal, TPtrArg<ICudaModelMatrixBase<TFastModelFloat>> final)
    {
        yint dim = ModelDescr.Dims.Dim;
        yint finalLayerRoundSize = GetFinalLayerSizeRounded();
        Final = final.Get();
        PoolFinal = poolFinal.Get();
        Bias.Allocate(ModelDescr.OutputTokenCount);
        // gradients
        DeltaFinalLayer.AllocateCuda(dim, finalLayerRoundSize, poolFinal);
        //
        LogitBuf.AllocateCuda(finalLayerRoundSize, FinalMaxLen, poolFinal);
        LogitBufRowTileLogSum.AllocateCuda(FinalMaxLen, finalLayerRoundSize / MM_TILE, PoolFinal);
        LogitBufHost.AllocateHost(finalLayerRoundSize, FinalMaxLen);
        if (TargetType == TARGET_CUSTOM) {
            CustomLoss->Allocate(DeviceId, MaxLen);
        }
        TargetResProbArr.Allocate(MaxLen);
        SumScore.Allocate(1);
        SumTrainErr.Allocate(4);
        SumTrainErr.ClearDeviceMem();
        for (yint mbId = 0; mbId < YSize(MBatchArr); ++mbId) {
            TMicroBatch &mb = *MBatchArr[mbId];
            mb.TargetArr.Allocate(MaxLen);
        }
        IsInitialized = true;
    }

    void SetFinalStates(yint microBatchId, TPtrArg<TFragmentStates> finalStates)
    {
        MBatchArr[microBatchId]->FinalStates = finalStates.Get();
    }

    template <class TStoreDelta>
    void AddBackpropFinalLayerGraph(
        TPtrArg<TGraph> c, yint microBatchId, TFinalLayerWindows::TWin &window, TCuda2DArray<TStateFloat> &stateGrad, yint xOffset)
    {
        const TModelDims &dims = ModelDescr.Dims;
        int stateDim = dims.Dim;
        int finalSizeRounded = GetFinalLayerSizeRounded();
        int finalTiles = DivCeil(finalSizeRounded, MM_TILE);
        int outputCount = ModelDescr.OutputTokenCount;
        TMicroBatch &mb = *MBatchArr[microBatchId];

        if (TargetType == TARGET_TOKEN) {
            // compute gradient (scale gradient by VEC_SCALE to avoid fp16 sum overflow in DeltaFinalLayer computation)
            CudaCall(c, ComputeGradient)
                .Grid(finalTiles, window.LenRound)
                .Read(window.Len, window.Offset, outputCount, LogitBuf, LogitBufRowTileLogSum, mb.TargetArr)
                .Write(&LogitBuf, &SumTrainErr)
                .Link(window.Enabled);
            CudaCall(c, CollectSumTrainErr).Write(&SumTrainErr).Link(window.Enabled);

        } else if (TargetType == TARGET_CUSTOM) {
            Y_VERIFY(window.Offset == 0 && "final layer windows not supported by custom loss");
            CustomLoss->ComputeGradient(DeviceId, c, window.Len, finalSizeRounded, MM_TILE, &LogitBuf);

        } else if (TargetType == TARGET_NO_SOFTMAX) {
            c->ClearMem(&LogitBuf).Link(window.Enabled);

        } else {
            Y_VERIFY(0);
        }

        // mul backward, scaling gradient by constant does not change result so we avoid scaling by final layer scale and MODEL_DISCR_SCALE
        yint xSize = stateGrad.GetXSize();
        yint ySize = Min<yint>(window.MaxLen, stateGrad.GetYSize() - window.Offset);
        auto stateGradFrag = stateGrad.MakeFragment(0, xSize, window.Offset, ySize);
        auto finalFrag = Final->GetFast().MakeFragment(xOffset, xSize, 0, finalSizeRounded);
        // using half precision accumulators on this mul backward destroys quality for some reason, so use full precision always
        Fp16MatMul<0, 1, TStore>(c, LogitBuf, finalFrag, &stateGradFrag, window.Len, xSize, finalSizeRounded).Struct().Link(window.Enabled);

        if (!ModelDescr.HasFlag(MPF_DISABLE_TUNE_FINAL_LAYER)) {
            auto finalStateNormalized = mb.FinalStates->GetFull().MakeFragment(0, window.Offset);
            // using half precisio sums is problematic due to sum overflow with typical state vec scale 1/32
            Fp16MatMul<1, 1, TStoreDelta>(c, LogitBuf, finalStateNormalized, &DeltaFinalLayer, finalSizeRounded, stateDim, window.Len)
                .Struct()
                .Link(window.Enabled);
        }
    }

    void AddBackprop(TPtrArg<TGraph> c, yint microBatchId, EBackpropMode bm, TCuda2DArray<TStateFloat> &stateGrad, yint xOffset)
    {
        Y_ASSERT(IsInitialized);
        TMicroBatch &mb = *MBatchArr[microBatchId];
        if (TargetType == TARGET_TOKEN || TargetType == TARGET_CUSTOM || TargetType == TARGET_NO_SOFTMAX) {
            // backprop final layer
            c->SetMemPool(PoolFinal);
            for (yint w = 0; w < WindowCount; ++w) {
                TFinalLayerWindows::TWin &window = mb.FinalWindows.GetWindow(w);
                // apply final layer on fragment
                AddFinalLayerWindow(c, microBatchId, window);
                // compute final layer gradient on fragment
                if (w == 0) {
                    AddBackpropFinalLayerGraph<TStore>(c, microBatchId, window, stateGrad, xOffset);
                } else {
                    AddBackpropFinalLayerGraph<TStoreAdd>(c, microBatchId, window, stateGrad, xOffset);
                }
            }
            if (!ModelDescr.HasFlag(MPF_DISABLE_TUNE_FINAL_LAYER)) {
                Final->AddDelta(c, DeltaFinalLayer, bm);
            }

        } else {
            Y_VERIFY(0 && "unknown target type");
        }
    }


    TIntrusivePtr<TGraph> CreateFinalLayerWindowGraph(yint microBatchId, yint dstWindow)
    {
        TMicroBatch &mb = *MBatchArr[microBatchId];
        TIntrusivePtr<TGraph> c = new TGraph;
        c->SetMemPool(PoolFinal);
        for (yint w = 0; w < WindowCount; ++w) {
            TFinalLayerWindows::TWin &window = mb.FinalWindows.GetWindow(w);
            if (dstWindow == -1 || dstWindow == w) {
                // window
                AddFinalLayerWindow(c, microBatchId, window);
                // target compute
                if (TargetType == TARGET_TOKEN) {
                    int flSize = GetFinalLayerSizeRounded();
                    int outputCount = ModelDescr.OutputTokenCount;
                    TMicroBatch &mb = *MBatchArr[microBatchId];
                    CudaCall(c, ComputeFinalProbKernel)
                        .Grid(flSize / MM_TILE, window.Len)
                        .Read(window.Offset, outputCount, LogitBufRowTileLogSum, mb.TargetArr)
                        .Write(&LogitBuf, &TargetResProbArr)
                        .Link(window.Enabled);
                }
            }
        }
        return c;
    }


    void CopyModelParams(TStream &stream, const TVector<float> &bias)
    {
        if (IsInitialized) {
            Put(stream, &Bias, bias);
        }
    }

    void Init(TStream &stream, yint microBatchId, yint len, const TVector<TNodeTarget> &target, bool isBackprop)
    {
        TMicroBatch &mb = *MBatchArr[microBatchId];
        // windows
        mb.FinalWindows.Init(len);
        if (IsInitialized) {
            // target arr
            TVector<int> targetArr;
            targetArr.resize(len, -1);
            yint count = 0;
            for (const TNodeTarget &nt : target) {
                Y_VERIFY(nt.TargetId >= 0 && nt.TargetId < ModelDescr.OutputTokenCount);
                Y_VERIFY(targetArr[nt.Node] == -1); // current ComputeGradient() supports single target per node
                targetArr[nt.Node] = nt.TargetId;
                ++count;
            }
            mb.TargetNodeCount = count;
            Put(stream, &mb.TargetArr, targetArr);
            // gradient data
            if (isBackprop) {
                if (TargetType == TARGET_CUSTOM) {
                    Y_ASSERT(microBatchId == 0);
                    // move to Init() for correct work with merged graph
                    CustomLoss->CopyDataToDevice(DeviceId, stream);
                }
            }
        }
    }

    void ComputeFragmentPredictions(TStream &stream, yint len, TVector<TVector<float>> *pPrediction)
    {
        // far from optimal, debug purposes only
        Y_ASSERT(IsInitialized);
        yint microBatchId = 0; // support single micro batch
        TMicroBatch &mb = *MBatchArr[microBatchId];
        pPrediction->resize(len);
        for (yint w = 0; w < WindowCount; ++w) {
            TFinalLayerWindows::TWin &window = mb.FinalWindows.GetWindow(w);
            if (!window.IsEnabled) {
                continue;
            }
            yint winOffset = window.Offset;
            yint winLen = window.Len.Get();
            FinalLayerWindowComputerArr[w]->Run(stream);
            stream.Sync();
            TVector<TVector<half>> winPred;
            GetAllData(LogitBufHost, &winPred);
            // scale result
            for (yint t = 0; t < winLen; ++t) {
                TVector<float> &dst = (*pPrediction)[winOffset + t];
                TVector<half> &pred = winPred[t];
                Y_VERIFY(YSize(pred) >= ModelDescr.OutputTokenCount);
                yint width = ModelDescr.OutputTokenCount;
                dst.yresize(width);
                for (yint c = 0; c < width; ++c) {
                    dst[c] = float(pred[c]);
                }
            }
        }
    }

    void ComputeFragmentPredictions(TStream &stream, yint len, TVector<float> *pPrediction)
    {
        Y_VERIFY(TargetType == TARGET_TOKEN);
        Y_ASSERT(IsInitialized);
        FinalLayerComputer->Run(stream);
        pPrediction->resize(len);
        TargetResProbArr.CopyToHost(stream);
        stream.Sync();
        GetData(TargetResProbArr, pPrediction, len);
    }

    float ComputeScore(TStream &stream)
    {
        Y_VERIFY(TargetType == TARGET_TOKEN);
        Y_ASSERT(IsInitialized);
        yint microBatchId = 0; // support single micro batch
        TMicroBatch &mb = *MBatchArr[microBatchId];
        SumScoreComputer->Run(stream);
        SumScore.CopyToHost(stream);
        stream.Sync();
        TVector<float> sumScore;
        GetAllData(SumScore, &sumScore);
        return sumScore[0] / mb.TargetNodeCount;
    }

    float GetAvrgTrainErr(TStream &stream)
    {
        if (IsInitialized) {
            Y_VERIFY(TargetType == TARGET_TOKEN);
            SumTrainErr.CopyToHost(stream);
            stream.Sync();
            TVector<float> sumTrainErr;
            GetAllData(SumTrainErr, &sumTrainErr);
            SumTrainErr.ClearDeviceMem(stream);
            return sumTrainErr[3] / sumTrainErr[2];
        } else {
            return 0;
        }
    }

    void CreateGraphs()
    {
        if (IsInitialized) {
            yint microBatchId = 0;
            SumScoreComputer = CreateSumScoreGraph(microBatchId);
            FinalLayerComputer = CreateFinalLayerWindowGraph(microBatchId, -1);
            FinalLayerWindowComputerArr.resize(WindowCount);
            for (yint w = 0; w < WindowCount; ++w) {
                TIntrusivePtr<TGraph> c = CreateFinalLayerWindowGraph(microBatchId, w);
                CopyLogitBufToHost(c);
                FinalLayerWindowComputerArr[w] = c;
            }
        }
    }
};
}
