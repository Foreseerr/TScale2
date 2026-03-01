#include <util/pch.h>
#define KERNEL_UNIT "layer_att/"
#include "layer_att.cuh"
#include "layer.h"
#include "layer_cuda_base.cuh"
#include "cpu_util.h"
#include "gpu_att_fp16.cuh"
#include "gpu_att_infer.cuh"
#include "gpu_rope.cuh"
#include "gpu_lolu.cuh"
#include "gpu_layer_norm.cuh"
#include "layer_att_reference.h"
#include <gpt/model_params/model_dim.h>
#include <gpt/att/rope.h>
#include <lib/hp_timer/hp_timer.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaAttLayer : public TCudaLayerBase
{
    float AlibiSlope = 0;
    yint AttentionTypeId = 0;
    // Q
    TCuda2DArray<float> Qscale;
    TCuda2DArray<TAttVecFloat> Q;
    TCuda2DArray<TAttVecFloat> QT;
    // K
    TCuda2DArray<float> Kscale;
    TCuda2DArray<TAttVecFloat> K;
    TCuda2DArray<TAttVecFloat> KT;
    // V
    TCuda2DArray<float> Vscale;
    TCuda2DArray<TAttVecFloat> V;
    TCuda2DArray<TAttVecFloat> VT;
    // rest
    TCuda2DArray<half> AttGate;
    TCuda2DArray<half> ValLookup;
    TCuda2DArray<float> SumWeightLog;
    TCuda2DArray<float> DScale;
    TIntrusivePtr<TMultiDevice2DArray<TNormStateFloat>> VLG;
    // grad
    TCuda2DArray<TFastGradientFloat> DQ16;
    TCuda2DArray<TFastGradientFloat> DK16;
    TCuda2DArray<TFastGradientFloat> DV16;
    TIntrusivePtr<TMultiDeviceFRed2DArray<half>> DVLG;
    TCuda2DArray<TFastGradientFloat> DAttGate;
    TCuda2DArray<half> DValLookup16;
    TCuda2DArray<e4m3> DValLookupE4;
    TCuda2DArray<e4m3> DValLookupE4T;
    TCuda2DArray<float> DValLookupE4Scale;
    //
    TMatMulBuffers MatmulQ;
    TMatMulBuffers MatmulK;
    TMatMulBuffers MatmulV;
    TMatMulBuffers MatmulAttGate;
    TMatMulBuffers MatmulCombiner;
    //
    TCudaLayerPools Pools;

    yint GetDim(yint dim) const
    {
        yint dgSize = MultiBuffers->GetDgSize();
        return dim / dgSize;
    }

public:
    TCudaAttLayer(TPtrArg<TMultiDeviceBuffers> multiBuffers, const void *groupPtr, yint deviceId, const TModelDims &dims, bool updateLayers,
        float alibiSlope, yint attentionTypeId, yint len, const TVector<TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>>> &matrArr,
        TCudaLayerPools &pools)
        : TCudaLayerBase(multiBuffers, deviceId, dims, updateLayers, matrArr), AlibiSlope(alibiSlope), AttentionTypeId(attentionTypeId),
          Pools(pools)
    {
        yint qSum = GetDim(Dims.GetQSum());
        yint ttSum = GetDim(Dims.GetTTSum());
        yint headCount = GetDim(Dims.HeadCount);
        yint ttSumFull = Dims.GetTTSum();

        Qscale.AllocateCuda(len, headCount, pools.Fwd);
        Q.AllocateCuda(qSum, len, pools.Fwd);
        Kscale.AllocateCuda(len, headCount, pools.Fwd);
        K.AllocateCuda(qSum, len, pools.Fwd);
        Vscale.AllocateCuda(len, headCount, pools.Fwd);
        V.AllocateCuda(ttSum, len, pools.Fwd);
        AttGate.AllocateCuda(ttSum, len, pools.Fwd);
        ValLookup.AllocateCuda(ttSum, len, pools.Fwd);
        SumWeightLog.AllocateCuda(len, headCount, pools.Fwd);
        DScale.AllocateCuda(len, headCount, pools.Fwd);
        VLG = MultiBuffers->Fab().Create2DArray<TNormStateFloat>(Sprintf("%p-VLG", groupPtr));
        VLG->AllocateCuda(DeviceId, ttSumFull, len, pools.Fwd);
        //
        DQ16.AllocateCuda(qSum, len, pools.Bwd);
        DK16.AllocateCuda(qSum, len, pools.Bwd);
        DV16.AllocateCuda(ttSum, len, pools.Bwd);
        DVLG = MultiBuffers->Fab().CreateFRed2DArray<half>(Sprintf("%p-DVLG", groupPtr));
        DVLG->AllocateCuda(DeviceId, ttSumFull, len, pools.Bwd);
        DAttGate.AllocateCuda(ttSum, len, pools.Bwd);
        DValLookup16.AllocateCuda(ttSum, len, pools.Bwd);
        MatmulQ.AllocateCuda(MatrArr[MP_ATT_Q], len, pools.Bwd);
        MatmulK.AllocateCuda(MatrArr[MP_ATT_K], len, pools.Bwd);
        MatmulV.AllocateCuda(MatrArr[MP_ATT_V], len, pools.Bwd);
        MatmulAttGate.AllocateCuda(MatrArr[MP_ATT_GATE], len, pools.Bwd);
        MatmulCombiner.AllocateCuda(MatrArr[MP_ATT_COMBINER], len, pools.Bwd);
    }


    void ComputeVLG(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState)
    {
        yint stateDim = Dims.Dim;
        yint ttSum = GetDim(Dims.GetTTSum());
        yint headCount = GetDim(Dims.HeadCount);
        float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;

        TAttentionGroupData<ATT_GROUP> &attGD = *pParams->AttGDArr[AttentionTypeId];
        TCudaPOD<float> scaleK = MatmulK.GetScale();

        MatmulQ.Forward<TStoreRowTileNormalizeRope>(c, pParams, normState, &Q)
            .Read(pParams->RopeBuf, stateNormScale, Q_VEC_SCALE)
            .Write(&Qscale);
        MatmulK.Forward<TStoreRowTileNormalizeRope>(c, pParams, normState, &K)
            .Read(pParams->RopeBuf, stateNormScale, K_VEC_SCALE)
            .Write(&Kscale);
        MatmulV.Forward<TStoreRowTileNormalize>(c, pParams, normState, &V)
            .Read(stateNormScale, V_VEC_SCALE) //
            .Write(&Vscale);
        MatmulAttGate.Forward<TStoreScaled>(c, pParams, normState, &AttGate).Read(nullptr, stateNormScale);

        // fp16 attention
        CudaCall(c, Fp16Att)
            .Grid(headCount, DivCeil(pParams->Len, ATT_GROUP, MM_TILE))
            .Read(scaleK, Q, K, Kscale)
            .Read(V)
            .Read(attGD.AttSpans, attGD.AttSpanPtr, AlibiSlope)
            .Write(&SumWeightLog, &ValLookup);

        auto vlgWin = XSplitWindow(VLG->GetData(DeviceId));
        CudaCall(c, MatrixLoLU<TNormStateFloat>)
            .Grid(ttSum / MM_TILE, MMTiles(pParams->Len))
            .Read(pParams->Len, AttGate, ValLookup)
            .Write(&vlgWin);
        MultiBuffers->Op().AllGatherXSplit(c, VLG, DeviceId);
    }

    void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState, TCuda2DArray<TStateFloat> *pState,
        TFragmentStates *pNextNormState) // override
    {
        Pools.SetMemPools(c);
        yint ttSumFull = Dims.GetTTSum();

        ComputeVLG(c, pParams, normState);

        float combinerNormScale = CalcDotScale(ttSumFull) * MODEL_DISCR_SCALE * V_VEC_SCALE;
        MatmulCombiner.Forward<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, VLG->GetData(DeviceId), pState)
            .Read(combinerNormScale, MatmulCombiner.GetScale(), STATE_VEC_SCALE)
            .Write(&pNextNormState->GetLocal(), &pNextNormState->GetScaleLocal());
    }


    TCudaPOD<float> AddBackward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState,
        TCuda2DArray<TFastGradientFloat> *pStateGrad, TCudaPOD<float> gradScale,
        TPtrArg<TMultiDeviceFRed2DArray<float>> dNormStatePtr) override
    {
        Pools.SetMemPools(c);
        yint qSum = GetDim(Dims.GetQSum());
        yint ttSum = GetDim(Dims.GetTTSum());
        yint headCount = GetDim(Dims.HeadCount);

        TAttentionGroupData<ATT_GROUP> &attGD = *pParams->AttGDArr[AttentionTypeId];
        TCudaPOD<float> scaleK = MatmulK.GetScale();

        ComputeVLG(c, pParams, normState);

        // combiner derivatives
        auto bmCombiner = MatmulCombiner.BackpropMulti(c, pParams, *pStateGrad);
        Backprop(c, pParams, MultiArg() + bmCombiner, &DVLG->GetData(DeviceId));
        MultiBuffers->Op().ReduceXSplit(c, DVLG, pParams->Len, DeviceId);
        auto dvlg = XSplitWindow(DVLG->GetData(DeviceId));

        // fp16 attention derivatives
        CudaCall(c, BackpropLoLUdScale<half, TFastGradientFloat, half>)
            .Grid(ttSum / MM_TILE, pParams->LenRound)
            .Read(pParams->Len, AttGate, ValLookup, V_VEC_SCALE, dvlg)
            .Write(&DAttGate, &DValLookup16, &DScale);

        CudaCall(c, Fp16AttGradQ<TStoreAttBackNormalizeRope>)
            .Grid(headCount, DivCeil(pParams->Len, ATT_GROUP, MM_TILE))
            .Read(scaleK, Q, K, Kscale)
            .Read(V)
            .Read(DValLookup16)
            .Read(DScale, SumWeightLog)
            .Read(attGD.AttSpans, attGD.AttSpanPtr, AlibiSlope)
            .Write(&DQ16)
            .Struct()
            .Read(Q, Qscale, pParams->RopeBuf);
        // CudaCall(c, PrintArr<128, float>)(40, DQFloat);

        CudaCall(c, Fp16AttGradKV<TStoreAttBackRope, TStoreAttBackNormalize>)
            .Grid(headCount, DivCeil(pParams->Len, ATT_GROUP, MM_TILE))
            .Read(scaleK, Q, K, Kscale)
            .Read(V)
            .Read(DValLookup16)
            .Read(DScale, SumWeightLog)
            .Read(attGD.RevAttSpans, attGD.RevAttSpanPtr, AlibiSlope)
            .Write(&DK16, &DV16)
            .Struct()
            .Read(pParams->RopeBuf)
            .Struct()
            .Read(V, Vscale);

        TCuda2DArray<float> &dNormState = dNormStatePtr->GetData(DeviceId);
        auto bm1 = MatmulQ.BackpropMulti(c, pParams, DQ16);
        auto bm2 = MatmulK.BackpropMulti(c, pParams, DK16);
        auto bm3 = MatmulV.BackpropMulti(c, pParams, DV16);
        auto bm4 = MatmulAttGate.BackpropMulti(c, pParams, DAttGate);
        Backprop(c, pParams, MultiArg() + bm1 + bm2 + bm3 + bm4, &dNormState);
        MultiBuffers->Op().ReduceXSplit(c, dNormStatePtr, pParams->Len, DeviceId);

        MatmulCombiner.BackpropDelta(c, pParams, VLG->GetData(DeviceId), *pStateGrad, gradScale, UpdateLayers);
        MatmulQ.BackpropDelta(c, pParams, normState, DQ16, gradScale, UpdateLayers);
        MatmulK.BackpropDelta(c, pParams, normState, DK16, gradScale, UpdateLayers);
        MatmulV.BackpropDelta(c, pParams, normState, DV16, gradScale, UpdateLayers);
        MatmulAttGate.BackpropDelta(c, pParams, normState, DAttGate, gradScale, UpdateLayers);

        return MatmulCombiner.GetScale();
    }

    void AddDelta(TPtrArg<TGraph> c, EBackpropMode bm) override
    {
        if (UpdateLayers) {
            MatmulCombiner.AddDelta(c, bm);
            MatmulQ.AddDelta(c, bm);
            MatmulK.AddDelta(c, bm);
            MatmulV.AddDelta(c, bm);
            MatmulAttGate.AddDelta(c, bm);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaAttLayerInfer : public TCudaInferLayerBase
{
    float AlibiSlope = 0;
    int MaxWidth = 0;
    // Q
    TCuda2DArray<float> Qscale;
    TCuda2DArray<TAttVecFloat> Q;
    // K
    TCuda2DArray<float> Kscale;
    TCuda2DArray<TAttVecFloat> K;
    TCuda2DArray<float> KeyCacheScale;
    TCuda2DArray<TAttVecFloat> KeyCache;
    // V
    TCuda2DArray<float> Vscale;
    TCuda2DArray<TAttVecFloat> V;
    TCuda2DArray<TAttVecFloat> ValueCache;
    // rest
    TCuda2DArray<half> AttGate;
    TCuda2DArray<half> ValLookup;
    TCuda2DArray<TNormStateFloat> VLG;
    //
    TInferMatMul MatmulQ;
    TInferMatMul MatmulK;
    TInferMatMul MatmulV;
    TInferMatMul MatmulAttGate;
    TInferMatMul MatmulCombiner;
    //
    TIntrusivePtr<TCudaMemoryPool> Pool;

public:
    TCudaAttLayerInfer(const TModelDims &dims, float alibiSlope, yint maxWidth, yint len, yint kvCacheSize,
        const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr, TPtrArg<TCudaMemoryPool> pool)
        : TCudaInferLayerBase(dims), AlibiSlope(alibiSlope), MaxWidth(maxWidth), Pool(pool)
    {
        yint qSum = dims.GetQSum();
        yint ttSum = dims.GetTTSum();
        yint headCount = dims.HeadCount;

        Qscale.AllocateCuda(len, headCount, pool);
        Q.AllocateCuda(qSum, len, pool);
        Kscale.AllocateCuda(len, headCount, pool);
        K.AllocateCuda(qSum, len, pool);
        KeyCacheScale.AllocateCuda(kvCacheSize, headCount);
        KeyCache.AllocateCuda(qSum, kvCacheSize);
        Vscale.AllocateCuda(len, headCount, pool);
        V.AllocateCuda(ttSum, len, pool);
        ValueCache.AllocateCuda(ttSum, kvCacheSize);
        AttGate.AllocateCuda(ttSum, len, pool);
        ValLookup.AllocateCuda(ttSum, len, pool);
        VLG.AllocateCuda(ttSum, len, pool);
        //
        MatmulQ.Init(matrArr[MP_ATT_Q]);
        MatmulK.Init(matrArr[MP_ATT_K]);
        MatmulV.Init(matrArr[MP_ATT_V]);
        MatmulAttGate.Init(matrArr[MP_ATT_GATE]);
        MatmulCombiner.Init(matrArr[MP_ATT_COMBINER]);
    }


    void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TStateFloat> *pState,
        TCuda2DArray<TNormStateFloat> *pNormState, TCuda2DArray<float> *pNormStateScale) override
    {
        c->SetMemPool(Pool);
        yint stateDim = Dims.Dim;
        yint qSum = Dims.GetQSum();
        yint ttSum = Dims.GetTTSum();
        yint headCount = Dims.HeadCount;
        float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;
        TCuda2DArray<TNormStateFloat> &normState = *pNormState;
        TInferAttentionData &att = *pParams->InferAtt;
        TCudaPOD<float> scaleK = MatmulK.GetScale();

        MatmulQ.Forward<TStoreRowTileNormalize>(c, pParams, normState, &Q).Read(stateNormScale, Q_VEC_SCALE).Write(&Qscale);
        MatmulK.Forward<TStoreRowTileNormalize>(c, pParams, normState, &K).Read(stateNormScale, K_VEC_SCALE).Write(&Kscale);
        MatmulV.Forward<TStoreRowTileNormalize>(c, pParams, normState, &V).Read(stateNormScale, V_VEC_SCALE).Write(&Vscale);
        MatmulAttGate.Forward<TStoreScaled>(c, pParams, normState, &AttGate).Read(nullptr, stateNormScale);

        CudaCall(c, InferAtt<Q_DIM, TT_DIM, USE_ROPE>)
            .Grid(pParams->Len, headCount)
            .Read(Q)
            .Read(KeyCache, scaleK, KeyCacheScale)
            .Read(ValueCache)
            .Read(AlibiSlope, MaxWidth)
            .Read(att.ReadArr, att.ReadArrPtr, att.RopeBuf)
            .Write(&ValLookup);

        CudaCall(c, CopyToKVcache<Q_DIM, TT_DIM>)
            .Grid(pParams->Len)
            .Read(headCount)
            .Read(K, Kscale, V)
            .Read(att.WriteArr, att.RopeBuf)
            .Write(&KeyCache, &KeyCacheScale, &ValueCache);

        CudaCall(c, MatrixLoLU<TNormStateFloat>)
            .Grid(ttSum / MM_TILE, MMTiles(pParams->Len))
            .Read(pParams->Len, AttGate, ValLookup)
            .Write(&VLG);

        float combinerNormScale = CalcDotScale(ttSum) * MODEL_DISCR_SCALE * V_VEC_SCALE;
        MatmulCombiner.Forward<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, VLG, pState)
            .Read(combinerNormScale, MatmulCombiner.GetScale(), STATE_VEC_SCALE)
            .Write(pNormState, pNormStateScale);
    }
};


TIntrusivePtr<TCudaInferLayerBase> CreateAttLayerInference(const TModelDescr &modelDescr, float alibiSlope, const TAttentionType &attnType,
    yint len, yint kvCacheSize, const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr,
    TPtrArg<TCudaMemoryPool> pool)
{
    Y_VERIFY(attnType.Type == TAttentionType::CASUAL);
    return new TCudaAttLayerInfer(modelDescr.Dims, alibiSlope, attnType.Width, len, kvCacheSize, matrArr, pool);
}
}
using namespace NCuda;


///////////////////////////////////////////////////////////////////////////////////////////////////
// CPU Attention implementation
//
class TAttLayer : public TLayerBase
{
    TModelDims Dims;
    float AlibiSlope = 0;
    yint AttentionTypeId = 0;

    void ComputeForward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pState) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        yint headCount = Dims.HeadCount;

        *pState = prevState;
        TArray2D<float> normState = NormalizeState(*pState, dim / STATE_NORM_TILE);

        auto &attCombiner = MatrArr[MP_ATT_COMBINER];
        auto &attQ = MatrArr[MP_ATT_Q];
        auto &attK = MatrArr[MP_ATT_K];
        auto &attGate = MatrArr[MP_ATT_GATE];
        auto &attV = MatrArr[MP_ATT_V];

        TAttentionComputer attComp(Dims, AlibiSlope, 1.0f);
        const TAttentionInfo &attInfo = common.AttArr[AttentionTypeId].Att;

        TArray2D<float> qSrc = MulForward(normState, GetData(attQ));
        TArray2D<float> kSrc = MulForward(normState, GetData(attK));
        TArray2D<float> gate = MulForward(normState, GetDataNoScale(attGate));
        TArray2D<float> vSrc = MulForward(normState, GetData(attV));

        TArray2D<float> q = NormalizeState(qSrc, headCount);
        TArray2D<float> k = kSrc;
        TArray2D<float> v = NormalizeState(vSrc, headCount);
        ApplyRope(common.RopeBuf, 1, &q);
        ApplyRope(common.RopeBuf, 1, &k);

        TArray2D<float> valLookup;
        attComp.ComputeValLookup(len, Dims, q, k, v, attInfo, &valLookup);

        TArray2D<float> rg;
        LoLU(gate, valLookup, &rg);

        TArray2D<float> deltaState = MulForward(rg, GetData(attCombiner));
        AddScaledMatrix(pState, deltaState, 1);
    }

    void ComputeBackward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pGrad) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        yint headCount = Dims.HeadCount;

        TArray2D<float> normState = NormalizeState(prevState, dim / STATE_NORM_TILE);
        TArray2D<float> dNormState;
        InitDeltaMatrix(&dNormState, normState);

        auto &matrCombiner = MatrArr[MP_ATT_COMBINER];
        auto &matrQ = MatrArr[MP_ATT_Q];
        auto &matrK = MatrArr[MP_ATT_K];
        auto &matrGate = MatrArr[MP_ATT_GATE];
        auto &matrV = MatrArr[MP_ATT_V];

        TAttentionComputer attComp(Dims, AlibiSlope, 1.0f);
        const TAttentionInfo &attInfo = common.AttArr[AttentionTypeId].Att;
        const TAttentionInfo &revAttInfo = common.AttArr[AttentionTypeId].RevAtt;

        // recompute forward pass (could keep them)
        TArray2D<float> qSrc = MulForward(normState, GetData(matrQ));
        TArray2D<float> kSrc = MulForward(normState, GetData(matrK));
        TArray2D<float> gate = MulForward(normState, GetDataNoScale(matrGate));
        TArray2D<float> vSrc = MulForward(normState, GetData(matrV));

        TArray2D<float> q = NormalizeState(qSrc, headCount);
        TArray2D<float> k = kSrc;
        TArray2D<float> v = NormalizeState(vSrc, headCount);
        ApplyRope(common.RopeBuf, 1, &q);
        ApplyRope(common.RopeBuf, 1, &k);

        TArray2D<float> valLookup;
        attComp.ComputeValLookup(len, Dims, q, k, v, attInfo, &valLookup);
        //PrintArr(10, valLookup);

        TArray2D<float> rg;
        LoLU(gate, valLookup, &rg);

        // combiner backprop
        TArray2D<float> drg;
        InitDeltaMatrix(&drg, rg);
        MatmulBackprop(matrCombiner, rg, *pGrad, UpdateLayers, &drg);

        // lolu backprop
        TArray2D<float> dgate;
        TArray2D<float> dValLookup;
        BackpropLoLU(gate, valLookup, drg, &dgate, &dValLookup);

        // att backprop
        TArray2D<float> dScale;
        ComputeDScale(Dims, valLookup, dValLookup, &dScale);

        TAttentionComputer::TGradData gradData(Dims, q, k, v, dValLookup, dScale);

        TArray2D<float> dQ;
        attComp.GradQ(len, Dims, gradData, attInfo, &dQ);
        TArray2D<float> dK;
        TArray2D<float> dV;
        attComp.GradK(len, Dims, gradData, revAttInfo, &dK, &dV);
        ApplyRope(common.RopeBuf, -1, &dQ);
        ApplyRope(common.RopeBuf, -1, &dK);

        NormalizeStateBackward(qSrc, headCount, dQ, &dQ);
        NormalizeStateBackward(vSrc, headCount, dV, &dV);

        // matmul backprop
        MatmulBackprop(matrQ, normState, dQ, UpdateLayers, &dNormState);
        MatmulBackprop(matrK, normState, dK, UpdateLayers, &dNormState);
        MatmulBackprop(matrV, normState, dV, UpdateLayers, &dNormState);
        MatmulBackpropNoScale(matrGate, normState, dgate, UpdateLayers, &dNormState);

        TArray2D<float> deltaStateGrad;
        NormalizeStateBackward(prevState, dim / STATE_NORM_TILE, dNormState, &deltaStateGrad);
        AddScaledMatrix(pGrad, deltaStateGrad, 1);
    }

    TCudaLayerBase *CreateCudaLayer(TPtrArg<TCudaMemoryAllocator> cudaMem, TPtrArg<TMultiDeviceBuffers> multiBuffers, TStream &stream,
        yint deviceId, yint len, const TVector<TIntrusivePtr<NCuda::ICudaModelMatrixBase<TFastModelFloat>>> &matrArr,
        TCudaLayerPools &pools) override
    {
        return new TCudaAttLayer(
            multiBuffers, this, deviceId, Dims, UpdateLayers, AlibiSlope, AttentionTypeId, len, matrArr, pools);
    }

public:
    TAttLayer(const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr, float alibiSlope,
        yint attTypeId)
        : TLayerBase(modelDescr, matrArr), Dims(modelDescr.Dims), AlibiSlope(alibiSlope), AttentionTypeId(attTypeId)
    {
    }
};


TIntrusivePtr<TLayerBase> CreateAttLayer(const TModelDescr &modelDescr,
    const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr, float alibiSlope, yint attTypeId)
{
    Y_VERIFY(modelDescr.AttentionTypeArr[attTypeId].Type == TAttentionType::CASUAL && "only casual supported so far");
    return new TAttLayer(modelDescr, matrArr, alibiSlope, attTypeId);
}
