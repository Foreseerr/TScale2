#include <util/pch.h>
#define KERNEL_UNIT "layer_att_3lin/"
#include "layer_att_3lin.cuh"
#include "gpu_att_infer.cuh"
#include "layer.h"
#include "layer_cuda_base.cuh"
#include "cpu_util.h"
#include "gpu_att_fp16.cuh"
#include "gpu_layer_norm.cuh"
#include "gpu_cross_attn.cuh"
#include "layer_att_reference.h"
#include "layer_att_3lin_combiner.cuh"
#include <gpt/model_params/model_dim.h>
#include <lib/hp_timer/hp_timer.h>


namespace NCuda
{
constexpr float ATT_SCALE_MULT = 4.0f;
//constexpr float ATT_SCALE_MULT = 16.0f;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TFixedKScaleBuf
{
    TCudaVector<float> AttScaleMult;
    TCuda2DArray<float> KscaleFixed;

    void FillConstantBuffers(TStream &stream, yint keyCount, yint headCount)
    {
        AttScaleMult.Allocate(1);
        TVector<float> attScaleMult;
        attScaleMult.resize(1, ATT_SCALE_MULT);
        Put(stream, &AttScaleMult, attScaleMult);

        KscaleFixed.Allocate(keyCount, headCount);
        TArray2D<float> k;
        k.SetSizes(keyCount, headCount);
        k.FillEvery(K_VEC_SCALE);
        Put(stream, &KscaleFixed, k);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TCuda3LinAttLayer : public TCudaLayerBase
{
    float AlibiSlope = 0;
    yint AttentionTypeId = 0;
    bool IsCross = false;
    TFixedKScaleBuf KScaleBuf;
    // cross attn buffers
    TCuda2DArray<TNormStateFloat> NormStateShuffled;
    TCuda2DArray<TNormStateFloat> CCShuffled;
    TCuda2DArray<half> DCCShuffled;
    TCuda2DArray<float> DNormStateShuffled;
    // Q
    TCuda2DArray<float> Qscale;
    TCuda2DArray<TAttVecFloat> Q;
    TCuda2DArray<half> Qsrc;
    // K
    TCuda2DArray<float> Kscale;
    TCuda2DArray<TAttVecFloat> K;
    // V
    TCuda2DArray<float> Vscale;
    TCuda2DArray<TAttVecFloat> V;
    // rest
    TCuda2DArray<half> AttGate;
    TCuda2DArray<float> AttGateScale;
    TCuda2DArray<half> ValLookup;
    TCuda2DArray<float> SumWeightLog;
    TCuda2DArray<float> DScale;
    TIntrusivePtr<TMultiDevice2DArray<TNormStateFloat>> CC;
    // grad
    TCuda2DArray<TFastGradientFloat> DQ16;
    TCuda2DArray<TFastGradientFloat> DK16;
    TCuda2DArray<TFastGradientFloat> DV16;
    TIntrusivePtr<TMultiDeviceFRed2DArray<half>> DCC;
    TCuda2DArray<TFastGradientFloat> DAttGate;
    TCuda2DArray<half> DValLookup16;
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

    template <class T>
    void ShuffleForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<T> &src, TCuda2DArray<T> *pDst)
    {
        Y_ASSERT(src.GetXSize() == pDst->GetXSize());
        CudaCall(c, CrossAttentionForward<T>)
            .Grid(pDst->GetXSize() / MM_TILE, MMTiles(pParams->Len))
            .Read(pParams->Len, src, pParams->CrossAttnShuffle)
            .Write(pDst);
    }

    template <class T>
    void ShuffleBackward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<T> &src, TCuda2DArray<T> *pDst)
    {
        Y_ASSERT(src.GetXSize() == pDst->GetXSize());
        CudaCall(c, CrossAttentionBackward<T>)
            .Grid(pDst->GetXSize() / MM_TILE, MMTiles(pParams->Len))
            .Read(pParams->Len, src, pParams->CrossAttnShuffle)
            .Write(pDst);
    }

public:
    TCuda3LinAttLayer(TPtrArg<TCudaMemoryAllocator> cudaMem, TPtrArg<TMultiDeviceBuffers> multiBuffers, const void *groupPtr, yint deviceId,
        TStream &stream, const TModelDims &dims, bool updateLayers, float alibiSlope, yint attentionTypeId, bool isCross, yint len,
        const TVector<TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>>> &matrArr, TCudaLayerPools &pools)
        : TCudaLayerBase(multiBuffers, deviceId, dims, updateLayers, matrArr), AlibiSlope(alibiSlope), AttentionTypeId(attentionTypeId),
          IsCross(isCross), Pools(pools)
    {
        yint qSum = GetDim(Dims.GetQSum());
        yint ttSum = GetDim(Dims.GetTTSum());
        yint headCount = GetDim(Dims.HeadCount);
        yint ttSumFull = Dims.GetTTSum();
        yint kvRep = dims.GetKVrep();

        KScaleBuf.FillConstantBuffers(stream, len, headCount);
        if (IsCross) {
            NormStateShuffled.AllocateCuda(Dims.Dim, len, pools.Fwd);
            CCShuffled.AllocateCuda(ttSumFull * kvRep, len, pools.Fwd);
            DCCShuffled.AllocateCuda(ttSumFull * kvRep, len, pools.Bwd);
            DNormStateShuffled.AllocateCuda(Dims.Dim, len, pools.Bwd);
        }
        //
        Qscale.AllocateCuda(len, headCount, pools.Fwd);
        Q.AllocateCuda(qSum, len, pools.Fwd);
        Qsrc.AllocateCuda(qSum, len, pools.Fwd);
        Kscale.AllocateCuda(len, headCount, pools.Fwd);
        K.AllocateCuda(qSum, len, pools.Fwd);
        Vscale.AllocateCuda(len, headCount, pools.Fwd);
        V.AllocateCuda(ttSum, len, pools.Fwd);
        AttGate.AllocateCuda(ttSum, len, pools.Fwd);
        AttGateScale.AllocateCuda(len, headCount, pools.Fwd);
        ValLookup.AllocateCuda(ttSum, len, pools.Fwd);
        SumWeightLog.AllocateCuda(len, headCount, pools.Fwd);
        DScale.AllocateCuda(len, headCount, pools.Bwd);
        CC = MultiBuffers->Fab().Create2DArray<TNormStateFloat>(Sprintf("%p-CC", groupPtr));
        CC->AllocateCuda(DeviceId, ttSumFull * kvRep, len, pools.Fwd);
        //
        DQ16.AllocateCuda(qSum, len, pools.Bwd);
        DK16.AllocateCuda(qSum, len, pools.Bwd);
        DV16.AllocateCuda(ttSum, len, pools.Bwd);
        DCC = MultiBuffers->Fab().CreateFRed2DArray<half>(Sprintf("%p-DCC", groupPtr));
        DCC->AllocateCuda(DeviceId, ttSumFull * kvRep, len, pools.Bwd);
        DAttGate.AllocateCuda(ttSum, len, pools.Bwd);
        DValLookup16.AllocateCuda(ttSum, len, pools.Bwd);

        MatmulQ.AllocateCuda(MatrArr[MP_ATT_Q], len, pools.Bwd);
        MatmulK.AllocateCuda(MatrArr[MP_ATT_K], len, pools.Bwd);
        MatmulV.AllocateCuda(MatrArr[MP_ATT_V], len, pools.Bwd);
        MatmulAttGate.AllocateCuda(MatrArr[MP_ATT_GATE], len, pools.Bwd);
        MatmulCombiner.AllocateCuda(MatrArr[MP_ATT_COMBINER], len, pools.Bwd);
    }


    void ComputeCC(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState)
    {
        yint stateDim = Dims.Dim;
        yint ttSum = GetDim(Dims.GetTTSum());
        yint headCount = GetDim(Dims.HeadCount);
        yint kvRep = Dims.GetKVrep();
        float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;

        TAttentionGroupData<ATT_GROUP> &attGD = *pParams->AttGDArr[AttentionTypeId];
        // TCudaPOD<float> scaleK = MatmulK.GetScale();
        TCudaPOD<float> scaleK = KScaleBuf.AttScaleMult.GetElement(0);

        MatmulQ.Forward<TStoreRowTileNormalize>(c, pParams, normState, &Q).Read(stateNormScale, Q_VEC_SCALE).Write(&Qscale);
        MatmulK.Forward<TStoreRowTileNormalize>(c, pParams, normState, &K).Read(stateNormScale, K_VEC_SCALE).Write(&Kscale);
        MatmulV.Forward<TStoreRowTileNormalize>(c, pParams, normState, &V).Read(stateNormScale, V_VEC_SCALE).Write(&Vscale);
        MatmulAttGate.Forward<TStoreRowTileNormalize>(c, pParams, normState, &AttGate)
            .Read(stateNormScale, V_VEC_SCALE)
            .Write(&AttGateScale);

        CudaCall(c, Fp16Att)
            .Grid(headCount, DivCeil(pParams->Len, ATT_GROUP, MM_TILE))
            .Read(scaleK, Q, K, KScaleBuf.KscaleFixed)
            .Read(V)
            .Read(attGD.AttSpans, attGD.AttSpanPtr, AlibiSlope)
            .Write(&SumWeightLog, &ValLookup);

        auto ccWin = XSplitWindow(CC->GetData(DeviceId));
        CudaCall(c, KVProductKernel<TNormStateFloat>)
            .Grid(ttSum / MM_TILE, pParams->LenRound)
            .Read(pParams->Len, (int)kvRep, AttGate, ValLookup)
            .Write(&ccWin);
        MultiBuffers->Op().AllGatherXSplit(c, CC, DeviceId);
    }

    void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState, TCuda2DArray<TStateFloat> *pState,
        TFragmentStates *pNextNormState) // override
    {
        Y_VERIFY(ATT_TYPE == ATT_FP16);
        Pools.SetMemPools(c);
        yint ttSumFull = Dims.GetTTSum();
        yint kvRep = Dims.GetKVrep();

        TCuda2DArray<TNormStateFloat> *pCCbuf = 0;
        if (IsCross) {
            ShuffleForward(c, pParams, normState, &NormStateShuffled);
            ComputeCC(c, pParams, NormStateShuffled);
            ShuffleBackward(c, pParams, CC->GetData(DeviceId), &CCShuffled);
            pCCbuf = &CCShuffled;

        } else {
            ComputeCC(c, pParams, normState);
            pCCbuf = &CC->GetData(DeviceId);
        }

        float combinerNormScale = CalcDotScale(ttSumFull * kvRep) * MODEL_DISCR_SCALE * V_VEC_SCALE;
        MatmulCombiner.Forward<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, *pCCbuf, pState)
            .Read(combinerNormScale, MatmulCombiner.GetScale(), STATE_VEC_SCALE)
            .Write(&pNextNormState->GetLocal(), &pNextNormState->GetScaleLocal());
    }


    TCudaPOD<float> AddBackward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState,
        TCuda2DArray<TFastGradientFloat> *pStateGrad, TCudaPOD<float> gradScale,
        TPtrArg<TMultiDeviceFRed2DArray<float>> dNormStatePtr) override
    {
        Pools.SetMemPools(c);
        yint ttSum = GetDim(Dims.GetTTSum());
        yint headCount = GetDim(Dims.HeadCount);
        yint kvRep = Dims.GetKVrep();

        TAttentionGroupData<ATT_GROUP> &attGD = *pParams->AttGDArr[AttentionTypeId];
        // TCudaPOD<float> scaleK = MatmulK.GetScale();
        TCudaPOD<float> scaleK = KScaleBuf.AttScaleMult.GetElement(0);

        TCuda2DArray<TNormStateFloat> *pNormState = 0;
        TCuda2DArray<TNormStateFloat> *pCC = 0;
        if (IsCross) {
            ShuffleForward(c, pParams, normState, &NormStateShuffled);
            ComputeCC(c, pParams, NormStateShuffled);
            ShuffleBackward(c, pParams, CC->GetData(DeviceId), &CCShuffled);
            pNormState = &NormStateShuffled;
            pCC = &CCShuffled;
        } else {
            ComputeCC(c, pParams, normState);
            pNormState = &normState;
            pCC = &CC->GetData(DeviceId);
        }

        // combiner derivatives
        auto bmCombiner = MatmulCombiner.BackpropMulti(c, pParams, *pStateGrad);
        Backprop(c, pParams, MultiArg() + bmCombiner, &DCC->GetData(DeviceId));
        MultiBuffers->Op().ReduceXSplit(c, DCC, pParams->Len, DeviceId);

        TCuda2DArray<half> *pDCC = 0;
        if (IsCross) {
            ShuffleForward(c, pParams, DCC->GetData(DeviceId), &DCCShuffled);
            pDCC = &DCCShuffled;
        } else {
            pDCC = &DCC->GetData(DeviceId);
        }

        // attention derivative
        auto dcc = XSplitWindow(*pDCC);
        CudaCall(c, KVProductBackpropKernel<TFastGradientFloat>)
            .Grid(ttSum / MM_TILE, pParams->LenRound)
            .Read(pParams->Len, (int)kvRep, AttGate, ValLookup, dcc)
            .Write(&DAttGate, &DValLookup16);

        CudaCall(c, CalcDScale<TT_DIM>).Grid(headCount, pParams->LenRound).Read(ValLookup, DValLookup16).Write(&DScale);

        CudaCall(c, Fp16AttGradQ<TStoreAttBackNormalize>)
            .Grid(headCount, DivCeil(pParams->Len, ATT_GROUP, MM_TILE))
            .Read(scaleK, Q, K, KScaleBuf.KscaleFixed)
            .Read(V)
            .Read(DValLookup16)
            .Read(DScale, SumWeightLog)
            .Read(attGD.AttSpans, attGD.AttSpanPtr, AlibiSlope)
            .Write(&DQ16)
            .Struct()
            .Read(Q, Qscale);

        CudaCall(c, Fp16AttGradKV<TStoreAttBackNormalize, TStoreAttBackNormalize>)
            .Grid(headCount, DivCeil(pParams->Len, ATT_GROUP, MM_TILE))
            .Read(scaleK, Q, K, KScaleBuf.KscaleFixed)
            .Read(V)
            .Read(DValLookup16)
            .Read(DScale, SumWeightLog)
            .Read(attGD.RevAttSpans, attGD.RevAttSpanPtr, AlibiSlope)
            .Write(&DK16, &DV16)
            .Struct()
            .Read(K, Kscale)
            .Struct()
            .Read(V, Vscale);

        // can be merged with KVProduct backprop
        CudaCall(c, BackpropRowTileNormalize<half, TFastGradientFloat>)
            .Grid(headCount, pParams->Len)
            .Read(AttGate, AttGateScale)
            .Write(&DAttGate);

        TCuda2DArray<float> &dNormState = IsCross ? DNormStateShuffled : dNormStatePtr->GetData(DeviceId);
        auto bm1 = MatmulQ.BackpropMulti(c, pParams, DQ16);
        auto bm2 = MatmulK.BackpropMulti(c, pParams, DK16);
        auto bm3 = MatmulV.BackpropMulti(c, pParams, DV16);
        auto bm4 = MatmulAttGate.BackpropMulti(c, pParams, DAttGate);
        Backprop(c, pParams, MultiArg() + bm1 + bm2 + bm3 + bm4, &dNormState);
        if (IsCross) {
            ShuffleBackward(c, pParams, DNormStateShuffled, &dNormStatePtr->GetData(DeviceId));
        }
        MultiBuffers->Op().ReduceXSplit(c, dNormStatePtr, pParams->Len, DeviceId);

        MatmulCombiner.BackpropDelta(c, pParams, *pCC, *pStateGrad, gradScale, UpdateLayers);
        MatmulQ.BackpropDelta(c, pParams, *pNormState, DQ16, gradScale, UpdateLayers);
        MatmulK.BackpropDelta(c, pParams, *pNormState, DK16, gradScale, UpdateLayers);
        MatmulV.BackpropDelta(c, pParams, *pNormState, DV16, gradScale, UpdateLayers);
        MatmulAttGate.BackpropDelta(c, pParams, *pNormState, DAttGate, gradScale, UpdateLayers);

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
class TCuda3LinAttLayerInfer : public TCudaInferLayerBase
{
    float AlibiSlope = 0;
    int MaxWidth = 0;
    bool IsCross = false;
    TFixedKScaleBuf KScaleBuf;
    // Q
    TCuda2DArray<float> Qscale;
    TCuda2DArray<TAttVecFloat> Q;
    // K
    TCuda2DArray<float> Kscale;
    TCuda2DArray<TAttVecFloat> K;
    TCuda2DArray<TAttVecFloat> KeyCache;
    // V
    TCuda2DArray<float> Vscale;
    TCuda2DArray<TAttVecFloat> V;
    TCuda2DArray<TAttVecFloat> ValueCache;
    // rest
    TCuda2DArray<half> AttGate;
    TCuda2DArray<float> AttGateScale;
    TCuda2DArray<half> ValLookup;
    TCuda2DArray<float> SumWeightLog;
    TCuda2DArray<TNormStateFloat> CC;
    //
    TInferMatMul MatmulQ;
    TInferMatMul MatmulK;
    TInferMatMul MatmulV;
    TInferMatMul MatmulAttGate;
    TInferMatMul MatmulCombiner;
    //
    TIntrusivePtr<TCudaMemoryPool> Pool;

public:
    TCuda3LinAttLayerInfer(TStream &stream, const TModelDims &dims, float alibiSlope, const TAttentionType &attnType, yint len,
        yint kvCacheSize, const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr, TPtrArg<TCudaMemoryPool> pool)
        : TCudaInferLayerBase(dims), AlibiSlope(alibiSlope), MaxWidth(attnType.Width), Pool(pool)
    {
        yint qSum = dims.GetQSum();
        yint ttSum = dims.GetTTSum();
        yint headCount = dims.HeadCount;
        yint kvRep = dims.GetKVrep();

        IsCross = (attnType.Type == TAttentionType::CROSS_BATCH);
        KScaleBuf.FillConstantBuffers(stream, kvCacheSize, headCount);
        Qscale.AllocateCuda(len, headCount, pool);
        Q.AllocateCuda(qSum, len, pool);
        Kscale.AllocateCuda(len, headCount, pool);
        K.AllocateCuda(qSum, len, pool);
        Vscale.AllocateCuda(len, headCount, pool);
        V.AllocateCuda(ttSum, len, pool);
        AttGate.AllocateCuda(ttSum, len, pool);
        AttGateScale.AllocateCuda(len, headCount, pool);
        ValLookup.AllocateCuda(ttSum, len, pool);
        CC.AllocateCuda(ttSum * kvRep, len, pool);
        if (!IsCross) {
            KeyCache.AllocateCuda(qSum, kvCacheSize);
            ValueCache.AllocateCuda(ttSum, kvCacheSize);
        } else {
            SumWeightLog.AllocateCuda(len, headCount);
        }
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
        int stateDim = Dims.Dim;
        int qSum = Dims.GetQSum();
        int ttSum = Dims.GetTTSum();
        int headCount = Dims.HeadCount;
        int kvRep = Dims.GetKVrep();
        float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;
        TCuda2DArray<TNormStateFloat> &normState = *pNormState;
        TInferAttentionData &att = *pParams->InferAtt;
        // TCudaPOD<float> scaleK = MatmulK.GetScale();
        TCudaPOD<float> scaleK = KScaleBuf.AttScaleMult.GetElement(0);

        MatmulQ.Forward<TStoreRowTileNormalize>(c, pParams, normState, &Q).Read(stateNormScale, Q_VEC_SCALE).Write(&Qscale);
        MatmulK.Forward<TStoreRowTileNormalize>(c, pParams, normState, &K).Read(stateNormScale, K_VEC_SCALE).Write(&Kscale);
        MatmulV.Forward<TStoreRowTileNormalize>(c, pParams, normState, &V).Read(stateNormScale, V_VEC_SCALE).Write(&Vscale);
        MatmulAttGate.Forward<TStoreRowTileNormalize>(c, pParams, normState, &AttGate)
            .Read(stateNormScale, V_VEC_SCALE)
            .Write(&AttGateScale);

        if (IsCross) {
            TAttentionGroupData<ATT_GROUP> &attGD = *pParams->InferCrossAtt;
            CudaCall(c, Fp16Att)
                .Grid(headCount, DivCeil(pParams->Len, ATT_GROUP, MM_TILE))
                .Read(scaleK, Q, K, KScaleBuf.KscaleFixed)
                .Read(V)
                .Read(attGD.AttSpans, attGD.AttSpanPtr, AlibiSlope)
                .Write(&SumWeightLog, &ValLookup);

        } else {
            CudaCall(c, InferAtt<Q_DIM, TT_DIM, NO_ROPE>)
                .Grid(pParams->Len, headCount)
                .Read(Q)
                .Read(KeyCache, scaleK, KScaleBuf.KscaleFixed)
                .Read(ValueCache)
                .Read(AlibiSlope, MaxWidth)
                .Read(att.ReadArr, att.ReadArrPtr, att.RopeBuf)
                .Write(&ValLookup);

            CudaCall(c, CopyToKVcache3Lin<Q_DIM, TT_DIM>)
                .Grid(pParams->Len)
                .Read(headCount)
                .Read(K, V)
                .Read(att.WriteArr)
                .Write(&KeyCache, &ValueCache);
        }

        CudaCall(c, KVProductKernel<TNormStateFloat>)
            .Grid(ttSum / MM_TILE, pParams->LenRound)
            .Read(pParams->Len, (int)kvRep, AttGate, ValLookup)
            .Write(&CC);

        float combinerNormScale = CalcDotScale(ttSum * kvRep) * MODEL_DISCR_SCALE * V_VEC_SCALE;
        MatmulCombiner.Forward<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, CC, pState)
            .Read(combinerNormScale, MatmulCombiner.GetScale(), STATE_VEC_SCALE)
            .Write(pNormState, pNormStateScale);
    }
};


TIntrusivePtr<TCudaInferLayerBase> Create3LinAttLayerInference(TStream &stream, const TModelDescr &modelDescr, float alibiSlope,
    const TAttentionType &attnType, yint len, yint kvCacheSize,
    const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr, TPtrArg<TCudaMemoryPool> pool)
{
    return new TCuda3LinAttLayerInfer(stream, modelDescr.Dims, alibiSlope, attnType, len, kvCacheSize, matrArr, pool);
}
}
using namespace NCuda;


///////////////////////////////////////////////////////////////////////////////////////////////////
// CPU Attention implementation
//
class T3LinAttLayer : public TLayerBase
{
    TModelDims Dims;
    float AlibiSlope = 0;
    yint AttentionTypeId = 0;
    bool IsCross = false;

    void ComputeForward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pState) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        yint headCount = Dims.HeadCount;
        yint kvRep = Dims.GetKVrep();

        *pState = prevState;
        if (IsCross) {
            CrossShuffleForward(pState, common.Cross.FwdShuffle);
        }
        TArray2D<float> normState = NormalizeState(*pState, dim / STATE_NORM_TILE);

        auto &matrCombiner = MatrArr[MP_ATT_COMBINER];
        auto &matrQ = MatrArr[MP_ATT_Q];
        auto &matrK = MatrArr[MP_ATT_K];
        auto &matrGate = MatrArr[MP_ATT_GATE];
        auto &matrV = MatrArr[MP_ATT_V];

        TAttentionComputer attComp(Dims, AlibiSlope, ATT_SCALE_MULT);
        const TAttentionInfo &attInfo = common.AttArr[AttentionTypeId].Att;

        TArray2D<float> qSrc = MulForward(normState, GetData(matrQ));
        TArray2D<float> kSrc = MulForward(normState, GetData(matrK));
        TArray2D<float> vSrc = MulForward(normState, GetData(matrV));
        TArray2D<float> gateSrc = MulForward(normState, GetData(matrGate));

        TArray2D<float> q = NormalizeState(qSrc, headCount);
        TArray2D<float> k = NormalizeState(kSrc, headCount);
        TArray2D<float> v = NormalizeState(vSrc, headCount);
        TArray2D<float> gate = NormalizeState(gateSrc, headCount);

        TArray2D<float> valLookup;
        attComp.ComputeValLookup(len, Dims, q, k, v, attInfo, &valLookup);

        TArray2D<float> cc;
        KVProduct(kvRep, gate, valLookup, &cc);

        TArray2D<float> deltaState = MulForward(cc, GetData(matrCombiner));
        AddScaledMatrix(pState, deltaState, 1);
        if (IsCross) {
            CrossShuffleBackward(pState, common.Cross.FwdShuffle);
        }
    }

    void ComputeBackward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pGrad) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        yint headCount = Dims.HeadCount;
        yint kvRep = Dims.GetKVrep();

        TArray2D<float> normState = NormalizeState(prevState, dim / STATE_NORM_TILE);
        if (IsCross) {
            CrossShuffleForward(&normState, common.Cross.FwdShuffle);
            CrossShuffleForward(pGrad, common.Cross.FwdShuffle);
        }
        TArray2D<float> dNormState;
        InitDeltaMatrix(&dNormState, normState);

        auto &matrCombiner = MatrArr[MP_ATT_COMBINER];
        auto &matrQ = MatrArr[MP_ATT_Q];
        auto &matrK = MatrArr[MP_ATT_K];
        auto &matrGate = MatrArr[MP_ATT_GATE];
        auto &matrV = MatrArr[MP_ATT_V];

        TAttentionComputer attComp(Dims, AlibiSlope, ATT_SCALE_MULT);
        const TAttentionInfo &attInfo = common.AttArr[AttentionTypeId].Att;
        const TAttentionInfo &revAttInfo = common.AttArr[AttentionTypeId].RevAtt;

        // recompute forward pass (could keep them)
        TArray2D<float> qSrc = MulForward(normState, GetData(matrQ));
        TArray2D<float> kSrc = MulForward(normState, GetData(matrK));
        TArray2D<float> vSrc = MulForward(normState, GetData(matrV));
        TArray2D<float> gateSrc = MulForward(normState, GetData(matrGate));

        TArray2D<float> q = NormalizeState(qSrc, headCount);
        TArray2D<float> k = NormalizeState(kSrc, headCount);
        TArray2D<float> v = NormalizeState(vSrc, headCount);
        TArray2D<float> gate = NormalizeState(gateSrc, headCount);

        TArray2D<float> valLookup;
        attComp.ComputeValLookup(len, Dims, q, k, v, attInfo, &valLookup);
        //PrintArr(10, valLookup);

        TArray2D<float> cc;
        KVProduct(kvRep, gate, valLookup, &cc);

        // combiner backprop
        TArray2D<float> dcc;
        InitDeltaMatrix(&dcc, cc);
        MatmulBackprop(matrCombiner, cc, *pGrad, UpdateLayers, &dcc);

        // combiner backprop
        TArray2D<float> dGate;
        TArray2D<float> dValLookup;
        KVProductBackprop(kvRep, gate, valLookup, dcc, &dGate, &dValLookup);

        // att backprop
        TArray2D<float> dScale;
        ComputeDScale(Dims, valLookup, dValLookup, &dScale);

        TAttentionComputer::TGradData gradData(Dims, q, k, v, dValLookup, dScale);

        TArray2D<float> dQ;
        attComp.GradQ(len, Dims, gradData, attInfo, &dQ);
        TArray2D<float> dK;
        TArray2D<float> dV;
        attComp.GradK(len, Dims, gradData, revAttInfo, &dK, &dV);

        NormalizeStateBackward(qSrc, headCount, dQ, &dQ);
        NormalizeStateBackward(kSrc, headCount, dK, &dK);
        NormalizeStateBackward(vSrc, headCount, dV, &dV);
        NormalizeStateBackward(gateSrc, headCount, dGate, &dGate);

        // matmul backprop
        MatmulBackprop(matrQ, normState, dQ, UpdateLayers, &dNormState);
        MatmulBackprop(matrK, normState, dK, UpdateLayers, &dNormState);
        MatmulBackprop(matrV, normState, dV, UpdateLayers, &dNormState);
        MatmulBackprop(matrGate, normState, dGate, UpdateLayers, &dNormState);

        if (IsCross) {
            CrossShuffleBackward(&dNormState, common.Cross.FwdShuffle);
            CrossShuffleBackward(pGrad, common.Cross.FwdShuffle);
        }
        TArray2D<float> deltaStateGrad;
        NormalizeStateBackward(prevState, dim / STATE_NORM_TILE, dNormState, &deltaStateGrad);
        AddScaledMatrix(pGrad, deltaStateGrad, 1);
    }

    TCudaLayerBase *CreateCudaLayer(TPtrArg<TCudaMemoryAllocator> cudaMem, TPtrArg<TMultiDeviceBuffers> multiBuffers, TStream &stream,
        yint deviceId, yint len, const TVector<TIntrusivePtr<NCuda::ICudaModelMatrixBase<TFastModelFloat>>> &matrArr,
        TCudaLayerPools &pools) override
    {
        return new TCuda3LinAttLayer(cudaMem, multiBuffers, this, deviceId, stream, Dims, UpdateLayers, AlibiSlope,
            AttentionTypeId, IsCross, len, matrArr, pools);
    }

public:
    T3LinAttLayer(const TModelDescr &modelDescr,
        const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr, float alibiSlope, yint attTypeId)
        : TLayerBase(modelDescr, matrArr), Dims(modelDescr.Dims), AlibiSlope(alibiSlope), AttentionTypeId(attTypeId)
    {
        IsCross = (modelDescr.AttentionTypeArr[attTypeId].Type == TAttentionType::CROSS_BATCH);
    }
};


TIntrusivePtr<TLayerBase> Create3LinAttLayer(const TModelDescr &modelDescr,
    const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr, float alibiSlope, yint attTypeId)
{
    return new T3LinAttLayer(modelDescr, matrArr, alibiSlope, attTypeId);
}
