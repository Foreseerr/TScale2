#include <util/pch.h>
#define KERNEL_UNIT "layer_ffn/"
#include "layer_ffn.cuh"
#include "layer.h"
#include "layer_cuda_base.cuh"
#include "cpu_util.h"
#include "cfg_precision.h"
#include "gpu_layer_norm.cuh"
#include "gpu_lolu.cuh"
#include <gpt/model_params/model_dim.h>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace NCuda
{
class TCudaFFLayer : public TCudaLayerBase
{
    TCuda2DArray<half> Wide;
    TCuda2DArray<float> WideScale;
    TCuda2DArray<half> Gate;
    TIntrusivePtr<TMultiDevice2DArray<TNormStateFloat>> RG;
    TCuda2DArray<TFastGradientFloat> DWide;
    TCuda2DArray<TFastGradientFloat> DGate;
    TIntrusivePtr<TMultiDeviceFRed2DArray<half>> DRG;
    TMatMulBuffers MatmulExpand;
    TMatMulBuffers MatmulGate;
    TMatMulBuffers MatmulContract;
    TCudaLayerPools Pools;

public:
    TCudaFFLayer(TPtrArg<TMultiDeviceBuffers> multiBuffers, const void *groupPtr, yint deviceId, const TModelDims &dims, bool updateLayers,
        yint len, const TVector<TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>>> &matrArr, TCudaLayerPools &pools)
        : TCudaLayerBase(multiBuffers, deviceId, dims, updateLayers, matrArr), Pools(pools)
    {
        yint dgSize = MultiBuffers->GetDgSize();
        int ffnDimFull = Dims.GetFfnDim();
        int ffnDim = ffnDimFull / dgSize;

        Wide.AllocateCuda(ffnDim, len, pools.Fwd);
        WideScale.AllocateCuda(len, MMTiles(ffnDim), pools.Fwd);
        Gate.AllocateCuda(ffnDim, len, pools.Fwd);
        RG = MultiBuffers->Fab().Create2DArray<TNormStateFloat>(Sprintf("%p-RG", groupPtr));
        RG->AllocateCuda(deviceId, ffnDimFull, len, pools.Fwd);
        DWide.AllocateCuda(ffnDim, len, pools.Bwd);
        DGate.AllocateCuda(ffnDim, len, pools.Bwd);
        DRG = MultiBuffers->Fab().CreateFRed2DArray<half>(Sprintf("%p-DRG", groupPtr));
        DRG->AllocateCuda(deviceId, ffnDimFull, len, pools.Bwd);
        MatmulExpand.AllocateCuda(MatrArr[MP_FFN_EXPAND], len, pools.Bwd);
        MatmulGate.AllocateCuda(MatrArr[MP_FFN_GATE], len, pools.Bwd);
        MatmulContract.AllocateCuda(MatrArr[MP_FFN_CONTRACT], len, pools.Bwd);
    }

    void ComputeRG(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState)
    {
        int stateDim = Dims.Dim;
        float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;

        MatmulExpand.Forward<TStoreRowTileNormalize>(c, pParams, normState, &Wide).Read(stateNormScale, FFN_VEC_SCALE).Write(&WideScale);
        MatmulGate.Forward<TStoreScaled>(c, pParams, normState, &Gate).Read(nullptr, stateNormScale);

        auto rgWin = XSplitWindow(RG->GetData(DeviceId));
        CudaCall(c, MatrixLoLU<TNormStateFloat>)
            .Grid(rgWin.GetXSize() / MM_TILE, MMTiles(pParams->Len))
            .Read(pParams->Len, Gate, Wide)
            .Write(&rgWin);
        MultiBuffers->Op().AllGatherXSplit(c, RG, DeviceId);
    }

    void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState, TCuda2DArray<TStateFloat> *pState,
        TFragmentStates *pNextNormState) override
    {
        Pools.SetMemPools(c);
        ComputeRG(c, pParams, normState);

        int ffnDim = Dims.GetFfnDim();
        float contractNormScale = CalcDotScale(ffnDim) * MODEL_DISCR_SCALE * FFN_VEC_SCALE;
        MatmulContract.Forward<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, RG->GetData(DeviceId), pState)
            .Read(contractNormScale, MatmulContract.GetScale(), STATE_VEC_SCALE)
            .Write(&pNextNormState->GetLocal(), &pNextNormState->GetScaleLocal());
    }


    TCudaPOD<float> AddBackward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState,
        TCuda2DArray<TFastGradientFloat> *pStateGrad, TCudaPOD<float> gradScale,
        TPtrArg<TMultiDeviceFRed2DArray<float>> dNormStatePtr) override
    {
        Pools.SetMemPools(c);
        ComputeRG(c, pParams, normState);

        // contract mulback
        auto bmContract = MatmulContract.BackpropMulti(c, pParams, *pStateGrad);
        Backprop(c, pParams, MultiArg() + bmContract, &DRG->GetData(DeviceId));
        MultiBuffers->Op().ReduceXSplit(c, DRG, pParams->Len, DeviceId);

        auto drgWin = XSplitWindow(DRG->GetData(DeviceId));
        CudaCall(c, BackpropRowTileNormalizeLoLU<half, TFastGradientFloat>)
            .Grid(drgWin.GetXSize() / MM_TILE, pParams->LenRound)
            .Read(pParams->Len, Gate, Wide, FFN_VEC_SCALE, WideScale, drgWin)
            .Write(&DGate, &DWide);

        TCuda2DArray<float> &dNormState = dNormStatePtr->GetData(DeviceId);
        auto bm1 = MatmulExpand.BackpropMulti(c, pParams, DWide);
        auto bm2 = MatmulGate.BackpropMulti(c, pParams, DGate);
        Backprop(c, pParams, MultiArg() + bm1 + bm2, &dNormState);

        MultiBuffers->Op().ReduceXSplit(c, dNormStatePtr, pParams->Len, DeviceId);

        MatmulContract.BackpropDelta(c, pParams, RG->GetData(DeviceId), *pStateGrad, gradScale, UpdateLayers);
        MatmulExpand.BackpropDelta(c, pParams, normState, DWide, gradScale, UpdateLayers);
        MatmulGate.BackpropDelta(c, pParams, normState, DGate, gradScale, UpdateLayers);

        return MatmulContract.GetScale();
    }

    void AddDelta(TPtrArg<TGraph> c, EBackpropMode bm) override
    {
        if (UpdateLayers) {
            MatmulContract.AddDelta(c, bm);
            MatmulExpand.AddDelta(c, bm);
            MatmulGate.AddDelta(c, bm);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaFFLayerInfer : public TCudaInferLayerBase
{
    TCuda2DArray<half> Wide;
    TCuda2DArray<float> WideScale;
    TCuda2DArray<half> Gate;
    TCuda2DArray<TNormStateFloat> RG;
    TInferMatMul MatmulExpand;
    TInferMatMul MatmulGate;
    TInferMatMul MatmulContract;
    TIntrusivePtr<TCudaMemoryPool> Pool;

public:
    TCudaFFLayerInfer(const TModelDims &dims, yint len, const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr,
        TPtrArg<TCudaMemoryPool> pool)
        : TCudaInferLayerBase(dims), Pool(pool)
    {
        int ffnDim = Dims.GetFfnDim();

        Wide.AllocateCuda(ffnDim, len, pool);
        WideScale.AllocateCuda(len, MMTiles(ffnDim), pool);
        Gate.AllocateCuda(ffnDim, len, pool);
        RG.AllocateCuda(ffnDim, len, pool);
        MatmulExpand.Init(matrArr[MP_FFN_EXPAND]);
        MatmulGate.Init(matrArr[MP_FFN_GATE]);
        MatmulContract.Init(matrArr[MP_FFN_CONTRACT]);
    }

    void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TStateFloat> *pState,
        TCuda2DArray<TNormStateFloat> *pNormState, TCuda2DArray<float> *pNormStateScale) override
    {
        c->SetMemPool(Pool);
        int stateDim = Dims.Dim;
        int ffnDim = Dims.GetFfnDim();
        float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;

        TCuda2DArray<TNormStateFloat> &normState = *pNormState;
        MatmulExpand.Forward<TStoreRowTileNormalize>(c, pParams, normState, &Wide).Read(stateNormScale, FFN_VEC_SCALE).Write(&WideScale);
        MatmulGate.Forward<TStoreScaled>(c, pParams, normState, &Gate).Read(nullptr, stateNormScale);

        CudaCall(c, MatrixLoLU<TNormStateFloat>).Grid(ffnDim / MM_TILE, MMTiles(pParams->Len)).Read(pParams->Len, Gate, Wide).Write(&RG);

        float contractNormScale = CalcDotScale(ffnDim) * MODEL_DISCR_SCALE * FFN_VEC_SCALE;
        MatmulContract.Forward<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, RG, pState)
            .Read(contractNormScale, MatmulContract.GetScale(), STATE_VEC_SCALE)
            .Write(pNormState, pNormStateScale);
    }
};


TIntrusivePtr<TCudaInferLayerBase> CreateFFLayerInference(const TModelDescr &modelDescr, yint len,
    const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr, TPtrArg<TCudaMemoryPool> pool)
{
    return new TCudaFFLayerInfer(modelDescr.Dims, len, matrArr, pool);
}
}
using namespace NCuda;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TFFLayer : public TLayerBase
{
    TModelDims Dims;

    void ComputeForward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pState) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        yint ffnDim = Dims.GetFfnDim();

        *pState = prevState;
        TArray2D<float> normState = NormalizeState(*pState, dim / STATE_NORM_TILE);

        auto &matrExpand = MatrArr[MP_FFN_EXPAND];
        auto &matrGate = MatrArr[MP_FFN_GATE];
        auto &matrContract = MatrArr[MP_FFN_CONTRACT];

        TArray2D<float> wide = MulForward(normState, GetData(matrExpand));
        TArray2D<float> wideNorm = NormalizeState(wide, MMTiles(ffnDim));
        TArray2D<float> gate = MulForward(normState, GetDataNoScale(matrGate));
        TArray2D<float> rg;
        LoLU(gate, wideNorm, &rg);

        TArray2D<float> deltaState = MulForward(rg, GetData(matrContract));
        AddScaledMatrix(pState, deltaState, 1);
    }

    void ComputeBackward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pGrad) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        yint ffnDim = Dims.GetFfnDim();

        TArray2D<float> normState = NormalizeState(prevState, dim / STATE_NORM_TILE);
        TArray2D<float> dNormState;
        InitDeltaMatrix(&dNormState, normState);

        auto &matrExpand = MatrArr[MP_FFN_EXPAND];
        auto &matrGate = MatrArr[MP_FFN_GATE];
        auto &matrContract = MatrArr[MP_FFN_CONTRACT];

        TArray2D<float> wide = MulForward(normState, GetData(matrExpand));
        TArray2D<float> wideNorm = NormalizeState(wide, MMTiles(ffnDim));
        TArray2D<float> gate = MulForward(normState, GetDataNoScale(matrGate));
        TArray2D<float> rg;
        LoLU(gate, wideNorm, &rg);

        TArray2D<float> drg;
        InitDeltaMatrix(&drg, rg);
        MatmulBackprop(matrContract, rg, *pGrad, UpdateLayers, &drg);

        TArray2D<float> dgate;
        TArray2D<float> dwide;
        BackpropLoLU(gate, wideNorm, drg, &dgate, &dwide);
        NormalizeStateBackward(wide, MMTiles(ffnDim), dwide, &dwide);

        MatmulBackprop(matrExpand, normState, dwide, UpdateLayers, &dNormState);
        MatmulBackpropNoScale(matrGate, normState, dgate, UpdateLayers, &dNormState);

        TArray2D<float> deltaStateGrad;
        NormalizeStateBackward(prevState, dim / STATE_NORM_TILE, dNormState, &deltaStateGrad);
        AddScaledMatrix(pGrad, deltaStateGrad, 1);
    }

    TCudaLayerBase *CreateCudaLayer(TPtrArg<TCudaMemoryAllocator> cudaMem, TPtrArg<TMultiDeviceBuffers> multiBuffers, TStream &stream,
        yint deviceId, yint len, const TVector<TIntrusivePtr<NCuda::ICudaModelMatrixBase<TFastModelFloat>>> &matrArr,
        TCudaLayerPools &pools) override
    {
        return new NCuda::TCudaFFLayer(multiBuffers, this, deviceId, Dims, UpdateLayers, len, matrArr, pools);
    }

public:
    TFFLayer(const TModelDescr &modelDescr,
        const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr)
        : TLayerBase(modelDescr, matrArr), Dims(modelDescr.Dims)
    {
    }
};


TIntrusivePtr<TLayerBase> CreateFFLayer(
    const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr)
{
    return new TFFLayer(modelDescr, matrArr);
}
