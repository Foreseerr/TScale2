#pragma once
#include "cfg_precision.h"
#include "matmul_fwdbwd.cuh"
#include <gpt/att/att.h>
#include <gpt/matrix/base_cuda.cuh>
#include <gpt/matrix/delta_cuda.h>
#include <gpt/matrix/infer_matrix_cuda.cuh>
#include <gpt/model_params/model_dim.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/multi_device_buf.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int GROUP_SIZE>
struct TAttentionGroupData : public TThrRefBase
{
    TCudaVector<TAttentionSpanGroup<GROUP_SIZE>> AttSpans;
    TCudaVector<int> AttSpanPtr;
    TCudaVector<TAttentionSpanGroup<GROUP_SIZE>> RevAttSpans;
    TCudaVector<int> RevAttSpanPtr;

    void Allocate(int maxLen, int groupPerBlock)
    {
        int spanGroups = DivCeil(maxLen, GROUP_SIZE);
        AttSpans.Allocate(spanGroups * groupPerBlock);
        RevAttSpans.Allocate(spanGroups * groupPerBlock);
        AttSpanPtr.Allocate(spanGroups + 1);
        RevAttSpanPtr.Allocate(spanGroups + 1);
    }

    template <int N>
    void AssignAttentionGroups(TStream &stream, TAttentionInfoGrouped<N> *pGroups, yint attGroupCount, TCudaVector<TAttentionSpanGroup<N>> *pSpans, TCudaVector<int> *pSpanPtr)
    {
        while (pGroups->GetGroupCount() < attGroupCount) {
            pGroups->AddEmptySpanGroup();
        }
        Put(stream, pSpans, pGroups->SpanGroups);
        Put(stream, pSpanPtr, pGroups->SpanGroupPtr);
    }

    void Init(TStream &stream, yint len, TAttentionInfoGrouped<GROUP_SIZE> *pAttGroups,
        TAttentionInfoGrouped<GROUP_SIZE> *pRevAttGroups)
    {
        int attGroupCount = RoundUp(len, MM_TILE) / GROUP_SIZE;
        AssignAttentionGroups(stream, pAttGroups, attGroupCount, &AttSpans, &AttSpanPtr);
        AssignAttentionGroups(stream, pRevAttGroups, attGroupCount, &RevAttSpans, &RevAttSpanPtr);
    }

    void InitForwardOnly(TStream &stream, yint len, TAttentionInfoGrouped<GROUP_SIZE> *pAttGroups)
    {
        int attGroupCount = RoundUp(len, MM_TILE) / GROUP_SIZE;
        AssignAttentionGroups(stream, pAttGroups, attGroupCount, &AttSpans, &AttSpanPtr);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TInferAttentionData : public TThrRefBase
{
    // KV cache read indices
    TCudaVector<int> ReadArr;
    TCudaVector<int> ReadArrPtr;
    // KV cache write indices
    TCudaVector<int> WriteArr;
    // Rope index for current entry
    TCuda2DArray<TRopeFloat> RopeBuf;

    void Allocate(const TModelDescr &modelDescr, int maxLen, int maxHistory)
    {
        ReadArr.Allocate(maxLen * maxHistory);
        ReadArrPtr.Allocate(maxLen + 1);
        WriteArr.Allocate(maxLen);
        RopeBuf.Allocate(modelDescr.Dims.QDim, maxLen);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TComputeParams
{
    TModelDims Dims;
    TKernelParameter <int> Len;
    TKernelParameterRoundUp<int> LenRound;
    TKernelParameter<int> LenMoe;
    TCuda2DArray<TRopeFloat> RopeBuf;
    TVector<TIntrusivePtr<TAttentionGroupData<ATT_GROUP>>> AttGDArr;
    TIntrusivePtr<TInferAttentionData> InferAtt;
    TIntrusivePtr<TAttentionGroupData<ATT_GROUP>> InferCrossAtt;
    TCudaVector<int> CrossAttnShuffle;

    TComputeParams();
    void Allocate(const TModelDescr &modelDescr, yint maxLen);
    void InitRope(TStream &stream, const TModelDescr &modelDescr, int ropeLen);
    void Init(TStream &stream, yint len);
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TCudaLayerPools
{
    TIntrusivePtr<TCudaMemoryPool> Fwd;
    TIntrusivePtr<TCudaMemoryPool> Bwd;

    void CreateNonshared(TPtrArg<TCudaMemoryAllocator> cudaMem, TPtrArg<TCudaMemoryPool> root)
    {
        Fwd = cudaMem->CreateNonsharedPool(root);
        Bwd = cudaMem->CreateNonsharedPool(root);
    }
    void Create(TPtrArg<TCudaMemoryAllocator> cudaMem, TCudaLayerPools &root)
    {
        Fwd = cudaMem->CreatePool(root.Fwd);
        Bwd = cudaMem->CreatePool(root.Bwd);
    }
    void SetMemPools(TPtrArg<TGraph> c)
    {
        c->SetMemPool(Fwd);
        c->SetMemPool(Bwd);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TFragmentStates : public TThrRefBase
{
    yint DeviceId = 0;
    TIntrusivePtr<TMultiDevice2DArray<TNormStateFloat>> NormState;
    TIntrusivePtr<TMultiDevice2DArray<float>> StateScale;
    TCuda2DArrayFragment<TNormStateFloat> NormStateLocal;

public:
    void AllocateCuda(
        TPtrArg<TMultiDeviceBuffers> multiBuffers, const TString &id, yint deviceId, yint stateDim, yint len)
    {
        DeviceId = deviceId;
        NormState = multiBuffers->Fab().Create2DArray<TNormStateFloat>(id + "-NormState");
        NormState->AllocateCuda(deviceId, stateDim, len, null_ptr_arg);
        StateScale = multiBuffers->Fab().Create2DArray<float>(id + "-NormStateScale");
        StateScale->AllocateCuda(deviceId, len, MMTiles(stateDim), null_ptr_arg);
        //
        TCuda2DArray<TNormStateFloat> &buf = NormState->GetData(DeviceId);
        NormStateLocal = CalcXSplitWindow(buf, buf.GetXSize(), DeviceId, multiBuffers->GetDeviceGroup());
    }
    TCuda2DArray<TNormStateFloat> &GetFull() { return NormState->GetData(DeviceId); }
    TCuda2DArrayFragment<TNormStateFloat> &GetLocal() { return NormStateLocal; }
    TCuda2DArray<float> &GetScaleLocal() { return StateScale->GetData(DeviceId); }
    void Sync(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceBuffers> multiBuffers) { multiBuffers->Op().AllGatherXSplit(c, NormState, DeviceId); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TMatMulBuffers
{
    TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>> Transform;
    yint SrcDim = 0;
    yint DstDim = 0;
    TCudaPackedDeltaMatrix Delta;
    TInt8MatMulBwdBuffers Int8bwd;
    TFp8MatMulBwdBuffers Fp8bwd;
    TCuda2DArray<half> TransformFp16;

    void AllocateCuda(TPtrArg<ICudaModelMatrixBase<TFastModelFloat>> transform, yint len, TPtrArg<TCudaMemoryPool> pool)
    {
        Transform = transform.Get();
        SrcDim = Transform->GetLocalXSize();
        DstDim = Transform->GetLocalYSize();
        Delta.AllocateCuda(RoundUp(SrcDim, MM_TILE), RoundUp(DstDim, MM_TILE)); // allocate globally, no pool
        if (BWD_MATMUL_TYPE == MATMUL_INT8) {
            Int8bwd.AllocateCuda(SrcDim, DstDim, len, pool);
        } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            Fp8bwd.AllocateCuda(SrcDim, DstDim, len, pool);
        }
    }

    void AllocateTransformFp16(TPtrArg<TCudaMemoryPool> pool)
    {
        if (BWD_MATMUL_TYPE != MATMUL_FP16) {
            TransformFp16.AllocateCuda(RoundUp(SrcDim, MM_TILE), RoundUp(DstDim, MM_TILE), pool);
        }
    }

    template <class TStoreFunc, class TSrc, class TDst>
    TKernelOp &Forward(TPtrArg<TGraph> c, TComputeParams *pParams, TSrc &src, TDst *pRes)
    {
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            return MulForwardFp16<TStoreFunc>(c, pParams, SrcDim, DstDim, src, PtrArg(Transform), pRes);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            return MulForwardFp8<TStoreFunc>(c, pParams, SrcDim, DstDim, src, PtrArg(Transform), pRes);
        } else {
            return MulForwardInt8<TStoreFunc>(c, pParams, SrcDim, DstDim, src, PtrArg(Transform), pRes);
        }
    }

    template <class TSrc, class TSrcGrad>
    void BackpropFp16(TPtrArg<TGraph> c, TComputeParams *pParams, TSrc &src, TCuda2DArray<half> &grad, TCudaPOD<float> gradScale,
        TCuda2DArray<TSrcGrad> *pSrcGrad, EResultStore rs, bool updateMatrix)
    {
        if (BWD_MATMUL_TYPE != MATMUL_FP16) {
            ConvertMatrix(c, Transform->GetFast(), SrcDim, DstDim, &TransformFp16);
            BackpropMatMulFp16(
                c, pParams, SrcDim, DstDim, src, TransformFp16, Transform->GetScale(), grad, gradScale, rs, updateMatrix, pSrcGrad, &Delta);
        } else {
            BackpropMatMulFp16(c, pParams, SrcDim, DstDim, src, PtrArg(Transform), grad, gradScale, rs, updateMatrix, pSrcGrad, &Delta);
        }
    }

    template <class TGradFloat>
    auto BackpropMulti(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TGradFloat> &grad)
    {
        if (BWD_MATMUL_TYPE == MATMUL_FP16) {
            return MakeMatMulArgs<0, 1>(nullptr, grad, Transform->GetFast(), (int)DstDim);
        } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            return Fp8bwd.BackpropMulti(c, SrcDim, DstDim, PtrArg(Transform), grad);
        } else {
            return Int8bwd.BackpropMulti(c, pParams, SrcDim, DstDim, PtrArg(Transform), grad);
        }
    }

    template <class TSrc, class TGradFloat>
    void BackpropDelta(TPtrArg<TGraph> c, TComputeParams *pParams, TSrc &src, TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale,
        bool updateMatrix)
    {
        if (BWD_MATMUL_TYPE == MATMUL_FP16) {
            BackpropDeltaFp16(c, pParams, SrcDim, DstDim, src, grad, gradScale, updateMatrix, &Delta);
        } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            Fp8bwd.BackpropDelta(c, pParams, SrcDim, DstDim, src, grad, gradScale, updateMatrix, &Delta);
        } else {
            Int8bwd.BackpropDelta(c, pParams, SrcDim, DstDim, src, grad, gradScale, updateMatrix, &Delta);
        }
    }

    void AddDelta(TPtrArg<TGraph> c, EBackpropMode bm) { Transform->AddDelta(c, Delta, bm); }
    TCudaPOD<float> GetScale() { return Transform->GetScale(); }
};


template <class TMultiArgs, class TSrcGrad>
inline void Backprop(TPtrArg<TGraph> c, TComputeParams *pParams, const TMultiArgs &multiArgs, TCuda2DArray<TSrcGrad> *pSrcGrad)
{
    int srcDim = pSrcGrad->GetXSize();
    float mulForwardScale = CalcDotScale(srcDim) * MODEL_DISCR_SCALE;
    if (BWD_MATMUL_TYPE == MATMUL_FP16) {
        Fp16MatMulMulti<TStoreScaled>(c, pSrcGrad, pParams->Len, srcDim, multiArgs).Struct().Read(nullptr, mulForwardScale);
    } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
        Fp8MatMulMulti<TStoreScaled>(c, pSrcGrad, pParams->Len, srcDim, multiArgs).Struct().Read(nullptr, mulForwardScale);
    } else {
        Int8BackpropMultiRun(c, pParams, multiArgs, pSrcGrad);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaLayerBase : public TThrRefBase
{
protected:
    TIntrusivePtr<TMultiDeviceBuffers> MultiBuffers;
    yint DeviceId = 0;
    TModelDims Dims;
    bool UpdateLayers = false;
    TVector<TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>>> MatrArr;

protected:
    template <class T>
    TCuda2DArrayFragment<T> XSplitWindow(TCuda2DArray<T> &data)
    {
        return CalcXSplitWindow(data, data.GetXSize(), DeviceId, MultiBuffers->GetDeviceGroup());
    }

    template <class T>
    TCuda2DArrayFragment<T> XSplitWindow(TCuda2DArray<T> &data, yint xSize)
    {
        return CalcXSplitWindow(data, xSize, DeviceId, MultiBuffers->GetDeviceGroup());
    }

    template <class T>
    TCuda2DArrayFragment<T> YSplitWindow(TCuda2DArray<T> &data)
    {
        return CalcYSplitWindow(data, data.GetYSize(), DeviceId, MultiBuffers->GetDeviceGroup());
    }

    TCudaLayerBase(TPtrArg<TMultiDeviceBuffers> multiBuffers, yint deviceId, const TModelDims &dims, bool updateLayers,
        const TVector<TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>>> &matrArr)
        : MultiBuffers(multiBuffers), DeviceId(deviceId), Dims(dims), UpdateLayers(updateLayers),
          MatrArr(matrArr)
    {
    }

public:
    void CopyToDevice(TPtrArg<NCuda::TGraph> c);
    virtual void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState,
        TCuda2DArray<TStateFloat> *pState, TFragmentStates *pNextNormState) = 0;
    virtual TCudaPOD<float> AddBackward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState,
        TCuda2DArray<TFastGradientFloat> *pStateGrad, TCudaPOD<float> gradScale, TPtrArg<TMultiDeviceFRed2DArray<float>> dNormStatePtr) = 0;
    virtual void AddDelta(TPtrArg<TGraph> c, EBackpropMode bm) = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TInferMatMul
{
    TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>> Transform;
    yint SrcDim = 0;
    yint DstDim = 0;

    void Init(TPtrArg<TCudaInferModelMatrix<TFastModelFloat>> transform)
    {
        Transform = transform.Get();
        SrcDim = Transform->GetXSize();
        DstDim = Transform->GetYSize();
    }

    template <class TStoreFunc, class TSrc, class TDst>
    TKernelOp &Forward(TPtrArg<TGraph> c, TComputeParams *pParams, TSrc &src, TCuda2DArray<TDst> *pRes)
    {
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            return MulForwardFp16<TStoreFunc>(c, pParams, SrcDim, DstDim, src, PtrArg(Transform), pRes);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            return MulForwardFp8<TStoreFunc>(c, pParams, SrcDim, DstDim, src, PtrArg(Transform), pRes);
        } else {
            return MulForwardInt8<TStoreFunc>(c, pParams, SrcDim, DstDim, src, PtrArg(Transform), pRes);
        }
    }

    TCudaPOD<float> GetScale()
    {
        return Transform->GetScale();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaInferLayerBase : public TThrRefBase
{
protected:
    TModelDims Dims;

protected:
    TCudaInferLayerBase(const TModelDims &dims) : Dims(dims) {}

public:
    virtual void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TStateFloat> *pState,
        TCuda2DArray<TNormStateFloat> *pNormState, TCuda2DArray<float> *pNormStateScale) = 0;
};
}
