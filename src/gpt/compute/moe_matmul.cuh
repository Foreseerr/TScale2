#pragma once
#include "gpu_lolu.cuh"
#include <gpt/matrix/base_cuda.cuh>
#include <lib/cuda/cuda_fp16.cuh>
#include <lib/cuda/cuda_fp8.cuh>
#include <lib/cuda/sm90_fp8.cuh>
#include "cfg_precision.h"
#include "layer_cuda_base.cuh"
#include "matmul_fwdbwd.cuh"


/////////////////////////////////////////////////////////////////////////////////////
namespace NCuda
{

__global__ void MoeTransposeI8Kernel(TCuda2DPtr<i8> src, TCudaSpan expertSpan, TCuda1DPtr<int> tileExpert, TCuda2DPtr<i8> dst);
KERNEL_BLOCK_SIZE(MoeTransposeI8Kernel, WARP_SIZE, I8_TRANSPOSE_WARPS);

template <class TXSize, class TYSize>
void MoeTranspose(TPtrArg<TGraph> c, const TCuda2DArray<e4m3> &src, const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert,
    TXSize &&xSize, TYSize &&ySize, TCuda2DArray<e4m3> *pDst)
{
    CudaCall(c, MoeTransposeI8Kernel).Grid(MMTiles(xSize), MMTiles(ySize)).Read(src, expertSpan, tileExpert).Write(pDst);
}

template <class TXSize, class TYSize>
void MoeTranspose(TPtrArg<TGraph> c, const TCuda2DArray<e5m2> &src, const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert,
    TXSize &&xSize, TYSize &&ySize, TCuda2DArray<e5m2> *pDst)
{
    CudaCall(c, MoeTransposeI8Kernel).Grid(MMTiles(xSize), MMTiles(ySize)).Read(src, expertSpan, tileExpert).Write(pDst);
}

template <class TXSize, class TYSize, class T1, class T2>
void MoeTranspose(TPtrArg<TGraph> c, const TCuda2DArray<T1> &src, const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert, TXSize &&xSize,
    TYSize &&ySize, TCuda2DArray<T2> *pDst)
{
    Y_VERIFY(0 && "not implemented");
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
// moe enn/ent matmul
template <class T1, class T2, int TRANSPOSE_B>
struct TMoeMatMulArgs
{
    void *Owner = 0;
    T1 &A;
    T2 &B;
    int KSize = 0;
    TCudaSpan ExpertSpan;
    TCudaVector<int> &TileExpert;
    int ExpertSize = 0;

    TMoeMatMulArgs(
        void *owner, T1 &a, T2 &b, int kSize, const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert, int expertSize)
        : Owner(owner), A(a), B(b), KSize(kSize), ExpertSpan(expertSpan), TileExpert(tileExpert), ExpertSize(expertSize)
    {
    }
    template <class TElem>
    bool TypeMatch(TElem *p) const
    {
        return typeid(typename T1::TElem) == typeid(TElem) && typeid(typename T2::TElem) == typeid(TElem);
    }
};


template <int TRANSPOSE_B, class T1, class T2>
inline TMoeMatMulArgs<T1, T2, TRANSPOSE_B> MakeMoeMatMulArgs(
    void *owner, T1 &a, T2 &b, int kSize, const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert, int expertSize)
{
    return TMoeMatMulArgs<T1, T2, TRANSPOSE_B>(owner, a, b, kSize, expertSpan, tileExpert, expertSize);
}


template <class TAdapter, class TPrev, class T1, class T2, int TRANSPOSE_B>
struct TMoeMatMulKernelCall : public TPrev
{
    typename TAdapter::TMatrixArg<T1> A;
    typename TAdapter::TMatrixArg<T2> B;
    int KSize;
    TCudaSpan ExpertSpan;
    TCuda1DPtr<int> TileExpert;
    int ExpertSize;

    template <class TFunc>
    __device__ void EnumArgs(TFunc func, int x, int y) const
    {
        TPrev::EnumArgs(func, x, y);
        int expertId = TileExpert[y / MM_TILE];
        if (expertId < ExpertSpan.Beg || expertId >= ExpertSpan.Fin) {
            return;
        }
        int bOffset = ExpertSize * (expertId - ExpertSpan.Beg);
        if (TRANSPOSE_B) {
            func.ComputeMatMul<0, TRANSPOSE_B>(A, 0, y, B, bOffset, x, KSize);
        } else {
            func.ComputeMatMul<0, TRANSPOSE_B>(A, 0, y, B, 0, x + bOffset, KSize);
        }
    }
};


template <class TPrev, class T1, class T2, int TRANSPOSE_B>
struct TMakeKernelParam<TMatMulArgsList<TPrev, TMoeMatMulArgs<T1, T2, TRANSPOSE_B>>>
{
    template <class TAdapter>
    using Result = TMoeMatMulKernelCall<TAdapter, typename TMakeKernelParam<TPrev>::Result<TAdapter>, T1, T2, TRANSPOSE_B>;
};


template <class TAdapter, class TPrev, class T1, class T2, int TRANSPOSE_B>
inline void PassKernelParams(
    TAdapter &ad, TKernelOp &op, const TMatMulArgsList<TPrev, TMoeMatMulArgs<T1, T2, TRANSPOSE_B>> &argList)
{
    PassKernelParams(ad, op, argList.Prev);
    auto &args = argList.Args;
    ad.PassMatrixA(op, args.A);
    ad.PassMatrixB(op, args.B);
    op.Read(args.KSize, args.ExpertSpan, args.TileExpert, args.ExpertSize);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// moe etet matmul
template <class T1, class T2>
struct TMoeDeltaMatMulArgs
{
    void *Owner = 0;
    T1 &A;
    T2 &B;
    TCudaVector<int> &ExpertOffset;
    TCudaSpan ExpertSpan;

    TMoeDeltaMatMulArgs(void *owner, T1 &a, T2 &b, TCudaVector<int> &expertOffset, const TCudaSpan &expertSpan)
        : Owner(owner), A(a), B(b), ExpertOffset(expertOffset), ExpertSpan(expertSpan)
    {
    }
    template <class TElem>
    bool TypeMatch(TElem *p) const
    {
        return typeid(typename T1::TElem) == typeid(TElem) && typeid(typename T2::TElem) == typeid(TElem);
    }
};


template <class T1, class T2>
inline TMoeDeltaMatMulArgs<T1, T2> MakeMoeDeltaMatMulArgs(
    void *owner, T1 &a, T2 &b, TCudaVector<int> &expertOffset, const TCudaSpan &expertSpan)
{
    return TMoeDeltaMatMulArgs<T1, T2>(owner, a, b, expertOffset, expertSpan);
}


template <class TAdapter, class TPrev, class T1, class T2>
struct TMoeDeltaMatMulKernelCall : public TPrev
{
    typename TAdapter::TMatrixArg<T1> A;
    typename TAdapter::TMatrixArg<T2> B;
    TCuda1DPtr<int> ExpertOffset;
    TCudaSpan ExpertSpan;

    template <class TFunc>
    __device__ void EnumArgs(TFunc func, int x, int y) const
    {
        TPrev::EnumArgs(func, x, y);
        int localExpertId = blockIdx.z;
        int expertId = ExpertSpan.Beg + localExpertId;
        int expertStart = ExpertOffset[expertId];
        int expertFinish = ExpertOffset[expertId + 1];
        int expertLen = (expertFinish - expertStart);
        func.ComputeMatMul<1, 1>(A, expertStart, y, B, expertStart, x, expertLen);
    }
};


template <class TPrev, class T1, class T2>
struct TMakeKernelParam<TMatMulArgsList<TPrev, TMoeDeltaMatMulArgs<T1, T2>>>
{
    template <class TAdapter>
    using Result = TMoeDeltaMatMulKernelCall<TAdapter, typename TMakeKernelParam<TPrev>::Result<TAdapter>, T1, T2>;
};


template <class TAdapter, class TPrev, class T1, class T2>
inline void PassKernelParams(TAdapter &ad, TKernelOp &op, const TMatMulArgsList<TPrev, TMoeDeltaMatMulArgs<T1, T2>> &argList)
{
    PassKernelParams(ad, op, argList.Prev);
    auto &args = argList.Args;
    ad.PassMatrixA(op, args.A);
    ad.PassMatrixB(op, args.B);
    op.Read(args.ExpertOffset, args.ExpertSpan);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TPreTransposedArray
{
    TCuda2DArray<T> Arr;
    TCuda2DArray<T> ArrT;

    void AllocateCuda(yint xSize, yint ySize, TPtrArg<TCudaMemoryPool> pool)
    {
        Arr.AllocateCuda(xSize, ySize, pool);
        if (sizeof(T) == 1) {
            ArrT.AllocateCuda(ySize, xSize, pool);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TMoeMatMulBuffers
{
    TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>> Transform;
    yint SrcDim = 0;
    yint DstDim = 0;
    yint ExpertCount = 0;
    TCudaPackedDeltaMatrix Delta;
    // fp8 buffers
    TCuda2DArray<TFastModelFloat> SrcE4T;
    TCuda2DArray<TFastModelFloat> TransformT;

    void AllocateCuda(TPtrArg<ICudaModelMatrixBase<TFastModelFloat>> transform, yint len, yint expertCount, TPtrArg<TCudaMemoryPool> pool)
    {
        Transform = transform.Get();
        SrcDim = Transform->GetLocalXSize();
        DstDim = Transform->GetLocalYSize() / expertCount;
        ExpertCount = expertCount;
        Y_VERIFY((SrcDim % MM_TILE) == 0);
        Y_VERIFY((DstDim % MM_TILE) == 0);
        Delta.AllocateCuda(SrcDim, DstDim * ExpertCount);
        if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            TransformT.AllocateCuda(DstDim * ExpertCount, SrcDim, pool);
        }
    }

    void AllocateCudaSrcTranspose(yint len, TPtrArg<TCudaMemoryPool> pool)
    {
        if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            SrcE4T.AllocateCuda(len, SrcDim, pool);
        }
    }

    template <class TStoreFunc, class TSrc, class TDst>
    TKernelOp &Forward(
        TPtrArg<TGraph> c, TComputeParams *pParams, TSrc &src, TDst *pRes, const TCudaSpan &expertSpan, TCudaVector<int> &xpTileExpertId)
    {
        int expertSize = DstDim;
        auto mm = MakeMoeMatMulArgs<0>(this, src, Transform->GetFast(), SrcDim, expertSpan, xpTileExpertId, expertSize);
        if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            return Fp8MatMulMulti<TStoreFunc>(c, pRes, pParams->LenMoe, DstDim, MultiArg() + mm).Struct();
        } else {
            return Fp16MatMulMulti<TStoreFunc>(c, pRes, pParams->LenMoe, DstDim, MultiArg() + mm).Struct();
        }
    }

    auto BackpropMulti(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TFastGradientFloat> &grad, const TCudaSpan &expertSpan,
        TCudaVector<int> &xpTileExpertId)
    {
        if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            Transpose(c, Transform->GetFast(), SrcDim, DstDim * ExpertCount, &TransformT);
            return MakeMoeMatMulArgs<1>(this, grad, TransformT, DstDim, expertSpan, xpTileExpertId, DstDim);
        } else {
            return MakeMoeMatMulArgs<1>(this, grad, Transform->GetFast(), DstDim, expertSpan, xpTileExpertId, DstDim);
        }
    }

    void BackpropDeltaImpl(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &src,
        TCuda2DArray<TNormStateFloat> &srcT, TCuda2DArray<TFastGradientFloat> &grad, TCuda2DArray<TFastGradientFloat> &gradT,
        TCudaPOD<float> gradScale, TCudaVector<int> &expertOffset, const TCudaSpan &expertSpan, TCudaVector<int> &xpTileExpertId)
    {
        Y_ASSERT(expertSpan.Fin - expertSpan.Beg == ExpertCount);
        if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            auto etet = MakeMoeDeltaMatMulArgs(this, gradT, srcT, expertOffset, expertSpan);
            Fp8MatMulMulti<TStoreRowTileMaxNormalize>(c, &Delta.Delta, DstDim, SrcDim, MultiArg() + etet)
                .GridZ(ExpertCount)
                .Struct()
                .Read(gradScale)
                .Write(&Delta.TileScale);
        } else {
            auto etet = MakeMoeDeltaMatMulArgs(this, grad, src, expertOffset, expertSpan);
            Fp16MatMulMulti<TStoreRowTileMaxNormalize>(c, &Delta.Delta, DstDim, SrcDim, MultiArg() + etet)
                .GridZ(ExpertCount)
                .Struct()
                .Read(gradScale)
                .Write(&Delta.TileScale);
        }
    }

    void BackpropDelta(TPtrArg<TGraph> c, TComputeParams *pParams, TPreTransposedArray<TNormStateFloat> &src,
        TPreTransposedArray<TFastGradientFloat> &grad, TCudaPOD<float> gradScale, TCudaVector<int> &expertOffset,
        const TCudaSpan &expertSpan, TCudaVector<int> &xpTileExpertId, bool updateMatrix)
    {
        if (updateMatrix) {
            BackpropDeltaImpl(c, pParams, src.Arr, src.ArrT, grad.Arr, grad.ArrT, gradScale, expertOffset, expertSpan, xpTileExpertId);
        }
    }

    void BackpropDelta(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &src,
        TPreTransposedArray<TFastGradientFloat> &grad, TCudaPOD<float> gradScale, TCudaVector<int> &expertOffset,
        const TCudaSpan &expertSpan, TCudaVector<int> &xpTileExpertId, bool updateMatrix)
    {
        if (updateMatrix) {
            if (BWD_MATMUL_TYPE == MATMUL_FP8) {
                MoeTranspose(c, src, expertSpan, xpTileExpertId, SrcDim, pParams->LenMoe, &SrcE4T);
                BackpropDeltaImpl(c, pParams, src, SrcE4T, grad.Arr, grad.ArrT, gradScale, expertOffset, expertSpan, xpTileExpertId);
            } else {
                BackpropDeltaImpl(c, pParams, src, src, grad.Arr, grad.ArrT, gradScale, expertOffset, expertSpan, xpTileExpertId);
            }
        }
    }

    void AddDelta(TPtrArg<TGraph> c, EBackpropMode bm) { Transform->AddDelta(c, Delta, bm); }
    TCudaPOD<float> GetScale() { return Transform->GetScale(); }
};


template <class TMultiArgs, class TSrcGrad>
inline void BackpropMoe(TPtrArg<TGraph> c, TComputeParams *pParams, const TMultiArgs &multiArgs, TCuda2DArray<TSrcGrad> *pSrcGrad)
{
    int srcDim = pSrcGrad->GetXSize();
    float mulForwardScale = CalcDotScale(srcDim) * MODEL_DISCR_SCALE;
    if (BWD_MATMUL_TYPE == MATMUL_FP8) {
        Fp8MatMulMulti<TStoreScaled>(c, pSrcGrad, pParams->LenMoe, srcDim, multiArgs).Struct().Read(nullptr, mulForwardScale);
    } else {
        Fp16MatMulMulti<TStoreScaled>(c, pSrcGrad, pParams->LenMoe, srcDim, multiArgs).Struct().Read(nullptr, mulForwardScale);
    }
}
}
