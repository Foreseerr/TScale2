#pragma once
#include "cfg_precision.h"
#include "row_scale.cuh"
#include "row_tile_scale.cuh"
#include <gpt/matrix/base_cuda.cuh>
#include <gpt/matrix/delta_cuda.h>
#include <gpt/model_params/model_dim.h>
#include <lib/cuda/cuda_fp16.cuh>
#include <lib/cuda/cuda_fp8.cuh>
#include <lib/cuda/cuda_i8.cuh>
#include <lib/cuda/cuda_sort.cuh>
#include <lib/cuda/vec_util.cuh>


namespace NCuda
{

constexpr bool DETERMINISTIC = true;
//constexpr bool DETERMINISTIC = false;


///////////////////////////////////////////////////////////////////////////////////////////////////
// forward matmul
template <class TStoreFunc, class TSrc, class TTransofrm, class TDst, class TParamsType>
TKernelOp &MulForwardInt8(
    TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int dstDim, TSrc &src, TPtrArg<TTransofrm> pTransform, TDst *pRes)
{
    return Int8MatMul<TStoreFunc>(c, src, pTransform->GetFast(), pRes, pParams->Len, srcDim, dstDim).Struct();
}


template <class TStoreFunc, class TSrc, class TTransofrm, class TDst, class TParamsType>
TKernelOp &MulForwardFp8(
    TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int dstDim, TSrc &src, TPtrArg<TTransofrm> pTransform, TDst *pRes)
{
    return Fp8MatMul<TStoreFunc>(c, src, pTransform->GetFast(), pRes, pParams->Len, dstDim, srcDim).Struct();
}


template <class TStoreFunc, class TSrc, class TTransofrm, class TDst, class TParamsType>
TKernelOp &MulForwardFp16(
    TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int dstDim, TSrc &src, TPtrArg<TTransofrm> pTransform, TDst *pRes)
{
    return Fp16MatMul<0, 0, TStoreFunc>(c, src, pTransform->GetFast(), pRes, pParams->Len, dstDim, srcDim).Struct();
}


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EResultStore {
    RESULT_GRAD_MOV,
    RESULT_GRAD_ADD,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// fp16 backprop matmul
template <class TSrc, class TTransformFloat, class TGradFloat, class TSrcGradType, class TParamsType>
void BackpropMatMulFp16(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TCuda2DArray<TSrc> &src,
    TCuda2DArray<TTransformFloat> &transformMatr, TCudaPOD<float> trScale, TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale,
    EResultStore rs, bool updateMatrix, TSrcGradType *pSrcGrad, TCudaPackedDeltaMatrix *pPackedDeltaMatrix)
{
    float normScale = CalcDotScale(srcDim) * MODEL_DISCR_SCALE;
    if (rs == RESULT_GRAD_MOV) {
        Fp16MatMul<0, 1, TStoreScaled>(c, grad, transformMatr, pSrcGrad, pParams->Len, srcDim, gradDim).Struct().Read(nullptr, normScale);
    } else if (rs == RESULT_GRAD_ADD) {
        Fp16MatMul<0, 1, TStoreAddScaled>(c, grad, transformMatr, pSrcGrad, pParams->Len, srcDim, gradDim)
            .Struct()
            .Read(nullptr, normScale);
    } else {
        Y_VERIFY(0);
    }

    if (updateMatrix) {
        // former sum rank one
        Fp16MatMul<1, 1, TStoreRowTileMaxNormalize>(c, grad, src, &pPackedDeltaMatrix->Delta, gradDim, srcDim, pParams->Len)
            .Struct()
            .Read(gradScale)
            .Write(&pPackedDeltaMatrix->TileScale);
    }
}


template <class TSrc, class TTransofrmFloat, class TGradFloat, class TSrcGradType, class TParamsType>
void BackpropMatMulFp16(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TCuda2DArray<TSrc> &src,
    TPtrArg<ICudaModelMatrixBase<TTransofrmFloat>> pTransform, TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale, EResultStore rs,
    bool updateMatrix, TSrcGradType *pSrcGrad, TCudaPackedDeltaMatrix *pPackedDeltaMatrix)
{
    BackpropMatMulFp16(c, pParams, srcDim, gradDim, src, pTransform->GetFast(), pTransform->GetScale(), grad, gradScale, rs, updateMatrix,
        pSrcGrad, pPackedDeltaMatrix);
}


template <class TSrc, class TGradFloat, class TParamsType>
void BackpropDeltaFp16(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TCuda2DArray<TSrc> &src,
    TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale, bool updateMatrix, TCudaPackedDeltaMatrix *pPackedDeltaMatrix)
{
    if (updateMatrix) {
        // former sum rank one
        Fp16MatMul<1, 1, TStoreRowTileMaxNormalize>(c, grad, src, &pPackedDeltaMatrix->Delta, gradDim, srcDim, pParams->Len)
            .Struct()
            .Read(gradScale)
            .Write(&pPackedDeltaMatrix->TileScale);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFp8MatMulBwdBuffers
{
    typedef e4m3 TMyGradFloat; // sufficient and precise enough

    TCuda2DArray<TMyGradFloat> GradE5;
    TCuda2DArray<TMyGradFloat> GradE5T;
    TCuda2DArray<e4m3> SrcE4T;
    TCuda2DArray<e4m3> TransformT;

    void AllocateCuda(int srcDim, int gradDim, int maxLen, TPtrArg<TCudaMemoryPool> pool)
    {
        int srcDimRound = RoundUp(srcDim, MM_TILE);
        int gradDimRound = RoundUp(gradDim, MM_TILE);
        GradE5.AllocateCuda(gradDimRound, maxLen, pool); // unused if gradient is passed in correct type
        GradE5T.AllocateCuda(maxLen, gradDimRound, pool);
        SrcE4T.AllocateCuda(maxLen, srcDimRound, pool);
        TransformT.AllocateCuda(gradDimRound, srcDimRound, pool);
    }

    template <class TModelFloat, class TGradFloat>
    auto BackpropMulti(
        TPtrArg<TGraph> c, int srcDim, int gradDim, TPtrArg<ICudaModelMatrixBase<TModelFloat>> pTransform, TCuda2DArray<TGradFloat> &grad)
    {
        Y_VERIFY(0 && "not supported arg types");
        return MakeMatMulArgs<0, 1>(this, grad, pTransform->GetFast(), gradDim);
    }

    auto BackpropMulti(
        TPtrArg<TGraph> c, int srcDim, int gradDim, TPtrArg<ICudaModelMatrixBase<e4m3>> pTransform, TCuda2DArray<TMyGradFloat> &grad)
    {
        Transpose(c, pTransform->GetFast(), srcDim, gradDim, &TransformT);
        return MakeMatMulArgs<0, 1>(this, grad, TransformT, gradDim);
    }

    template <class TParamsType, class TSrcFloat, class TGradFloat>
    void BackpropDelta(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TCuda2DArray<TSrcFloat> &src,
        TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale, bool updateMatrix, TCudaPackedDeltaMatrix *pPackedDeltaMatrix)
    {
        Y_VERIFY(0 && "not supported arg types");
    }

    template <class TParamsType>
    void BackpropDelta(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TCuda2DArray<e4m3> &src,
        TCuda2DArray<TMyGradFloat> &grad, TCudaPOD<float> gradScale, bool updateMatrix, TCudaPackedDeltaMatrix *pPackedDeltaMatrix)
    {
        if (updateMatrix) {
            // former sum rank one
            Transpose(c, src, srcDim, pParams->Len, &SrcE4T);
            Transpose(c, grad, gradDim, pParams->Len, &GradE5T);
            Fp8MatMul<TStoreRowTileMaxNormalize>(c, GradE5T, SrcE4T, &pPackedDeltaMatrix->Delta, gradDim, srcDim, pParams->Len)
                .Struct()
                .Read(gradScale)
                .Write(&pPackedDeltaMatrix->TileScale);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// int8 backprop matmul

static __global__ void ShuffleTransposeKernel(TCuda2DPtr<i8> src, TCuda1DPtr<TSortNode> sortNode, int rowCount, TCuda2DPtr<i8> dst)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    int h = threadIdx.x;
    int xBlock = blockIdx.x * MM_TILE;
    int yBlock = blockIdx.y * MM_TILE;

    __shared__ i8 buf[128][128];
    constexpr int yStep = WARP_SIZE / 8;
    int yOffset = h / 8;
    int xOffset = 16 * (h & 7);
    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        int t = yBlock + y;
        int4 *pDst = (int4 *)&buf[y][xOffset];
        if (t < rowCount) {
            int nodeId = sortNode[t].NodeId;
            *pDst = *(int4 *)&src[nodeId][xBlock + xOffset];
        } else {
            *pDst = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        union {
            int4 column;
            i8 columnBytes[16];
        };
        for (int k = 0; k < 16; ++k) {
            int x = xOffset + k;
            columnBytes[k] = buf[x][y];
        }
        int4 *pDst = (int4 *)&dst[xBlock + y][yBlock + xOffset];
        *pDst = column;
    }
}


template <class TSrcFloat>
__global__ void ShuffleMaxScaleTransposeKernel(TCuda2DPtr<TSrcFloat> src, TCuda1DPtr<TSortNode> sortNode, int rowCount, TCuda2DPtr<i8> dst, TCuda1DPtr<float> tileScale)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    int h = threadIdx.x;
    int xBlock = blockIdx.x * MM_TILE;
    int yBlock = blockIdx.y * MM_TILE;

    float myMaxScale = 0;
    for (int k = h; k < 128; k += WARP_SIZE) {
        int t = yBlock + k;
        if (t < rowCount) {
            myMaxScale = fmaxf(myMaxScale, sortNode[t].Score);
        }
    }
    float maxScale = WarpMax(myMaxScale);
    if (maxScale == 0) {
        maxScale = 1;
    }
    maxScale = RoundFloatUp(maxScale);

    TCudaRngLCG rng(*(ui32 *)&myMaxScale, xBlock + *(ui32 *)&maxScale, yBlock + h);
    rng.Gen();

    __shared__ i8 buf[128][128];
    constexpr int yStep = WARP_SIZE / 8;
    int yOffset = h / 8;
    int xOffset = 16 * (h & 7);
    float mult = 1 / maxScale;
    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        int t = yBlock + y;
        int4 *pDst = (int4 *)&buf[y][xOffset];
        if (t < rowCount) {
            int nodeId = sortNode[t].NodeId;
            //float mult = sortNode[t].Score / maxScale;
            union {
                int4 row;
                i8 rowBytes[16];
            };
            //row = *(int4 *)&src[nodeId][xBlock + xOffset];
            //for (int k = 0; k < 16; ++k) {
            //    rowBytes[k] = CvtToI8(rowBytes[k] * mult);
            //}
            for (int k = 0; k < 16; ++k) {
                float rshift = rng.GenUniformFloat() - 0.5f;
                rowBytes[k] = CvtToI8(float(src[nodeId][xBlock + xOffset + k]) * mult + rshift); // slow, uncoalesced loads, use half instead of float source
            }
            *pDst = row;
        } else {
            *pDst = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        union {
            int4 column;
            i8 columnBytes[16];
        };
        for (int k = 0; k < 16; ++k) {
            int x = xOffset + k;
            columnBytes[k] = buf[x][y];
        }
        int4 *pDst = (int4 *)&dst[xBlock + y][yBlock + xOffset];
        *pDst = column;
    }
    if (h == 0 && xBlock == 0) {
        tileScale[blockIdx.y] = maxScale;
    }
}


struct TInt8MatMulBwdBuffers
{
    TCuda2DArray<i8> Grad8;
    TCudaVector<float> Grad8RowScale;
    TCuda2DArray<i8> Grad8T;
    TCuda2DArray<i8> Src8T;
    TCudaVector<TSortNode> SortedSamples;
    TCudaVector<float> SortedTileScale;
    TCuda2DArray<i8> TransformT;

    void AllocateCuda(int srcDim, int gradDim, int maxLen, TPtrArg<TCudaMemoryPool> pool)
    {
        int srcDimRound = RoundUp(srcDim, MM_TILE);
        int gradDimRound = RoundUp(gradDim, MM_TILE);
        Grad8.AllocateCuda(gradDimRound, maxLen, pool);
        Grad8RowScale.AllocateCuda(maxLen, pool);
        Grad8T.AllocateCuda(maxLen, gradDimRound, pool);
        Src8T.AllocateCuda(maxLen, srcDimRound, pool);
        SortedSamples.AllocateCuda(maxLen, pool);
        SortedTileScale.AllocateCuda(maxLen / MM_TILE, pool);
        TransformT.AllocateCuda(gradDimRound, srcDimRound, pool);
    }

    template <class TParamsType, class TModelFloat, class TGradFloat>
    auto BackpropMulti(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim,
        TPtrArg<ICudaModelMatrixBase<TModelFloat>> pTransform, TCuda2DArray<TGradFloat> &grad)
    {
        Y_VERIFY(0 && "not supported arg types");
        return MakeMatMulArgs<0, 1>(this, grad, pTransform->GetFast(), gradDim);
    }

    template <class TParamsType, class TGradFloat>
    auto BackpropMulti(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TPtrArg<ICudaModelMatrixBase<i8>> pTransform,
        TCuda2DArray<TGradFloat> &grad)
    {
        // int tailTiles = MODEL_MATMUL_EXACT_BUF / MM_TILE;
        float mulForwardScale = CalcDotScale(srcDim) * MODEL_DISCR_SCALE;
        TCudaPOD<float> trScale = pTransform->GetScale();

        NormalizeVecsRowMaxWithNoise(c, gradDim, pParams->Len, pParams->LenRound, grad, &Grad8, &Grad8RowScale);
        if (DETERMINISTIC) {
            // SortFloats(c, Grad8RowScale, pParams->Len, &SortedSamples);
            SortPositiveFloatsApproxStable(c, Grad8RowScale, pParams->Len, &SortedSamples);
        } else {
            SortPositiveFloatsApproxFast(c, Grad8RowScale, pParams->Len, &SortedSamples);
        }

        // mul backward
        Transpose(c, pTransform->GetFast(), srcDim, gradDim, &TransformT);
        return MakeMatMulArgs<0, 1>(this, grad, TransformT, gradDim);
    }

    template <class TParamsType, class TSrcFloat, class TGradFloat>
    void BackpropDelta(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TCuda2DArray<TSrcFloat> &src,
        TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale, bool updateMatrix, TCudaPackedDeltaMatrix *pPackedDeltaMatrix)
    {
        Y_VERIFY(0 && "not supported arg types");
    }

    template <class TParamsType, class TGradFloat>
    void BackpropDelta(TPtrArg<TGraph> c, TParamsType *pParams, int srcDim, int gradDim, TCuda2DArray<i8> &src,
        TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale, bool updateMatrix, TCudaPackedDeltaMatrix *pPackedDeltaMatrix)
    {
        if (updateMatrix) {
            // delta matrix
            // use high precision source to avoid precision loss in int8 reconversion to same scaling factor per sample
            // add noise to increase stability
            CudaCall(c, ShuffleMaxScaleTransposeKernel<TGradFloat>)
                .Grid(MMTiles(gradDim), MMTiles(pParams->Len))
                .Read(grad, SortedSamples, pParams->Len)
                .Write(&Grad8T, &SortedTileScale);
            CudaCall(c, ShuffleTransposeKernel)
                .Grid(MMTiles(srcDim), MMTiles(pParams->Len))
                .Read(src, SortedSamples, pParams->Len)
                .Write(&Src8T);
            Int8MatMulYScale<TStoreRowTileMaxNormalize>(
                c, Grad8T, Src8T, SortedTileScale, &pPackedDeltaMatrix->Delta, gradDim, pParams->Len, srcDim)
                .Struct()
                .Read(gradScale)
                .Write(&pPackedDeltaMatrix->TileScale);
        }
    }
};


template <class TSrcGradType, class TParamsType>
inline bool Int8BackpropMultiRun(TPtrArg<TGraph> c, TParamsType *pParams, const TMatMulArgsRoot &multiArgs, TSrcGradType *pSrcGrad)
{
    return false;
}

template <class TSrcGradType, class TParamsType, class TMultiArgs>
inline bool Int8BackpropMultiRun(TPtrArg<TGraph> c, TParamsType *pParams, const TMultiArgs &multiArgs, TSrcGradType *pSrcGrad)
{
    bool targetBufNonZero = Int8BackpropMultiRun(c, pParams, multiArgs.Prev, pSrcGrad);
    int srcDim = pSrcGrad->GetXSize();
    float mulForwardScale = CalcDotScale(srcDim) * MODEL_DISCR_SCALE;
    TInt8MatMulBwdBuffers *p = (TInt8MatMulBwdBuffers *)multiArgs.Args.Owner;
    TKernelOp *op = 0;
    if (targetBufNonZero) {
        op = &Int8MatMulRowScale<TStoreAddScaled>(
            c, p->Grad8, p->Grad8RowScale, p->TransformT, pSrcGrad, pParams->Len, multiArgs.Args.KSize, srcDim);
    } else {
        op = &Int8MatMulRowScale<TStoreScaled>(
            c, p->Grad8, p->Grad8RowScale, p->TransformT, pSrcGrad, pParams->Len, multiArgs.Args.KSize, srcDim);
    }
    op->Struct().Read(nullptr, mulForwardScale);
    return true;
}
}
