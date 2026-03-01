#pragma once
#include <lib/cuda/vec_util.cuh>
#include <lib/cuda/cuda_matmul.cuh>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// convert float vectors into i8 vectors + per MM_TILE scale
template <class TSrc, class TDst>
__global__ void NormalizeVecsRowTile(int len, float vecScale, TCuda2DPtr<TSrc> srcArr, TCuda2DPtr<TDst> resArr, TCuda2DPtr<float> resScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    float4 vec;
    float discrScale = 0;
    if (t < len) {
        vec = LoadWarpVec(srcArr[t] + offset);
        float sum2 = CalcWarpVecSum2(vec);
        if (sum2 > 0) {
            discrScale = sqrt(sum2 / MM_TILE) * vecScale;
            vec = Scale(vec, 1 / discrScale);
        }
    } else {
        vec = ZeroWarpVec();
    }
    StoreWarpVec(resArr[t] + offset, vec);
    if (threadIdx.x == 0) {
        resScale[tile][t] = discrScale;
    }
}


// backprop through per tile normalize
template <class TSrc, class TGrad>
__global__ void BackpropRowTileNormalize(TCuda2DPtr<TSrc> src, TCuda2DPtr<float> srcTileScale, TCuda2DPtr<TGrad> grad)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    // normalize
    float4 v = LoadWarpVec(src[t] + offset);
    v = Scale(v, srcTileScale[tile][t]);
    float4 vGrad = LoadWarpVec(grad[t] + offset);
    float4 stateGrad = TileNormalizeBackpropWarpVec(v, vGrad);
    StoreWarpVec(grad[t] + offset, stateGrad);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TDst>
__global__ void NormalizeVecsRowTileMax(int len, TCuda2DPtr<TSrc> srcArr, TCuda2DPtr<TDst> resArr, TCuda2DPtr<float> resScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    float4 vec;
    float discrScale = 0;
    if (t < len) {
        vec = LoadWarpVec(srcArr[t] + offset);
        float maxVal = CalcWarpVecMaxAbsValue(vec);
        if (maxVal > 0) {
            discrScale = GetMaxDiscrScale(maxVal, (TDst *)0);
            vec = Scale(vec, 1 / discrScale);
        }
    } else {
        vec = ZeroWarpVec();
    }
    StoreWarpVec(resArr[t] + offset, vec);
    if (threadIdx.x == 0) {
        resScale[tile][t] = discrScale;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst>
__global__ void FillArray(float val, TCuda2DPtr<TDst> resScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    resScale[tile][t] = val;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int CONVERT_MATRIX_WARP_COUNT = 32;
template <class TSrc, class TDst>
__global__ void ConvertMatrixScaledKernel(int len, TCuda2DPtr<TSrc> srcArr, float mult, TCuda2DPtr<TDst> resArr)
{
    int tile = blockIdx.x;
    int offset = tile * MM_TILE;

    constexpr int SZ = MM_TILE / CONVERT_MATRIX_WARP_COUNT;
    float4 vecArr[SZ];
    for (int k = 0; k < SZ; ++k) {
        int t = blockIdx.y * MM_TILE + threadIdx.y * SZ + k;
        if (t < len) {
            vecArr[k] = LoadWarpVec(srcArr[t] + offset);
        } else {
            vecArr[k] = ZeroWarpVec();
        }
    }
    for (int k = 0; k < SZ; ++k) {
        int t = blockIdx.y * MM_TILE + threadIdx.y * SZ + k;
        StoreWarpVec(resArr[t] + offset, Scale(vecArr[k], mult));
    }
}
KERNEL_BLOCK_SIZE(ConvertMatrixScaledKernel, WARP_SIZE, CONVERT_MATRIX_WARP_COUNT);


// assume sizes are multiple of MM_TILE
template <class TSrc, class TDst, class TLen>
inline void ConvertMatrix(TPtrArg<TGraph> c, TSrc &src, int xDim, TLen &&yDim, TDst *p)
{
    typedef typename TSrc::TElem TSrcFloat;
    typedef typename TDst::TElem TDstFloat;
    CudaCall(c, ConvertMatrixScaledKernel<TSrcFloat, TDstFloat>).Grid(MMTiles(xDim), MMTiles(yDim)).Read(yDim, src, 1.0f).Write(p);
}
}

