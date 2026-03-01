#pragma once
#include <gpt/model_params/model_dim.h>
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/vec_util.cuh>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float ComputeLoLU(float x)
{
    float arg = x * LOLU_SCALE;
    if (arg < -20) {
        return 0;
    } else if (arg > 20) {
        return 1;
    } else {
        float w = expf(arg);
        return w / (1 + w);
    }
}

inline __device__ float4 ComputeLoLU(float4 v)
{
    return make_float4(ComputeLoLU(v.x), ComputeLoLU(v.y), ComputeLoLU(v.z), ComputeLoLU(v.w));
}


inline __device__ float ComputeLoLUGrad(float mult)
{
    return mult * (1 - mult) * LOLU_SCALE;
}

inline __device__ float4 ComputeLoLUGrad(float4 mult)
{
    return make_float4(ComputeLoLUGrad(mult.x), ComputeLoLUGrad(mult.y), ComputeLoLUGrad(mult.z), ComputeLoLUGrad(mult.w));
}


template <class TDst>
__global__ void MatrixLoLU(int len, TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valsArr, TCuda2DPtr<TDst> resArr)
{
    int xTile = blockIdx.x;
    int offset = xTile * MM_TILE;
    int tTile = blockIdx.y;
    int warpId = threadIdx.y;
    constexpr int SZ = MM_TILE / MAX_WARPS;

    float4 gate[SZ];
    float4 vals[SZ];
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        if (t < len) {
            gate[k] = LoadWarpVec(gateArr[t] + offset);
            vals[k] = LoadWarpVec(valsArr[t] + offset);
        }
    }
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        if (t < len) {
            float4 vec = vals[k] * ComputeLoLU(gate[k]);
            StoreWarpVec(resArr[t] + offset, vec);
        } else {
            StoreZeroWarpVec(resArr[t] + offset);
        }
    }
}
KERNEL_BLOCK_SIZE(MatrixLoLU, WARP_SIZE, MAX_WARPS);


inline __device__ void BackpropLoLUimpl(float4 gate, float4 grad, float4 vals, float4 *resGradGate, float4 *resGradVals)
{
    float4 mult = ComputeLoLU(gate);
    *resGradGate = grad * vals * ComputeLoLUGrad(mult);
    *resGradVals = grad * mult;
}


template <class TGrad>
__global__ void BackpropLoLU(int len, TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valsArr, float valsScale, TCuda2DPtr<TGrad> gradArr, TCuda2DPtr<TGrad> gradGate, TCuda2DPtr<TGrad> gradVals)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    if (t < len) {
        float4 gate = LoadWarpVec(gateArr[t] + offset);
        float4 vals = LoadWarpVec(valsArr[t] + offset);
        vals = Scale(vals, valsScale);
        // load rg grad
        float4 grad = LoadWarpVec(gradArr[t] + offset);

        // backprop alu
        float4 resGradGate;
        float4 resGradVals;
        BackpropLoLUimpl(gate, grad, vals, &resGradGate, &resGradVals);
        StoreWarpVec(gradGate[t] + offset, resGradGate);
        StoreWarpVec(gradVals[t] + offset, resGradVals);
    } else {
        StoreZeroWarpVec(gradGate[t] + offset);
        StoreZeroWarpVec(gradVals[t] + offset);
    }
}


template <class TGrad, class TResGrad>
__global__ void BackpropRowTileNormalizeLoLU(int len, TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valsArr, float valsScale,
    TCuda2DPtr<float> valsSrcTileScale, TCuda2DPtr<TGrad> gradArr, TCuda2DPtr<TResGrad> gradGate, TCuda2DPtr<TResGrad> gradVals)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    if (t < len) {
        float4 gate = LoadWarpVec(gateArr[t] + offset);
        float4 vals = LoadWarpVec(valsArr[t] + offset);
        vals = Scale(vals, valsScale);
        // load rg grad
        float4 grad = LoadWarpVec(gradArr[t] + offset);

        // backprop alu
        float4 resGradGate;
        float4 resGradVals;
        BackpropLoLUimpl(gate, grad, vals, &resGradGate, &resGradVals);
        StoreWarpVec(gradGate[t] + offset, resGradGate);

        // backprop row tile normalize
        vals = Scale(vals, valsSrcTileScale[tile][t] / valsScale);
        float4 valsNormGrad;
        valsNormGrad = TileNormalizeBackpropWarpVec(vals, resGradVals);
        StoreWarpVec(gradVals[t] + offset, valsNormGrad);
    } else {
        StoreZeroWarpVec(gradGate[t] + offset);
        StoreZeroWarpVec(gradVals[t] + offset);
    }
}


// fused backpropLoLU + CalcDScale
template <class TVLGGrad, class TGateGrad, class TValGrad>
__global__ void BackpropLoLUdScale(int len, TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valsArr, float valsScale,
    TCuda2DPtr<TVLGGrad> gradArr, TCuda2DPtr<TGateGrad> gradGate, TCuda2DPtr<TValGrad> gradVals, TCuda2DPtr<float> dScaleArr)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    float dScale = 0;
    if (t < len) {
        float4 gate = LoadWarpVec(gateArr[t] + offset);
        float4 vals = LoadWarpVec(valsArr[t] + offset);
        vals = Scale(vals, valsScale);
        // load rg grad
        float4 grad = LoadWarpVec(gradArr[t] + offset);

        // backprop alu
        float4 resGradGate;
        float4 resGradVals;
        BackpropLoLUimpl(gate, grad, vals, &resGradGate, &resGradVals);
        StoreWarpVec(gradGate[t] + offset, resGradGate);
        StoreWarpVec(gradVals[t] + offset, resGradVals);

        // compute dScale
        dScale = DotProductWarpVec(vals, resGradVals);
    } else {
        StoreZeroWarpVec(gradGate[t] + offset);
        StoreZeroWarpVec(gradVals[t] + offset);
    }
    if (threadIdx.x == 0) {
        dScaleArr[tile][t] = dScale;
    }
}
}
