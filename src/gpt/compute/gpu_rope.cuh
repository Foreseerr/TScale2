#pragma once
#include "cfg_precision.h"
#include <lib/cuda/cuda_matmul.cuh>

namespace NCuda
{

inline __device__ void ApplyWarpRopeImpl(float rCos, float rSin, float ropeRotateDir, float &v0, float &v1)
{
    float cosValue = rCos;
    float sinValue = rSin * ropeRotateDir;
    float r0 = v0 * cosValue - v1 * sinValue;
    float r1 = v0 * sinValue + v1 * cosValue;
    v0 = r0;
    v1 = r1;
}

inline __device__ void ApplyWarpRopeImpl(float4 rope, float ropeRotateDir, float4 *vec)
{
    ApplyWarpRopeImpl(rope.x, rope.y, ropeRotateDir, vec->x, vec->y);
    ApplyWarpRopeImpl(rope.z, rope.w, ropeRotateDir, vec->z, vec->w);
}


struct TStoreRowTileNormalizeRope : public TStoreRowBase
{
    struct TParams
    {
        TCuda2DPtr<TRopeFloat> RopeBuf;
        float Scale;
        float VecScale; // for normalization
        TCuda2DPtr<float> ScaleBuf;
    };

    __device__ static float GetScale(TParams &params)
    {
        return params.Scale;
    }

    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 vec = LoadWarpVecSmem(mmRes);
        float sum2 = CalcWarpVecSum2(vec);
        float discrScale = 0;
        if (sum2 > 0) {
            discrScale = sqrt(sum2 / MM_TILE) * params.VecScale;
            vec = Scale(vec, 1 / discrScale);
        }
        // apply rope
        float4 rope = LoadWarpVec(params.RopeBuf[dstY]);
        ApplyWarpRopeImpl(rope, 1, &vec);
        // save
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        if (threadIdx.x == 0) {
            params.ScaleBuf[tile][dstY] = discrScale;
        }
        resBuf.CheckCoords(dstX, dstY);
    }
};


// rope fwd/bwd (for bwd invert rotate dir)
template <class TGrad>
__global__ void ApplyRope(TCuda2DPtr<TRopeFloat> ropeBuf, float ropeRotateDir, TCuda2DPtr<TGrad> grad)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    float4 vec = LoadWarpVec(grad[t] + offset);
    float4 rope = LoadWarpVec(ropeBuf[t]);
    ApplyWarpRopeImpl(rope, ropeRotateDir, &vec);
    StoreWarpVec(grad[t] + offset, vec);
}
}
