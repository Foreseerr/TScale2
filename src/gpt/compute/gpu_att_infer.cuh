#pragma once
#include <gpt/model_params/model_dim.h>
#include "gpu_rope.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int NO_ROPE = 0;
constexpr int USE_ROPE = 1;


template <int Q_DIM, int TT_DIM, int USE_ROPE_VAL>
__global__ void InferAtt(
    TCuda2DPtr<TAttVecFloat> q8,
    TCuda2DPtr<TAttVecFloat> keyCache, float *kGlobalScale, TCuda2DPtr<float> keyCacheScale,
    TCuda2DPtr<TAttVecFloat> valueCache,
    float alibiSlope, int maxWidth,
    TCuda1DPtr<int> readArr, TCuda1DPtr<int> readArrPtr, TCuda2DPtr<TRopeFloat> ropeBuf,
    TCuda2DPtr<half> valLookup
)
{
    CUDA_STATIC_ASSERT(TT_DIM == WARP_VEC_DIM);
    CUDA_STATIC_ASSERT(Q_DIM == WARP_VEC_DIM);

    int batchId = blockIdx.x;
    int head = blockIdx.y;

    float attDotScale = CalcDotScale(Q_DIM) * CalcAttentionMult() * *kGlobalScale * Q_VEC_SCALE;

    // load query
    float4 query = LoadWarpVec(q8[batchId] + head * Q_DIM);

    // apply rope if needed
    if (USE_ROPE_VAL == USE_ROPE) {
        float4 rope = LoadWarpVec(ropeBuf[batchId]);
        ApplyWarpRopeImpl(rope, 1, &query);
    }

    // result accum
    float4 res = ZeroWarpVec();
    float maxDP = 0;
    float sumWeight = 1;

    // collect
    int readBeg = readArrPtr[batchId];
    int readFin = readArrPtr[batchId + 1];
    readBeg = max(readBeg, readFin - maxWidth);
    for (int readPtr = readBeg; readPtr < readFin; ++readPtr) {
        int dist = readFin - readPtr;
        int cacheSampleId = readArr[readPtr];

        float4 key = LoadWarpVec(keyCache[cacheSampleId] + head * Q_DIM);

        float4 value = LoadWarpVec(valueCache[cacheSampleId] + head * TT_DIM);

        // compute qk dot product
        float dp = DotProductWarpVec(query, key);
        dp *= attDotScale * keyCacheScale[head][cacheSampleId];
        dp += GetAttentionDecay(dist, alibiSlope);

        // maxDP corrected weight
        float dpScale = 1;
        if (dp > maxDP) {
            dpScale = exp2(maxDP - dp);
            maxDP = dp;
        }
        float w = exp2f(dp - maxDP);

        // accumulate
        res = Scale(res, dpScale) + Scale(value, w);
        sumWeight = sumWeight * dpScale + w;
    }

    // save result
    res = Scale(res, 1 / sumWeight);
    StoreWarpVec(valLookup[batchId] + head * TT_DIM, res);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Q_DIM, int TT_DIM>
__global__ void CopyToKVcache(
    int headCount,
    TCuda2DPtr<TAttVecFloat> k8, TCuda2DPtr<float> kScale, TCuda2DPtr<TAttVecFloat> v8,
    TCuda1DPtr<int> writeArr, TCuda2DPtr<TRopeFloat> ropeBuf,
    TCuda2DPtr<TAttVecFloat> keyCache, TCuda2DPtr<float> keyCacheScale, TCuda2DPtr<TAttVecFloat> valueCache
)
{
    CUDA_STATIC_ASSERT(TT_DIM == WARP_VEC_DIM);
    CUDA_STATIC_ASSERT(Q_DIM == WARP_VEC_DIM);

    int h = threadIdx.x;
    int batchId = blockIdx.x;
    int writePtr = writeArr[batchId];

    // load rope
    float4 rope = LoadWarpVec(ropeBuf[batchId]);

    // copy all heads
    for (int head = 0; head < headCount; ++head) {
        float4 key = LoadWarpVec(k8[batchId] + Q_DIM * head);
        ApplyWarpRopeImpl(rope, 1, &key);
        StoreWarpVec(keyCache[writePtr] + Q_DIM * head, key);

        float4 value = LoadWarpVec(v8[batchId] + TT_DIM * head);
        StoreWarpVec(valueCache[writePtr] + TT_DIM * head, value);
    }

    // plenty of cache misses
    for (int base = 0; base < headCount; base += WARP_SIZE) {
        int head = base + h;
        if (head < headCount) {
            keyCacheScale[head][writePtr] = kScale[head][batchId];
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Q_DIM, int TT_DIM>
__global__ void CopyToKVcache3Lin(int headCount, TCuda2DPtr<TAttVecFloat> k8, TCuda2DPtr<TAttVecFloat> v8,
    TCuda1DPtr<int> writeArr, TCuda2DPtr<TAttVecFloat> keyCache,
    TCuda2DPtr<TAttVecFloat> valueCache)
{
    CUDA_STATIC_ASSERT(TT_DIM == WARP_VEC_DIM);
    CUDA_STATIC_ASSERT(Q_DIM == WARP_VEC_DIM);

    int batchId = blockIdx.x;
    int writePtr = writeArr[batchId];

    // copy all heads
    for (int head = 0; head < headCount; ++head) {
        float4 key = LoadWarpVec(k8[batchId] + Q_DIM * head);
        StoreWarpVec(keyCache[writePtr] + Q_DIM * head, key);

        float4 value = LoadWarpVec(v8[batchId] + TT_DIM * head);
        StoreWarpVec(valueCache[writePtr] + TT_DIM * head, value);
    }
}
}
