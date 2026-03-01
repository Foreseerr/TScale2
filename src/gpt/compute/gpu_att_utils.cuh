#pragma once
#include <gpt/att/att.h>
#include <gpt/model_params/model_dim.h>
#include <lib/cuda/cuda_mma.cuh>
#include <lib/cuda/vec_util.cuh>
#include "gpu_rope.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__forceinline __device__ void CopyStripe(T4SMemHalfTile *dst, int warpId, TCuda2DPtr<T> src)
{
    Copy4Tile(&dst[0], warpId, src.Fragment(0, 0));
    Copy4Tile(&dst[1], warpId, src.Fragment(64, 0));
}

template <class T>
__forceinline __device__ void CopyStripeAsync(T4SMemHalfTile *dst, int warpId, TCuda2DPtr<T> src)
{
    Copy4TileAsync(&dst[0], warpId, src.Fragment(0, 0));
    Copy4TileAsync(&dst[1], warpId, src.Fragment(64, 0));
}


__forceinline __device__ void LoadStripe(TRegTile<half> *dst, T4SMemHalfTile *src)
{
    for (int i = 0; i < 4; ++i) {
        dst[i] = LoadTile(src[0], i);
        dst[i + 4] = LoadTile(src[1], i);
    }
}

__forceinline __device__ TRegTile<float> StripeDotProduct(TRegTile<half> *q, T4SMemHalfTile *k)
{
    TRegTile<float> qkTile;
    qkTile.Clear();
    for (int i = 0; i < 4; ++i) {
        MMA(&qkTile, q[i], LoadTile(k[0], i));
        MMA(&qkTile, q[i + 4], LoadTile(k[1], i));
    }
    return qkTile;
}

__forceinline __device__ TRegTile<float> StripeDotProduct(T4SMemHalfTile *q, TRegTile<half> *k)
{
    TRegTile<float> qkTile;
    qkTile.Clear();
    for (int i = 0; i < 4; ++i) {
        MMA(&qkTile, LoadTile(q[0], i), k[i]);
        MMA(&qkTile, LoadTile(q[1], i), k[i + 4]);
    }
    return qkTile;
}


template <int N>
__forceinline __device__ void CopyShortVec(float *dst, float *src)
{
    CUDA_STATIC_ASSERT(N <= WARP_SIZE);
    int h = threadIdx.x;
    if (h < N) {
        dst[h] = src[h];
    }
}

template <int N>
__forceinline __device__ void CopyShortVecAsync(float *dst, float *src)
{
    CUDA_STATIC_ASSERT(N <= WARP_SIZE);
    CUDA_STATIC_ASSERT((N & 3) == 0);
    int h = threadIdx.x;
    if (h * 4 < N) {
        AsyncCopy16(dst + h * 4, src + h * 4);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TQKComputer
{
    float ElemAlibi[TTileCoord::num_elements];
    TRegTileRow<int> RowSpanStart;
    TRegTileRow<int> RowSpanFinish;
    int AlibiFromMinusTo;
    float AttDotScale;
    float AlibiSlope;

    struct TQKctx
    {
        float TileAlibi;
        TRegTileColumn<float> qkScaleColumn;
    };

public:
    __device__ TQKComputer(const TTileCoord &tc, float alibiSlope, float attDotScale)
    {
        // precompute alibi
        tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) { //
            ElemAlibi[elem] = GetAttentionDecay(y - x, alibiSlope);
        });
        AttDotScale = attDotScale;
        AlibiSlope = alibiSlope;
    }

    __device__ void SubtractSumWeightLog(const TTileCoord &tc, TRegTileRow<float> sumWeightLogRow)
    {
        tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) { //
            ElemAlibi[elem] -= sumWeightLogRow.x[rowIndex];
        });
    }

    __device__ void StartSpan(const TTileCoord &tc, const TAttentionSpanGroup<ATT_GROUP> &gg, int fromBase, int fromWarp)
    {
        RowSpanStart.Load(tc, gg.SpanStart + fromWarp);
        RowSpanFinish.Load(tc, gg.SpanFinish + fromWarp);
        RowSpanStart.Add(-gg.Start);
        RowSpanFinish.Add(-gg.Start);
        AlibiFromMinusTo = fromBase + fromWarp - gg.Start;
    }

    __forceinline __device__ TQKctx MakeCtx(const TTileCoord &tc, int toBase, TRegTileColumn<float> kScaleColumn)
    {
        TQKctx res;
        res.TileAlibi = GetAttentionDecay(AlibiFromMinusTo, AlibiSlope);
        res.qkScaleColumn = kScaleColumn;
        res.qkScaleColumn.Scale(Q_VEC_SCALE * AttDotScale); // qScale is constant VEC_SCALE
        return res;
    }

    __forceinline __device__ float CalcQK(const TQKctx &ctx, float qkProduct, int elem, int columnIndex) const
    {
        // float dp = qqTile.x[elem] * qScale[head][glFrom] * kScale[head][glTo] * attDotScale;
        // dp += GetAttentionDecay(glFrom - glTo, AlibiSlope);
        float dp = qkProduct * ctx.qkScaleColumn.x[columnIndex];
        dp += ctx.TileAlibi + ElemAlibi[elem];
        return dp;
    }

    __forceinline __device__ bool IsTaken(int x, int rowIndex) const
    {
        return (x >= RowSpanStart.x[rowIndex] && x <= RowSpanFinish.x[rowIndex]);
    }

    __device__ void NextTile()
    {
        RowSpanStart.Add(-TILE);
        RowSpanFinish.Add(-TILE);
        AlibiFromMinusTo -= TILE;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TQKReverseComputer
{
    TRegTileColumn<int> ColumnSpanStart;
    TRegTileColumn<int> ColumnSpanFinish;
    int AlibiFromMinusTo;
    float AttDotScale;
    float AlibiSlope;
    TRegTileColumn<float> qkScaleColumn;

    struct TQKctx
    {
        float ElemAlibi[TTileCoord::num_elements];
    };

public:
    __device__ TQKReverseComputer(const TTileCoord &tc, float alibiSlope, float attDotScale, TRegTileColumn<float> kScaleColumn)
    {
        qkScaleColumn = kScaleColumn;
        qkScaleColumn.Scale(Q_VEC_SCALE * attDotScale); // qScale is constant Q_VEC_SCALE
        AttDotScale = attDotScale;
        AlibiSlope = alibiSlope;
    }

    __device__ void StartSpan(const TTileCoord &tc, const TAttentionSpanGroup<ATT_GROUP> &gg, int toBase, int toWarp)
    {
        ColumnSpanStart.Load(tc, gg.SpanStart + toWarp);
        ColumnSpanFinish.Load(tc, gg.SpanFinish + toWarp);
        ColumnSpanStart.Add(-gg.Start);
        ColumnSpanFinish.Add(-gg.Start);
        AlibiFromMinusTo = gg.Start - toBase - toWarp;
    }

    __forceinline __device__ TQKctx MakeCtx(const TTileCoord &tc, int toBase, TRegTileRow<float> sumWeightLogRow)
    {
        TQKctx res;
        float tileAlibi = GetAttentionDecay(AlibiFromMinusTo, AlibiSlope);
        tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
            res.ElemAlibi[elem] = GetAttentionDecay(y - x, AlibiSlope) + tileAlibi - sumWeightLogRow.x[rowIndex];
        });
        return res;
    }

    __forceinline __device__ float CalcQK(const TQKctx &ctx, float qkProduct, int elem, int columnIndex) const
    {
        // float dp = qqTile.x[elem] * qScale[head][glFrom] * kScale[head][glTo] * attDotScale;
        // dp += GetAttentionDecay(glFrom - glTo, AlibiSlope);
        float dp = qkProduct * qkScaleColumn.x[columnIndex];
        dp += ctx.ElemAlibi[elem];
        return dp;
    }

    __forceinline __device__ bool IsTaken(int y, int columnIndex) const
    {
        return (y >= ColumnSpanStart.x[columnIndex] && y <= ColumnSpanFinish.x[columnIndex]);
    }

    __device__ void NextTile()
    {
        ColumnSpanStart.Add(-TILE);
        ColumnSpanFinish.Add(-TILE);
        AlibiFromMinusTo += TILE;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int HEAD_DIM>
__global__ void CalcDScale(TCuda2DPtr<half> vVecArr, TCuda2DPtr<half> dValLookup16, TCuda2DPtr<float> dScaleArr)
{
    CUDA_STATIC_ASSERT(HEAD_DIM == WARP_VEC_DIM);
    int head = blockIdx.x;
    int t = blockIdx.y;
    int offset = head * HEAD_DIM;
    float4 v;
    v = LoadWarpVec(vVecArr[t] + offset);
    v = Scale(v, V_VEC_SCALE);
    float4 dV = LoadWarpVec(dValLookup16[t] + offset);
    float dScale = DotProductWarpVec(v, dV);
    if (threadIdx.x == 0) {
        dScaleArr[head][t] = dScale;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAttGradResultShmem
{
    float Vecs[64][128];
};

template <class TGradStoreFunc, class TRes>
__forceinline __device__ void StoreAttGrad(const TTileCoord &tc, typename TGradStoreFunc::TParams &params, int warpId,
    TAttGradResultShmem *buf, TRegTile<float> *sumArr, int head, int dstY, TCuda2DPtr<TRes> resBuf)
{
    int warpBase = warpId * TILE;
    for (int sumX = 0; sumX < 8; ++sumX) {
        sumArr[sumX].Store(tc, TCuda2DPtr<float>(&buf->Vecs[warpBase][sumX * TILE], 128 * sizeof(float), 128, TILE));
    }
    __syncwarp();
    for (int y = 0; y < TILE; ++y) {
        TGradStoreFunc::StoreRow(params, head, head * 128, dstY + warpBase + y, buf->Vecs[warpBase + y], resBuf);
    }
    __syncwarp();
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreAtt
{
    struct TParams
    {
    };

    template <class TRes>
    __device__ static void StoreRow(TParams &params, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 v = LoadWarpVecSmem(mmRes);
        StoreWarpVec(resBuf[dstY] + dstX, v);
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreAttBackNormalizeRope
{
    struct TParams
    {
        TCuda2DPtr<TAttVecFloat> Q;
        TCuda2DPtr<float> QSrcTileScale;
        TCuda2DPtr<TRopeFloat> RopeBuf;
    };

    template <class TRes>
    __device__ static void StoreRow(TParams &params, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        // normally we should backprop rope then backprop tile normalization but we kept rotated Q/K only, so we perform backprop in reverse
        // order to get correct results
        float4 dQ = LoadWarpVecSmem(mmRes);

        float4 q = LoadWarpVec(params.Q[dstY] + dstX);
        q = Scale(q, params.QSrcTileScale[tile][dstY]);
        dQ = TileNormalizeBackpropWarpVec(q, dQ);

        float4 rope = LoadWarpVec(params.RopeBuf[dstY]);
        ApplyWarpRopeImpl(rope, -1, &dQ);

        StoreWarpVec(resBuf[dstY] + dstX, dQ);
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreAttBackNormalize
{
    struct TParams
    {
        TCuda2DPtr<TAttVecFloat> Q;
        TCuda2DPtr<float> QSrcTileScale;
    };

    template <class TRes>
    __device__ static void StoreRow(TParams &params, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 dQ = LoadWarpVecSmem(mmRes);

        float4 q = LoadWarpVec(params.Q[dstY] + dstX);
        q = Scale(q, params.QSrcTileScale[tile][dstY]);
        dQ = TileNormalizeBackpropWarpVec(q, dQ);

        StoreWarpVec(resBuf[dstY] + dstX, dQ);
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreAttBackRope
{
    struct TParams
    {
        TCuda2DPtr<TRopeFloat> RopeBuf;
    };

    template <class TRes>
    __device__ static void StoreRow(TParams &params, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 dQ = LoadWarpVecSmem(mmRes);

        float4 rope = LoadWarpVec(params.RopeBuf[dstY]);
        ApplyWarpRopeImpl(rope, -1, &dQ);
        
        StoreWarpVec(resBuf[dstY] + dstX, dQ);
        resBuf.CheckCoords(dstX, dstY);
    }
};
}