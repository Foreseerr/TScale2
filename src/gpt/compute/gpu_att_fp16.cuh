#pragma once
#include "cfg_precision.h"
#include "gpu_att_utils.cuh"
#include <gpt/att/att.h>
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/cuda_mma.cuh>
#include <lib/cuda/vec_util.cuh>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int ATT_PREFETCH_LEN = 3;

__forceinline __device__ int NextPrefetchBuf(int k, int prefetchLen)
{
    return (k >= prefetchLen - 1) ? 0 : k + 1;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Fp16Att(float *kGlobalScale, TCuda2DPtr<TAttVecFloat> q8, TCuda2DPtr<TAttVecFloat> k8, TCuda2DPtr<float> kScale,
    TCuda2DPtr<TAttVecFloat> v8, TCuda1DPtr<TAttentionSpanGroup<ATT_GROUP>> attSpans2, TCuda1DPtr<int> attSpanPtr, float alibiSlope,
    TCuda2DPtr<float> sumWeightLog, TCuda2DPtr<half> resMatr);
KERNEL_BLOCK_SIZE(Fp16Att, WARP_SIZE, 4);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFp16FastAttGradQData
{
    struct TStripe
    {
        T4SMemHalfTile KFrag[2];
        T4SMemHalfTile VFrag[2];
        float KScale[16];

        __forceinline __device__ void LoadAsync(int warpId, int head, int xOffset, int yOffset, TCuda2DPtr<TAttVecFloat> &k8,
            TCuda2DPtr<float> &kScale, TCuda2DPtr<TAttVecFloat> &v8)
        {
            CopyStripeAsync(KFrag, warpId, k8.Fragment(xOffset, yOffset));
            CopyStripeAsync(VFrag, warpId, v8.Fragment(xOffset, yOffset));
            CopyShortVecAsync<TILE>(KScale, kScale[head] + yOffset); // all warps copy same data
        }
    };

    union {
        struct
        {
            T4SMemHalfTile qFrag[4][2];
            T4SMemHalfTile dValFrag[4][2];
        };
        TStripe B[ATT_PREFETCH_LEN];
        TAttGradResultShmem dQ;
    };
};


template <class TGradQStore>
__global__ void __launch_bounds__(4 * WARP_SIZE, 2) //
    Fp16AttGradQ(float *kGlobalScale, TCuda2DPtr<TAttVecFloat> q8, TCuda2DPtr<TAttVecFloat> k8, TCuda2DPtr<float> kScale,
        TCuda2DPtr<TAttVecFloat> v8, TCuda2DPtr<half> dValLookup, TCuda2DPtr<float> dScaleArr, TCuda2DPtr<float> sumWeightLog,
        TCuda1DPtr<TAttentionSpanGroup<ATT_GROUP>> attSpans, TCuda1DPtr<int> attSpanPtr, float alibiSlope,
        TCuda2DPtr<TFastGradientFloat> dQres, typename TGradQStore::TParams dQStoreParams)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    CUDA_ASSERT(ATT_GROUP == 64);
    CUDA_ASSERT(ATT_ALIGN == 16);
    constexpr int qDim = 128;
    TTileCoord tc;

    // blockIdx.x - head group
    // blockIdx.y - len

    int head = blockIdx.x;
    float attDotScale = CalcDotScale(qDim) * CalcAttentionMult() * *kGlobalScale;

    int warpId = threadIdx.y;
    int eOffset = head * MM_TILE;
    int attBlock = blockIdx.y;
    int fromBase = attBlock * ATT_GROUP;
    int fromWarp = warpId * TILE;

    __shared__ TFp16FastAttGradQData data;

    // load k to shmem
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        CopyStripe(data.qFrag[fromTile], warpId, q8.Fragment(eOffset, fromBase + fromTile * TILE));
        CopyStripe(data.dValFrag[fromTile], warpId, dValLookup.Fragment(eOffset, fromBase + fromTile * TILE));
    }
    __syncthreads();
    TRegTile<half> qFrag[8];
    TRegTile<half> dValFrag[8];
    LoadStripe(qFrag, data.qFrag[warpId]);
    LoadStripe(dValFrag, data.dValFrag[warpId]);
    __syncthreads();

    TRegTileRow<float> dScaleRow;
    dScaleRow.Load(tc, dScaleArr[head] + fromBase + fromWarp);

    TQKComputer qkComputer(tc, alibiSlope, attDotScale);

    TRegTileRow<float> sumWeightLogRow;
    sumWeightLogRow.Load(tc, &sumWeightLog[head][fromBase + fromWarp]);
    qkComputer.SubtractSumWeightLog(tc, sumWeightLogRow);

    TRegTile<float> dQ[8];
    for (int sumX = 0; sumX < 8; ++sumX) {
        dQ[sumX].Clear();
    }

    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans[attIndex];
        qkComputer.StartSpan(tc, gg, fromBase, fromWarp);
        int toBase = gg.Start;
        int ggFinish = gg.Finish;

        // prefetch
        for (int k = 0; k < ATT_PREFETCH_LEN; ++k) {
            if (toBase + k * TILE <= ggFinish) {
                data.B[k].LoadAsync(warpId, head, eOffset, toBase + k * TILE, k8, kScale, v8);
            }
            AsyncCommitGroup();
        }
        int bufId = 0;
        for (; toBase <= ggFinish; toBase += TILE) {
            AsyncWaitGroup<ATT_PREFETCH_LEN - 1>();
            __syncthreads();

            // qqTile[from][to] = dot(q[from], k[to])
            TRegTile<float> qk = StripeDotProduct(qFrag, data.B[bufId].KFrag);

            // dW[from][to] = dot(dValLookup[from], v[to])
            TRegTile<float> dW = StripeDotProduct(dValFrag, data.B[bufId].VFrag);

            TRegTileColumn<float> kScaleColumn;
            kScaleColumn.Load(tc, data.B[bufId].KScale);
            TQKComputer::TQKctx qkCtx = qkComputer.MakeCtx(tc, toBase, kScaleColumn);

            TRegTile<half> dDotTile;
            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                if (qkComputer.IsTaken(x, rowIndex)) {
                    // int glFrom = fromBase + fromWarp + y;
                    // int glTo = toBase + toTile * TILE + x;
                    float dp = qkComputer.CalcQK(qkCtx, qk.x[elem], elem, columnIndex);
                    float w = exp2f(dp);

                    // log(2) from using exp2() instread of exp()
                    float dDot = w * (dW.x[elem] * V_VEC_SCALE - dScaleRow.x[rowIndex]) * attDotScale * LOG2;
                    dDotTile.x[elem] = dDot;
                } else {
                    dDotTile.x[elem] = 0;
                }
            });

            // add to dQ
            for (int i = 0; i < 4; ++i) {
                for (int xx = 0; xx < 2; ++xx) {
                    TRegTile<half> kTile = LoadTileTransposed(data.B[bufId].KFrag[xx], i);
                    kTile.Scale(kScaleColumn);
                    MMA(&dQ[i + xx * 4], dDotTile, kTile);
                }
            }
            __syncthreads();

            // increment & prefetch next
            qkComputer.NextTile();
            if (toBase + ATT_PREFETCH_LEN * TILE <= ggFinish) {
                data.B[bufId].LoadAsync(warpId, head, eOffset, toBase + ATT_PREFETCH_LEN * TILE, k8, kScale, v8);
            }
            AsyncCommitGroup();
            bufId = NextPrefetchBuf(bufId, ATT_PREFETCH_LEN);
        }
    }
    __syncthreads();
    StoreAttGrad<TGradQStore>(tc, dQStoreParams, warpId, &data.dQ, dQ, head, fromBase, dQres);
}
KERNEL_BLOCK_SIZE(Fp16AttGradQ, WARP_SIZE, 4);


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFp16AttGradKVData
{
    struct TStripe
    {
        T4SMemHalfTile QFrag[2];
        T4SMemHalfTile DValFrag[2];
        float SumWeightLog[TILE];
        float DScale[TILE];

        __forceinline __device__ void LoadAsync(int warpId, int head, int xOffset, int yOffset, TCuda2DPtr<TAttVecFloat> &q8,
            TCuda2DPtr<half> &dValLookup, TCuda2DPtr<float> &sumWeightLog, TCuda2DPtr<float> &dScaleArr)
        {
            CopyStripeAsync(QFrag, warpId, q8.Fragment(xOffset, yOffset));
            CopyStripeAsync(DValFrag, warpId, dValLookup.Fragment(xOffset, yOffset));
            CopyShortVecAsync<TILE>(SumWeightLog, sumWeightLog[head] + yOffset); // all warps copy same data
            CopyShortVecAsync<TILE>(DScale, dScaleArr[head] + yOffset); // all warps copy same data
        }
    };

    union {
        struct
        {
            T4SMemHalfTile KFrag[4][2];
            T4SMemHalfTile VFrag[4][2];
        };
        TStripe B[ATT_PREFETCH_LEN];
        TAttGradResultShmem dK;
        TAttGradResultShmem dV;
    };
};


template <class TGradKStore, class TGradVStore>
__global__ void __launch_bounds__(4 * WARP_SIZE, 2) //
    Fp16AttGradKV(float *kGlobalScale, TCuda2DPtr<TAttVecFloat> q8, TCuda2DPtr<TAttVecFloat> k8, TCuda2DPtr<float> kScale,
        TCuda2DPtr<TAttVecFloat> v8, TCuda2DPtr<half> dValLookup, TCuda2DPtr<float> dScaleArr, TCuda2DPtr<float> sumWeightLog,
        TCuda1DPtr<TAttentionSpanGroup<ATT_GROUP>> attSpans, TCuda1DPtr<int> attSpanPtr, float alibiSlope,
        TCuda2DPtr<TFastGradientFloat> dKres, TCuda2DPtr<TFastGradientFloat> dVres, typename TGradKStore::TParams dKStoreParams,
        typename TGradVStore::TParams dVStoreParams)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    CUDA_ASSERT(ATT_GROUP == 64);
    CUDA_ASSERT(ATT_ALIGN == 16);
    constexpr int qDim = 128;
    TTileCoord tc;

    // blockIdx.x - head group
    // blockIdx.y - len

    int head = blockIdx.x;
    float attDotScale = CalcDotScale(qDim) * CalcAttentionMult() * *kGlobalScale;

    int warpId = threadIdx.y;
    int eOffset = head * MM_TILE;
    int attBlock = blockIdx.y;
    int toBase = attBlock * ATT_GROUP;
    int toWarp = warpId * TILE;

    __shared__ TFp16AttGradKVData data;

    // load k to shmem
    for (int toTile = 0; toTile < 4; ++toTile) {
        CopyStripe(data.KFrag[toTile], warpId, k8.Fragment(eOffset, toBase + toTile * TILE));
        CopyStripe(data.VFrag[toTile], warpId, v8.Fragment(eOffset, toBase + toTile * TILE));
    }
    __syncthreads();
    TRegTile<half> kFrag[8];
    TRegTile<half> vFrag[8];
    LoadStripe(kFrag, data.KFrag[warpId]);
    LoadStripe(vFrag, data.VFrag[warpId]);
    __syncthreads();

    TRegTileColumn<float> kScaleColumn;
    kScaleColumn.Load(tc, kScale[head] + toBase + toWarp);
    TQKReverseComputer qkComputer(tc, alibiSlope, attDotScale, kScaleColumn);

    TRegTile<float> dK[8];
    TRegTile<float> dV[8];
    for (int sumX = 0; sumX < 8; ++sumX) {
        dK[sumX].Clear();
        dV[sumX].Clear();
    }

    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans[attIndex];
        qkComputer.StartSpan(tc, gg, toBase, toWarp);
        int fromBase = gg.Start;
        int ggFinish = gg.Finish;

        // prefetch
        for (int k = 0; k < ATT_PREFETCH_LEN; ++k) {
            if (fromBase + k * TILE <= ggFinish) {
                data.B[k].LoadAsync(warpId, head, eOffset, fromBase + k * TILE, q8, dValLookup, sumWeightLog, dScaleArr);
            }
            AsyncCommitGroup();
        }
        int bufId = 0;
        for (; fromBase <= ggFinish; fromBase += TILE) {
            AsyncWaitGroup<ATT_PREFETCH_LEN - 1>();
            __syncthreads();

            TRegTileRow<float> sumWeightLogRow;
            sumWeightLogRow.Load(tc, data.B[bufId].SumWeightLog);
            TQKReverseComputer::TQKctx qkCtx = qkComputer.MakeCtx(tc, toBase, sumWeightLogRow);

            // qqTile[from][to] = dot(q[from], k[to])
            TRegTile<float> qk = StripeDotProduct(data.B[bufId].QFrag, kFrag);

            // dV compute
            TRegTile<half> wTile;
            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                if (qkComputer.IsTaken(y, columnIndex)) {
                    // int glFrom = fromBase + y;
                    // int glTo = toBase + toWarp + x;
                    float dp = qkComputer.CalcQK(qkCtx, qk.x[elem], elem, columnIndex);
                    float w = exp2f(dp);
                    // wExactTile.x[elem] = w;
                    wTile.x[elem] = w;
                } else {
                    wTile.x[elem] = 0;
                }
            });
            wTile = wTile.Transpose();

            // add to dV
            for (int i = 0; i < 4; ++i) {
                for (int xx = 0; xx < 2; ++xx) {
                    MMA(&dV[i + xx * 4], wTile, LoadTileTransposed(data.B[bufId].DValFrag[xx], i));
                }
            }

            // dK compute
            // dW[from][to] = dot(dValLookup[from], v[to])
            TRegTile<float> dW = StripeDotProduct(data.B[bufId].DValFrag, vFrag);

            TRegTileRow<float> dScaleRow;
            dScaleRow.Load(tc, data.B[bufId].DScale);

            TRegTile<half> dDotTile;
            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                if (qkComputer.IsTaken(y, columnIndex)) {
                    // log(2) from using exp2() instread of exp()
                    float dDot = (dW.x[elem] * V_VEC_SCALE - dScaleRow.x[rowIndex]) * attDotScale * LOG2; // "* w" is postponed
                    dDotTile.x[elem] = dDot;
                } else {
                    dDotTile.x[elem] = 0;
                }
            });
            dDotTile = dDotTile.Transpose();
            dDotTile.Scale(wTile);

            // add to dK
            for (int i = 0; i < 4; ++i) {
                for (int xx = 0; xx < 2; ++xx) {
                    MMA(&dK[i + xx * 4], dDotTile, LoadTileTransposed(data.B[bufId].QFrag[xx], i));
                }
            }
            __syncthreads();
            qkComputer.NextTile();
            if (fromBase + ATT_PREFETCH_LEN * TILE <= ggFinish) {
                data.B[bufId].LoadAsync(warpId, head, eOffset, fromBase + ATT_PREFETCH_LEN * TILE, q8, dValLookup, sumWeightLog, dScaleArr);
            }
            AsyncCommitGroup();
            bufId = NextPrefetchBuf(bufId, ATT_PREFETCH_LEN);
        }
    }

    // save dK dV
    StoreAttGrad<TGradVStore>(tc, dVStoreParams, warpId, &data.dV, dV, head, toBase, dVres);
    for (int sumX = 0; sumX < 8; ++sumX) {
        dK[sumX].Scale(Q_VEC_SCALE);
    }
    StoreAttGrad<TGradKStore>(tc, dKStoreParams, warpId, &data.dK, dK, head, toBase, dKres);
}
KERNEL_BLOCK_SIZE(Fp16AttGradKV, WARP_SIZE, 4);

}
