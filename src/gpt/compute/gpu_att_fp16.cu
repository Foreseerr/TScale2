#include <util/pch.h>
#define KERNEL_UNIT "gpt_att_fp16/"
#include "gpu_att_fp16.cuh"
#include <lib/hp_timer/hp_timer.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFp16FastAttData
{
    struct TStripe
    {
        T4SMemHalfTile KFrag[2];
        T4SMemHalfTile VFrag[2];
        float KScale[TILE];

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
            T4SMemHalfTile QFrag[4][2]; // from 64 samples
        } A;
        TStripe B[ATT_PREFETCH_LEN];
    };
    float RowMax[64];
    float NewRowMax[64];
    float RowScale[64];
    float SumWeight[64];
};


__global__ void __launch_bounds__(4 * WARP_SIZE, 3) //
    Fp16Att(float *kGlobalScale, TCuda2DPtr<TAttVecFloat> q8, TCuda2DPtr<TAttVecFloat> k8, TCuda2DPtr<float> kScale,
        TCuda2DPtr<TAttVecFloat> v8, TCuda1DPtr<TAttentionSpanGroup<ATT_GROUP>> attSpans2, TCuda1DPtr<int> attSpanPtr, float alibiSlope,
        TCuda2DPtr<float> sumWeightLog, TCuda2DPtr<half> resMatr)
{
    CUDA_ASSERT(MM_TILE == 128);
    CUDA_ASSERT(ATT_GROUP == 64);
    CUDA_ASSERT(ATT_ALIGN == 16);
    constexpr int qDim = 128;
    TTileCoord tc;

    // blockIdx.x - head group
    // blockIdx.y - len

    int head = blockIdx.x;
    float attDotScale = CalcDotScale(qDim) * CalcAttentionMult() * *kGlobalScale;

    int h = threadIdx.x;
    int warpId = threadIdx.y;
    int eOffset = head * MM_TILE;
    int attBlock = blockIdx.y;
    int fromBase = attBlock * ATT_GROUP;
    int fromWarp = warpId * TILE;

    __shared__ TFp16FastAttData data;

    // load q
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        CopyStripe(data.A.QFrag[fromTile], warpId, q8.Fragment(eOffset, fromBase + fromTile * TILE));
    }
    __syncthreads();
    TRegTile<half> qFrag[8];
    LoadStripe(qFrag, data.A.QFrag[warpId]);
    __syncthreads();

    TQKComputer qkComputer(tc, alibiSlope, attDotScale);

    TRegTile<float> sum[8];
    for (int sumX = 0; sumX < 8; ++sumX) {
        sum[sumX].Clear();
    }

    TRegTileRow<float> rowMax;
    rowMax.SetMax(0);
    rowMax.Store(tc, data.RowMax + fromWarp);
    TRegTileRow<float> sumWeight;
    sumWeight.Clear();

    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
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

            TRegTile<float> qk = StripeDotProduct(qFrag, data.B[bufId].KFrag);

            TRegTileColumn<float> kScaleColumn;
            kScaleColumn.Load(tc, data.B[bufId].KScale);
            TQKComputer::TQKctx qkCtx = qkComputer.MakeCtx(tc, toBase, kScaleColumn);

            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                constexpr float MIN_DOT_PRODUCT = -1e5f;
                float dp = MIN_DOT_PRODUCT;
                if (qkComputer.IsTaken(x, rowIndex)) {
                    // int glFrom = fromBase + fromWarp + y;
                    // int glTo = toBase + toTile * TILE + x;
                    dp = qkComputer.CalcQK(qkCtx, qk.x[elem], elem, columnIndex);
                }
                qk.x[elem] = dp;
            });


            // test if taken and if we have new dot product max
            unsigned hasNewMax = false;
            constexpr float THRESHOLD = 3;
            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                float dp = qk.x[elem];
                hasNewMax |= (dp > rowMax.x[rowIndex] + THRESHOLD);
            });

            if (__reduce_or_sync(0xffffffff, hasNewMax)) {
                // rescale to new dotproduct max
                tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                    float dp = qk.x[elem];
                    rowMax.x[rowIndex] = max(rowMax.x[rowIndex], dp);
                });
                for (int rowIndex = 0; rowIndex < rowMax.GetNumElements(); ++rowIndex) {
                    rowMax.x[rowIndex] = ceil(rowMax.x[rowIndex]);
                }
                rowMax.WarpMaxReduce();
                rowMax.Store(tc, data.NewRowMax + fromWarp);
                __syncwarp();

                // compute online result scale, write data.OnlineRowScale
                if (h < TILE) {
                    int y = fromWarp + h;
                    float oldMax = data.RowMax[y];
                    float newMax = data.NewRowMax[y];
                    data.RowScale[y] = exp2f(oldMax - newMax);
                    data.RowMax[y] = newMax;
                }
                __syncwarp();

                // scale sum by online scale results, read data.OnlineRowScale
                TRegTileRow<float> rowScale;
                rowScale.Load(tc, data.RowScale + fromWarp);
                sumWeight.Scale(rowScale);
                for (int sumX = 0; sumX < 8; ++sumX) {
                    sum[sumX].Scale(rowScale);
                }
                __syncwarp();
            }

            // compute weights
            TRegTile<half> wTile;
            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                float dp = qk.x[elem];
                dp -= rowMax.x[rowIndex];
                float weight = exp2f(dp);
                sumWeight.x[rowIndex] += weight;
                wTile.x[elem] = weight;
            });

            // add to sum scaled v[], read data.vFrag, data.weights
            for (int i = 0; i < 4; ++i) {
                MMA(&sum[i], wTile, LoadTileTransposed(data.B[bufId].VFrag[0], i));
                MMA(&sum[i + 4], wTile, LoadTileTransposed(data.B[bufId].VFrag[1], i));
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
    sumWeight.StoreSum(tc, data.SumWeight + fromWarp);
    __syncthreads();
    if (h < TILE) {
        float sumWeight = data.SumWeight[fromWarp + h];
        float dpMax = data.RowMax[fromWarp + h];
        if (dpMax < -20) {
            data.RowScale[fromWarp + h] = 0;
            sumWeightLog[head][fromBase + fromWarp + h] = 0;
        } else {
            sumWeight += exp2f(-dpMax); // add zero vector with weight 1
            data.RowScale[fromWarp + h] = 1.0f / sumWeight;
            sumWeightLog[head][fromBase + fromWarp + h] = dpMax + log2f(sumWeight);
        }
    }
    __syncthreads();

    TRegTileRow<float> rowScale;
    rowScale.Load(tc, data.RowScale + fromWarp);
    for (int sumX = 0; sumX < 8; ++sumX) {
        TRegTile<float> &rsum = sum[sumX];
        rsum.Scale(rowScale);
        rsum.Store(tc, resMatr.Fragment(eOffset + sumX * TILE, fromBase + fromWarp));
    }
}


}

///////////////////////////////////////////////////////////////////////////////////////////////////
// perf tests
using namespace NCuda;
static void PutValue(TStream &stream, TCudaVector<float> *p, float val)
{
    TVector<float> vec;
    vec.push_back(val);
    Put(stream, p, vec);
}

void TestAttFp16()
{
    Y_VERIFY(ATT_TYPE == ATT_FP16);
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<TAttVecFloat> q;
    TCuda2DArray<float> qScale;
    TCuda2DArray<TAttVecFloat> k;
    TCuda2DArray<float> kScale;
    TCuda2DArray<TAttVecFloat> v;
    TCuda2DArray<half> valLookup;
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> attSpans;
    TCudaVector<int> attSpanPtr;
    TCuda2DArray<float> sumWeightLog;
    TCudaVector<float> kGlobalScale;

    // const int ITER_COUNT = 1;
    const int ITER_COUNT = 10;

    yint len = 16 * 1024; // sample count
    yint headCount = 4;
    yint qSum = 128 * headCount;
    yint ttSum = 128 * headCount;
    int spanGroups = DivCeil(len, ATT_GROUP);
    q.Allocate(qSum, len);
    qScale.Allocate(headCount, len);
    k.Allocate(qSum, len);
    kScale.Allocate(headCount, len);
    v.Allocate(ttSum, len);
    valLookup.Allocate(ttSum, len);
    attSpans.Allocate(spanGroups * 4);
    attSpanPtr.Allocate(spanGroups + 1);
    sumWeightLog.AllocateCuda(len, headCount);
    kGlobalScale.Allocate(1);

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            yint lenTiles = len / ATT_GROUP;
            float alibiSlope = 0.01f;
            TCudaPOD<float> kgs = kGlobalScale.GetElement(0);
            CudaCall(c, Fp16Att)
                .Grid(headCount, lenTiles)
                .Read(kgs, q, k, kScale)
                .Read(v)
                .Read(attSpans, attSpanPtr, alibiSlope)
                .Write(&sumWeightLog, &valLookup);
        }
    }

    // create all to all attention graph
    TAttentionInfoGrouped<ATT_GROUP> aig;
    aig.Init();
    for (yint t = 0; t < len / ATT_GROUP; ++t) {
        TAttentionSpanGroup<ATT_GROUP> ag;
        for (yint k = 0; k < ATT_GROUP; ++k) {
            ag.SpanStart[k] = 0;
            ag.SpanFinish[k] = len - 1;
        }
        ag.Start = 0;
        ag.Finish = len - MM_TILE;
        TVector<TAttentionSpanGroup<ATT_GROUP>> agArr;
        agArr.push_back(ag);
        aig.AddSpanGroups(agArr);
    }
    Put(stream, &attSpans, aig.SpanGroups);
    Put(stream, &attSpanPtr, aig.SpanGroupPtr);

    PutValue(stream, &kGlobalScale, 1.0f);

    FillRandom(rng, stream, &q);
    qScale.ClearDeviceMem(stream);
    FillRandom(rng, stream, &k);
    kScale.ClearDeviceMem(stream);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = ITER_COUNT * (qSum * len * len + ttSum * len * len) * 2. / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops, %g\n", maxTFlops, tFlops);
    }
}


void TestAttGradKfp16()
{
    Y_VERIFY(ATT_TYPE == ATT_FP16);
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<TAttVecFloat> q;
    TCuda2DArray<float> qScale;
    TCuda2DArray<half> qSrc;
    TCuda2DArray<TAttVecFloat> k;
    TCuda2DArray<float> kScale;
    TCuda2DArray<TAttVecFloat> v;
    TCuda2DArray<half> dValLookup;
    TCuda2DArray<float> vScale;
    TCuda2DArray<half> dV;
    TCuda2DArray<half> dK;
    TCuda2DArray<half> dQ;
    TCuda2DArray<float> dScaleArr;
    TCuda2DArray<float> sumWeightLog;
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> attSpans;
    TCudaVector<int> attSpanPtr;
    TCudaVector<float> kGlobalScale;

    // const int ITER_COUNT = 1;
    const int ITER_COUNT = 10;

    yint len = 16 * 1024; // sample count
    yint headCount = 4;
    yint qSum = 128 * headCount;
    yint ttSum = 128 * headCount;
    int spanGroups = DivCeil(len, ATT_GROUP);
    q.Allocate(qSum, len);
    qScale.Allocate(headCount, len);
    qSrc.Allocate(qSum, len);
    k.Allocate(qSum, len);
    kScale.Allocate(headCount, len);
    v.Allocate(ttSum, len);
    vScale.Allocate(headCount, len);
    dValLookup.Allocate(ttSum, len);
    dV.Allocate(ttSum, len);
    dK.Allocate(qSum, len);
    dQ.Allocate(qSum, len);
    dScaleArr.Allocate(len, headCount);
    dScaleArr.ClearDeviceMem(stream);
    sumWeightLog.Allocate(len, headCount);
    sumWeightLog.ClearDeviceMem(stream);
    attSpans.Allocate(spanGroups * 4);
    attSpanPtr.Allocate(spanGroups + 1);
    kGlobalScale.Allocate(1);

    float usefulOps = 0;
    usefulOps += (2 * qSum * len * len + ttSum * len * len) * 2.; // grad Q
    usefulOps += (2 * qSum * len * len + 2 * ttSum * len * len) * 2.; // grad KV

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            yint lenAttTiles = len / ATT_GROUP;
            float alibiSlope = 0.01f;
            TCudaPOD<float> kgs = kGlobalScale.GetElement(0);
            CudaCall(c, Fp16AttGradQ<TStoreAtt>)
                .Grid(headCount, lenAttTiles)
                .Read(kgs, q, k, kScale)
                .Read(v)
                .Read(dValLookup)
                .Read(dScaleArr, sumWeightLog)
                .Read(attSpans, attSpanPtr, alibiSlope)
                .Write(&dQ)
                .Struct();

            CudaCall(c, Fp16AttGradKV<TStoreAtt, TStoreAtt>)
                .Grid(headCount, lenAttTiles)
                .Read(kgs, q, k, kScale)
                .Read(v)
                .Read(dValLookup)
                .Read(dScaleArr, sumWeightLog)
                .Read(attSpans, attSpanPtr, alibiSlope)
                .Write(&dK, &dV)
                .Struct()
                .Struct();
        }
    }

    // create all to all attention graph
    TAttentionInfoGrouped<ATT_GROUP> aig;
    aig.Init();
    for (yint t = 0; t < len / ATT_GROUP; ++t) {
        TAttentionSpanGroup<ATT_GROUP> ag;
        for (yint k = 0; k < ATT_GROUP; ++k) {
            ag.SpanStart[k] = 0;
            ag.SpanFinish[k] = len - 1;
        }
        ag.Start = 0;
        ag.Finish = len - MM_TILE;
        TVector<TAttentionSpanGroup<ATT_GROUP>> agArr;
        agArr.push_back(ag);
        aig.AddSpanGroups(agArr);
    }
    Put(stream, &attSpans, aig.SpanGroups);
    Put(stream, &attSpanPtr, aig.SpanGroupPtr);

    PutValue(stream, &kGlobalScale, 1.0f);

    FillRandom(rng, stream, &q);
    FillRandom(rng, stream, &k);
    FillRandom(rng, stream, &v);
    FillRandom(rng, stream, &dValLookup);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = ITER_COUNT * usefulOps / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops, %g\n", maxTFlops, tFlops);
    }
}
