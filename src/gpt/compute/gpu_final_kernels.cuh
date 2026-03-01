#pragma once

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// final layer kernels

// compute rowtile logusm
struct TStoreFinalLayerLogits : public TStoreRowBase
{
    struct TParams
    {
        float *ScalePtr;
        float NormScale;
        int VocabSize;
        TCuda1DPtr<float> BiasArr;
        int ComputeLogit;
        TCuda2DPtr<float> RowTileLogSum;
    };

    struct TWgShmem
    {
        float Bias[MM_TILE];
    };

    __device__ static float GetScale(TParams &params)
    {
        float sumScale = params.NormScale;
        if (params.ScalePtr) {
            sumScale *= *params.ScalePtr;
        }
        return sumScale;
    }

    __device__ static void Init(TParams &params, TWgShmem &shmem, int wgThreadId, int dstX, int dstY)
    {
        // load bias, assume all warps are sync
        int cc = wgThreadId; // expect 128 threads
        shmem.Bias[cc] = (dstX + cc < params.VocabSize) ? params.BiasArr[dstX + cc] : 0;
    }


    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        int h = threadIdx.x;
        int vocabSize = params.VocabSize;
        float4 vec = LoadWarpVecSmem(mmRes);
        vec = vec + LoadWarpVecSmem(shmem.Bias);

        if (params.ComputeLogit) {
            // mask out after vocabSize elements
            EnumWarpVecElements(&vec, [&](int idx, float *p) {
                if (dstX + idx >= vocabSize) {
                    *p = -1e38f;
                }
            });

            float maxVal = CalcWarpVecMax(vec);
            float sum = CalcWarpVecSum(Exp2f(vec - InitWarpVec(maxVal)));
            float logSum = log2f(sum) + maxVal;
            if (h == 0) {
                params.RowTileLogSum[tile][dstY] = logSum;
            }
            // store relative to rowtile max to preserve precision
            vec = vec - InitWarpVec(logSum);
        }
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        resBuf.CheckCoords(dstX, dstY);
    }
};


static __global__ void SumLogWeight(int tileCount, TCuda2DPtr<float> rowTileLogSum)
{
    int tile = blockIdx.x;
    int offset = tile * MM_TILE;

    // find max
    float4 maxVal = LoadWarpVecCached(rowTileLogSum[0] + offset);
    for (int tile = 1; tile < tileCount; ++tile) {
        float4 vec = LoadWarpVecCached(rowTileLogSum[tile] + offset);
        maxVal = Max(maxVal, vec);
    }
    // calc sum
    float4 sum = ZeroWarpVec();
    for (int tile = 0; tile < tileCount; ++tile) {
        float4 vec = LoadWarpVecCached(rowTileLogSum[tile] + offset);
        sum = sum + Exp2f(vec - maxVal);
    }
    float4 logSum = maxVal + Log2f(sum);
    // subtract sum
    for (int tile = 0; tile < tileCount; ++tile) {
        float4 vec = LoadWarpVecCached(rowTileLogSum[tile] + offset);
        vec = vec - logSum;
        StoreWarpVec(rowTileLogSum[tile] + offset, vec);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static __global__ void ComputeFinalProbKernel(int targetOffset, int vocabSize, TCuda2DPtr<float> logitRowTileLogSum, TCuda1DPtr<int> targetArr, TCuda2DPtr<half> logitBuf, TCuda1DPtr<float> resTargetProb)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;;
    int cc = targetArr[targetOffset + t];

    float rowTileLogSum = logitRowTileLogSum[tile][t];
    for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
        int c = offset + base + h;
        if (c < vocabSize) {
            float pred = exp2f(float(logitBuf[t][c]) + rowTileLogSum);
            logitBuf[t][c] = pred;
            if (c == cc) {
                resTargetProb[targetOffset + t] = pred;
            }
        } else {
            logitBuf[t][c] = 0;
        }
    }
}


static __global__ void ComputeGradient(
    int len,
    int targetOffset, int vocabSize,
    TCuda2DPtr<half> logitBuf, TCuda2DPtr<float> logitRowTileLogSum, TCuda1DPtr<int> targetArr,
    TCuda2DPtr<half> gradArr, TCuda1DPtr<float> sumTrainErr
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;
    int cc = -1;
    if (t < len) {
        cc = targetArr[targetOffset + t];
    }
    if (cc >= 0) {
        float rowTileLogSum = logitRowTileLogSum[tile][t];
        for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
            int c = offset + base + h;
            if (c < vocabSize) {
                float pred = exp2f(float(logitBuf[t][c]) + rowTileLogSum);
                pred = fmaxf(pred, 1e-20f); // avoid nans
                // omit scale gradient by log2, constant scale does not change anything
                if (c == cc) {
                    gradArr[t][c] = (1 - pred);
                    atomicAdd(&sumTrainErr[0], 1);
                    atomicAdd(&sumTrainErr[1], log(pred));
                } else {
                    gradArr[t][c] = -pred;
                }
            } else {
                gradArr[t][c] = 0;
            }
        }
    } else {
        for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
            int c = offset + base + h;
            gradArr[t][c] = 0;
        }
    }
}


static __global__ void CopyLogitGradient(
    int len,
    int targetOffset, int vocabSize,
    TCuda2DPtr<float> targetStateVectors,
    TCuda2DPtr<half> gradArr
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;
    if (t < len) {
        for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
            int c = offset + base + h;
            if (c < vocabSize) {
                gradArr[t][c] = targetStateVectors[targetOffset + t][c];
            } else {
                gradArr[t][c] = 0;
            }
        }
    } else {
        for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
            int c = offset + base + h;
            gradArr[t][c] = 0;
        }
    }
}


template <class TStateFloat, class TStateGradFloat>
__global__ void ComputeStateVectorGradient(
    int len,
    TCuda2DPtr<TStateFloat> finalState, TCuda2DPtr<float> targetStateVector,
    TCuda2DPtr<TStateGradFloat> gradArr, TCuda1DPtr<float> sumTrainErr
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    float4 grad;
    if (t < len) {
        float4 state = LoadWarpVec(finalState[t] + offset);
        float sum2 = CalcWarpVecSum2(state);
        if (sum2 > 0) {
            state = Scale(state, 1 / sqrt(sum2 / MM_TILE));
        }

        float4 target = LoadWarpVec(targetStateVector[t] + offset);
        sum2 = CalcWarpVecSum2(target);
        if (sum2 > 0) {
            target = Scale(target, 1 / sqrt(sum2 / MM_TILE));
        }

        grad = target - state;
        float err = CalcWarpVecSum2(grad);
        if (threadIdx.x == 0) {
            atomicAdd(&sumTrainErr[0], MM_TILE);
            atomicAdd(&sumTrainErr[1], err);
        }
    } else {
        grad = ZeroWarpVec();
    }
    StoreWarpVec(gradArr[t] + offset, grad);
}


static __global__ void CollectSumTrainErr(TCuda1DPtr<float> sumTrainErr)
{
    if (threadIdx.x == 0) {
        sumTrainErr[2] += sumTrainErr[0];
        sumTrainErr[3] += sumTrainErr[1];
        sumTrainErr[0] = 0;
        sumTrainErr[1] = 0;
    }
}


template <class TStateGradFloat>
__global__ void CopyGradient(int len, TCuda2DPtr<float> targetStateVector, TCuda2DPtr<TStateGradFloat> gradArr)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    float4 grad;
    if (t < len) {
        grad = LoadWarpVec(targetStateVector[t] + offset);
    } else {
        grad = ZeroWarpVec();
    }
    StoreWarpVec(gradArr[t] + offset, grad);
}


static __global__ void ComputeLossKernel(int offset, int len, TCuda2DPtr<half> logitBuf, TCuda2DPtr<float> logitRowTileLogSum, TCuda1DPtr<int> targetArr, TCuda1DPtr<float> resArr)
{
    int h = threadIdx.x;
    float sum = 0;
    for (int base = 0; base < len; base += WARP_SIZE) {
        int t = base + h;
        if (t < len) {
            int target = targetArr[offset + t];
            if (target >= 0) {
                sum += (float(logitBuf[t][target]) + logitRowTileLogSum[target / MM_TILE][t]) * LOG2;
            }
        }
    }
    sum = WarpSum(sum);
    if (h == 0) {
        resArr[0] += sum;
    }
}
}
