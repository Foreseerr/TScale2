#include <util/pch.h>
#define KERNEL_UNIT "moe_kernels/"
#include "moe_kernels.cuh"
#include <lib/cuda/cuda_util.cuh>
#include <lib/hp_timer/hp_timer.h>
#include <lib/random/rand_utils.h>


/////////////////////////////////////////////////////////////////////////////////////
namespace NCuda
{
// Kernels
template <class F>
inline __device__ void CudaForEach(int dim, F func)
{
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int x = base + h;
        if (x < dim) {
            func(x);
        }
    }
}

template <class TSrc, class TDst>
inline __device__ void CopyScaledVec(int dim, TDst *dst, TSrc *src, float scale)
{
    CudaForEach(dim, [&](int x) { StoreConvertedFloat(float(src[x]) * scale, dst + x); });
}

template <class TDst>
inline __device__ void FillVec(int dim, TDst *dst, float val)
{
    CudaForEach(dim, [&](int x) { StoreConvertedFloat(val, dst +x); });
}


inline __device__ int SelectTop(int dim, float *buf)
{
    float top = -1e10;
    int topIdx = 0;
    CudaForEach(dim, [&](int x) {
        float val = buf[x];
        if (val > top) {
            top = val;
            topIdx = x;
        }
    });
    return WarpMaxIdx(top, topIdx);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void MoeCalcSampleExpertLists(
    int expertCount, int selectedCount, TCuda2DPtr<half> moeArr, TCuda1DPtr<int> sampleExpertIds, TCuda1DPtr<float> sampleExpertWeights)
{
    int t = blockIdx.x;
    int h = threadIdx.x;

    extern __shared__ float floatBuf[];
    float *moeSrc = floatBuf;
    float *moe = moeSrc + expertCount;
    int *topIdxArr = (int *)(moe + selectedCount);

    // load moe src
    CopyScaledVec(expertCount, moeSrc, moeArr[t], 1.0f);

    // select top
    for (int z = 0; z < selectedCount; ++z) {
        int topIdx = SelectTop(expertCount, moeSrc);

        if (h == 0) {
            float top = moeSrc[topIdx];
            moeSrc[topIdx] = -1e10;
            moe[z] = top;
            topIdxArr[z] = topIdx;
        }
        __syncwarp();
    }
    float maxVal = moe[0];

    // assign weights to selected experts
    float sum2 = 0;
    CudaForEach(selectedCount, [&](int x) {
        float w = expf((moe[x] - maxVal) * MOE_SCALE);
        moe[x] = w;
        sum2 += w * w;
    });
    sum2 = WarpSum(sum2);

    // compute scale
    float scale = 1;
    if (sum2 > 0) {
        scale = sqrt(expertCount / sum2);
    }

    // store result
    int writeOffset = t * selectedCount;
    CudaForEach(selectedCount, [&](int z) {
        int dstIdx = writeOffset + z;
        sampleExpertIds[dstIdx] = topIdxArr[z];
        sampleExpertWeights[dstIdx] = moe[z] * scale;
    });
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TGroupExpertListData
{
    int Data[1];

    __device__ int *GetExpertSampleCount(int expertCount, int warpId) { return Data + warpId * expertCount; }
};


__global__ void GroupPerExpertLists(int len, int lenMoe, int expertCount, int selectedCount, TCuda1DPtr<int> sampleExpertIds,
    TCuda1DPtr<int> expertOffsetArr, TCuda1DPtr<int> sampleExpertOffset, TCuda1DPtr<int> xpTileExpertId)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    int blockThrId = h + warpId * WARP_SIZE;
    extern __shared__ char smem_data[];
    TGroupExpertListData &smem = *(TGroupExpertListData*)smem_data;

    int *localCount = smem.GetExpertSampleCount(expertCount, warpId);
    // last expert offset is equal to total count after assignment
    int *expertSampleCount = smem.GetExpertSampleCount(expertCount, blockDim.y - 1);

    CudaForEach(expertCount, [&](int x) { localCount[x] = 0; });
    __syncthreads();

    // count samples per expert for each warp separately
    int sz = len * selectedCount;
    for (int t = blockThrId; t < sz; t += WARP_SIZE * blockDim.y) {
        int expertId = sampleExpertIds[t];
        atomicAdd(&localCount[expertId], 1);
    }
    __syncthreads();

    // sum counts to get offsets
    if (warpId == 0) {
        CudaForEach(expertCount, [&](int x) {
            int sum = 0;
            for (int k = 0; k < blockDim.y; ++k) {
                int oldSum = sum;
                int *warpCounts = smem.GetExpertSampleCount(expertCount, k);
                sum += warpCounts[x];
                warpCounts[x] = oldSum;
            }
        });
    }
    __syncthreads();

    // assign samples to experts
    for (int t = blockThrId; t < sz; t += WARP_SIZE * blockDim.y) {
        int expertId = sampleExpertIds[t];
        int ptr = atomicAdd(&localCount[expertId], 1);
        sampleExpertOffset[t] = ptr;
    }
    __syncthreads();

    // all the rest is computed in single warp
    if (warpId != 0) {
        return;
    }

    // round up counts
    CudaForEach(expertCount, [&](int expertId) {
        int count = expertSampleCount[expertId];
        count = (count + MM_TILE - 1) & ~(MM_TILE - 1);
        expertSampleCount[expertId] = count;
    });
    __syncthreads();

    // compute prefix sum and store expert offset
    for (int bit = 1; bit < expertCount; bit *= 2) {
        CudaForEach(expertCount, [&](int x) {
            if (x & bit) {
                expertSampleCount[x] += expertSampleCount[(x ^ bit) | (bit - 1)];
            }
        });
        __syncthreads();
    }
    CudaForEach(expertCount, [&](int x) { expertOffsetArr[x + 1] = expertSampleCount[x]; });
    if (h == 0) {
        expertOffsetArr[0] = 0;
        expertOffsetArr[expertCount + 1] = lenMoe;
    }
    __syncthreads();

    // fill tileExpertId
    CudaForEach(expertCount + 1, [&](int x) {
        int kStart = expertOffsetArr[x] / MM_TILE;
        int kFinish = expertOffsetArr[x + 1] / MM_TILE;
        int resTile = (x == expertCount) ? -1 : x;
        for (int k = kStart; k < kFinish; ++k) {
            xpTileExpertId[k] = resTile;
        }
    });
}


__global__ void InitXpTables(TCuda1DPtr<int> xpSampleId, TCuda1DPtr<float> xpSampleWeight)
{
    int thrId = threadIdx.x + threadIdx.y * WARP_SIZE;
    int offset = blockIdx.x * MM_TILE;
    int t = offset + thrId;
    xpSampleId[t] = -1;
    xpSampleWeight[t] = 0;
}


__global__ void FillXpTables(int len, TIntDivision selectedCount, TCuda1DPtr<int> sampleExpertIds, TCuda1DPtr<float> sampleExpertWeights,
    TCuda1DPtr<int> sampleExpertOffset, TCuda1DPtr<int> expertOffsetArr, TCuda1DPtr<int> xpSampleId, TCuda1DPtr<float> xpSampleWeight)
{
    int thrId = threadIdx.x + threadIdx.y * WARP_SIZE;
    int offset = blockIdx.x * MM_TILE;
    int t = offset + thrId;
    if (t < len * selectedCount) {
        int expertId = sampleExpertIds[t];
        int dstPtr = expertOffsetArr[expertId] + sampleExpertOffset[t];
        xpSampleId[dstPtr] = selectedCount.Div(t);
        xpSampleWeight[dstPtr] = sampleExpertWeights[t];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void BackpropRowTileNormalizeLoLUmoe(int saveTransposed, TCudaSpan expertSpan, TCuda1DPtr<int> tileExpert, int xTileCount,
    TCuda1DPtr<float> xpSampleWeight, TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valsArr, float valsScale,
    TCuda2DPtr<float> valsSrcTileScale, TCuda2DPtr<half> gradArr, float gradNormScale, //
    TCuda2DPtr<TFastGradientFloat> gradGate, TCuda2DPtr<TFastGradientFloat> gradGateT, TCuda1DPtr<float> dExpertWeightArr,
    TCuda2DPtr<TFastGradientFloat> gradVals, TCuda2DPtr<TFastGradientFloat> gradValsT)
{
    int tTile = blockIdx.x;
    int tOffset = tTile * MM_TILE;
    int expertId = tileExpert[tTile];
    if (expertId < expertSpan.Beg || expertId >= expertSpan.Fin) {
        return;
    }

    extern __shared__ char shmem[];
    TBackpropNormLoLUmoeData &data = *(TBackpropNormLoLUmoeData *)shmem;

    if (threadIdx.y == 0) {
        for (int k = threadIdx.x; k < MM_TILE; k += WARP_SIZE) {
            data.GradMoe[k] = 0;
        }
    }
    __syncthreads();

    for (int xTile = 0; xTile < xTileCount; ++xTile) {
        int xOffset = xTile * MM_TILE;
        for (int k = threadIdx.y; k < MM_TILE; k += blockDim.y) {
            int t = tOffset + k;

            float moe = xpSampleWeight[t];
            float4 gate = LoadWarpVec(gateArr[t] + xOffset);
            float4 vals = LoadWarpVec(valsArr[t] + xOffset);
            vals = Scale(vals, valsScale);

            // load rg grad
            float4 grad = LoadWarpVec(gradArr[t] + xOffset);
            grad = Scale(grad, gradNormScale);

            // backprop alu
            float4 mult = ComputeLoLU(gate);
            float4 resGradGate = Scale(grad * vals * ComputeLoLUGrad(mult), moe);
            float4 resGradVals = Scale(grad * mult, moe);
            float resGradMoe = CalcWarpVecSum(grad * vals * mult);
            if (threadIdx.x == 0) {
                data.GradMoe[k] += resGradMoe;
            }
            StoreWarpVec(gradGate[t] + xOffset, resGradGate);

            // backprop row tile normalize
            vals = Scale(vals, valsSrcTileScale[xTile][t] / valsScale);
            float4 valsNormGrad = TileNormalizeBackpropWarpVec(vals, resGradVals);
            StoreWarpVec(gradVals[t] + xOffset, valsNormGrad);

            // store transposed
            if (saveTransposed) {
                data.DGate.StoreVecTransposed(k, resGradGate);
                data.DVals.StoreVecTransposed(k, valsNormGrad);
            }
        }
        if (saveTransposed) {
            __syncthreads();
            data.DGate.CopyToGmem(gradGateT, tOffset, xOffset);
            data.DVals.CopyToGmem(gradValsT, tOffset, xOffset);
            __syncthreads();
        }
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        for (int k = threadIdx.x; k < MM_TILE; k += WARP_SIZE) {
            int t = tOffset + k;
            dExpertWeightArr[t] = data.GradMoe[k];
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void BackpropMoeKernel(int len, int bufferDim, int expertCount, int selectedCount, TCuda1DPtr<int> sampleExpertIds,
    TCuda1DPtr<float> sampleExpertWeights, TCuda1DPtr<int> sampleExpertOffset, TCuda1DPtr<int> expertOffsetArr,
    TCuda1DPtr<float> dExpertWeightArr, TCuda2DPtr<half> dMoeArr)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    CUDA_ASSERT(selectedCount <= WARP_SIZE);

    extern __shared__ float floatBuf[];
    float *moe = floatBuf;
    float *dMoe = floatBuf + expertCount;

    if (t < len) {
        // load
        float moeScale = sqrtf(expertCount);
        FillVec(expertCount, moe, 0);
        FillVec(expertCount, dMoe, 0);
        if (h < selectedCount) {
            int srcIdx = t * selectedCount + h;
            int expertId = sampleExpertIds[srcIdx];
            int row = sampleExpertOffset[srcIdx] + expertOffsetArr[expertId];

            // load moe
            moe[expertId] = sampleExpertWeights[srcIdx] * (1 / moeScale); // unscaled, sum squares equal 1

            // load dMoe
            dMoe[expertId] = dExpertWeightArr[row] * moeScale;
        }
        __syncwarp();

        // compute gradient
        float avrgGrad = 0;
        CudaForEach(expertCount, [&](int x) { avrgGrad += moe[x] * dMoe[x]; });
        avrgGrad = WarpSum(avrgGrad);

        CudaForEach(expertCount, [&](int x) {
            float w = moe[x];
            dMoe[x] = (w * dMoe[x] - w * w * avrgGrad) * MOE_SCALE;
        });
        __syncwarp();

        // store result
        CopyScaledVec(expertCount, dMoeArr[t], dMoe, 1.0f);
        // clear buffer tail
        FillVec(bufferDim - expertCount, dMoeArr[t] + expertCount, 0);
    } else {
        // clear buffer tail
        FillVec(bufferDim, dMoeArr[t], 0);
    }
}
}
