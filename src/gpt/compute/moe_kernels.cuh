#pragma once
#include "cfg_precision.h"
#include "gpu_lolu.cuh"
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/cuda_transpose.cuh>
#include <lib/cuda/fast_div.cuh>
#include <lib/cuda/vec_util.cuh>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst>
__global__ void MatrixLoLUmoe(TCudaSpan expertSpan, TCuda1DPtr<int> tileExpert, TCuda1DPtr<float> xpSampleWeight,
    TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valsArr, TCuda2DPtr<TDst> resArr)
{
    int xTile = blockIdx.x;
    int offset = xTile * MM_TILE;
    int tTile = blockIdx.y;
    int warpId = threadIdx.y;
    int expertId = tileExpert[tTile];
    if (expertId < expertSpan.Beg || expertId >= expertSpan.Fin) {
        return;
    }

    constexpr int SZ = MM_TILE / MAX_WARPS;
    float4 gate[SZ];
    float4 vals[SZ];
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        gate[k] = LoadWarpVec(gateArr[t] + offset);
        vals[k] = LoadWarpVec(valsArr[t] + offset);
    }
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        float moe = xpSampleWeight[t];
        float4 vec = vals[k] * ComputeLoLU(gate[k]);
        vec = Scale(vec, moe);
        StoreWarpVec(resArr[t] + offset, vec);
    }
}
KERNEL_BLOCK_SIZE(MatrixLoLUmoe, WARP_SIZE, MAX_WARPS);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void MoeCalcSampleExpertLists(int expertCount, int selectedCount, TCuda2DPtr<half> moeArr,
    TCuda1DPtr<int> sampleExpertIds, TCuda1DPtr<float> sampleExpertWeights);

__global__ void GroupPerExpertLists(int len, int lenMoe, int expertCount, int selectedCount, TCuda1DPtr<int> sampleExpertIds,
    TCuda1DPtr<int> expertOffsetArr, TCuda1DPtr<int> sampleExpertOffset, TCuda1DPtr<int> xpTileExpertId);

__global__ void InitXpTables(TCuda1DPtr<int> xpSampleId, TCuda1DPtr<float> xpSampleWeight);

__global__ void FillXpTables(int len, TIntDivision selectedCount, TCuda1DPtr<int> sampleExpertIds, TCuda1DPtr<float> sampleExpertWeights,
    TCuda1DPtr<int> sampleExpertOffset, TCuda1DPtr<int> expertOffsetArr, TCuda1DPtr<int> xpSampleId, TCuda1DPtr<float> xpSampleWeight);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TBackpropNormLoLUmoeData
{
    TTransposeBuf<TFastGradientFloat> DGate;
    TTransposeBuf<TFastGradientFloat> DVals;
    float GradMoe[MM_TILE];
};

__global__ void BackpropRowTileNormalizeLoLUmoe(int saveTransposed, TCudaSpan expertSpan, TCuda1DPtr<int> tileExpert, int xTileCount,
    TCuda1DPtr<float> xpSampleWeight, TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valsArr, float valsScale,
    TCuda2DPtr<float> valsSrcTileScale, TCuda2DPtr<half> gradArr, float gradNormScale, //
    TCuda2DPtr<TFastGradientFloat> gradGate, TCuda2DPtr<TFastGradientFloat> gradGateT, TCuda1DPtr<float> dExpertWeightArr,
    TCuda2DPtr<TFastGradientFloat> gradVals, TCuda2DPtr<TFastGradientFloat> gradValsT);

__global__ void BackpropMoeKernel(int len, int bufferDim, int expertCount, int selectedCount, TCuda1DPtr<int> sampleExpertIds,
    TCuda1DPtr<float> sampleExpertWeights, TCuda1DPtr<int> sampleExpertOffset, TCuda1DPtr<int> expertOffsetArr,
    TCuda1DPtr<float> dExpertWeightArr, TCuda2DPtr<half> dMoeArr);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int MOE_GROUP_DATA_WARPS = 32;

template <class T>
struct TMoeGroupSamplesData
{
    TTransposeBuf<T> Buf;
    float SampleId[MM_TILE];
};

template <class TFloat>
__global__ void MoeGroupSamples(int saveTransposed, TCudaSpan expertSpan, TCuda1DPtr<int> expertOffsetArr, TCuda2DPtr<TFloat> src,
    TCuda1DPtr<int> xpSampleId, TCuda2DPtr<TFloat> dst, TCuda2DPtr<TFloat> dstT)
{
    CUDA_ASSERT(blockDim.y == MOE_GROUP_DATA_WARPS);
    int tile = blockIdx.x;
    int offset = tile * MM_TILE;
    int blk = blockIdx.y * MM_TILE;
    if (blk < expertOffsetArr[expertSpan.Beg] || blk >= expertOffsetArr[expertSpan.Fin]) {
        return;
    }

    int warpId = threadIdx.y;
    int thrId = threadIdx.x + threadIdx.y * WARP_SIZE;

    extern __shared__ char shmem[];
    TMoeGroupSamplesData<TFloat> &data = *(TMoeGroupSamplesData<TFloat> *)shmem;
    if (thrId < MM_TILE) {
        int tt = thrId;
        int t = blk + tt;
        data.SampleId[tt] = xpSampleId[t];
    }
    __syncthreads();

    constexpr int SAMPLES_PER_WARP = MM_TILE / MOE_GROUP_DATA_WARPS;
    TVec4<TFloat> vecArr[SAMPLES_PER_WARP];
    // batch load
    for (int k = 0; k < SAMPLES_PER_WARP; ++k) {
        int tt = warpId * SAMPLES_PER_WARP + k;
        int idx = data.SampleId[tt];
        if (idx >= 0) {
            vecArr[k] = LoadPackedVec4<TFloat>(src[idx] + offset);
        } else {
            vecArr[k] = ZeroPackedVec4<TFloat>();
        }
    }

    // batch store
    for (int k = 0; k < SAMPLES_PER_WARP; ++k) {
        int tt = warpId * SAMPLES_PER_WARP + k;
        int t = blk + tt;
        StorePackedVec4<TFloat>(dst[t] + offset, vecArr[k]);
    }

    // store transposed if needed
    if (saveTransposed) {
        __syncthreads();
        for (int k = 0; k < SAMPLES_PER_WARP; ++k) {
            int tt = warpId * SAMPLES_PER_WARP + k;
            data.Buf.StoreVecTransposed(tt, vecArr[k]);
        }
        __syncthreads();
        data.Buf.CopyToGmem(dstT, blk, offset);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class TRowProcess, class TResFloat>
__global__ void CombineExperts(TCudaSpan expertSpan, TCuda2DPtr<half> contractResult, int selectedCount, TCuda1DPtr<int> sampleExpertIds,
    TCuda1DPtr<int> sampleExpertOffset, TCuda1DPtr<int> expertOffsetArr, TCuda2DPtr<TResFloat> resBuf, typename TRowProcess::TParams params)
{
    // to speed up process several rows in single warp
    __shared__ typename TRowProcess::TShmem<TRowProcess> shmem;
    __shared__ float vec[MM_TILE];

    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;

    for (int warpId = 0; warpId < 4; ++warpId) {
        TRowProcess::Init(params, shmem.WgShmem[0], warpId * WARP_SIZE + h, offset, t);
    }
    float globalScale = TRowProcess::GetScale(params);
    __syncwarp();

    float4 sum = ZeroWarpVec();
    for (int k = 0; k < selectedCount; ++k) {
        int srcIdx = t * selectedCount + k;
        int expertId = sampleExpertIds[srcIdx];
        if (expertId >= expertSpan.Beg && expertId < expertSpan.Fin) {
            int row = sampleExpertOffset[srcIdx] + expertOffsetArr[expertId];
            sum = sum + LoadWarpVec(contractResult[row] + offset);
        }
    }
    StoreWarpVec(vec, Scale(sum, globalScale));
    __syncwarp();
    TRowProcess::StoreRow(params, shmem.WgShmem[0], tile, tile * MM_TILE, t, vec, resBuf);
}
}
