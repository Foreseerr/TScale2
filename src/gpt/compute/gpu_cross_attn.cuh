#pragma once
#include <lib/cuda/cuda_arrays.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void CrossAttentionForward(int len, TCuda2DPtr<T> src, TCuda1DPtr<int> fwdShuffle, TCuda2DPtr<T> dst)
{
    int xTile = blockIdx.x;
    int offset = xTile * MM_TILE;
    int tTile = blockIdx.y;
    int warpId = threadIdx.y;
    constexpr int SZ = MM_TILE / MAX_WARPS;

    float4 data[SZ];
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        if (t < len) {
            int srcRow = fwdShuffle[t];
            data[k] = LoadWarpVec(src[srcRow] + offset);
        } else {
            data[k] = ZeroWarpVec();
        }
    }
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        StoreWarpVec(dst[t] + offset, data[k]);
    }
}
KERNEL_BLOCK_SIZE(CrossAttentionForward, WARP_SIZE, MAX_WARPS);


template <class T>
__global__ void CrossAttentionBackward(int len, TCuda2DPtr<T> src, TCuda1DPtr<int> fwdShuffle, TCuda2DPtr<T> dst)
{
    int xTile = blockIdx.x;
    int offset = xTile * MM_TILE;
    int tTile = blockIdx.y;
    int warpId = threadIdx.y;
    constexpr int SZ = MM_TILE / MAX_WARPS;

    float4 data[SZ];
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        if (t < len) {
            data[k] = LoadWarpVec(src[t] + offset);
        }
    }
    for (int k = 0; k < SZ; ++k) {
        int t = tTile * MM_TILE + warpId * SZ + k;
        if (t < len) {
            int dstRow = fwdShuffle[t];
            StoreWarpVec(dst[dstRow] + offset, data[k]);
        } else {
            StoreWarpVec(dst[t] + offset, ZeroWarpVec());
        }
    }
}
KERNEL_BLOCK_SIZE(CrossAttentionBackward, WARP_SIZE, MAX_WARPS);
}
