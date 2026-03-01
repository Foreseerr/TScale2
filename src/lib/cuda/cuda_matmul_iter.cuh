#pragma once
#include "cuda_arrays.h"
#include "cuda_matmul.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int XBlockSize, int YBlockSize, class TStoreFunc>
class TMatMulTileIterator
{
    int XSize;
    int YSize;
    int XBlockCount;
    int Ptr;
    int PtrBase;
    int Y;
    int X;

public:
    __device__ TMatMulTileIterator(int mSize, int nSize) : XSize(nSize), YSize(mSize)
    {
        XBlockCount = (XSize + XBlockSize - 1) / XBlockSize;
        Ptr = blockIdx.x;
        PtrBase = 0;
        Y = 0;
    }

    __device__ bool Next()
    {
        while (Y < YSize) {
            X = (Ptr - PtrBase) * XBlockSize;
            if (X < XSize) {
                Ptr += gridDim.x;
                return true;
            }
            PtrBase += XBlockCount;
            Y += YBlockSize;
        }
        return false;
    }

    template <class TSumBuf, class TRes>
    __device__ void Store(int wgId, int wgCount, TSumBuf &sumBuf, typename TStoreFunc::TParams &params,
        typename TStoreFunc::TShmem<TStoreFunc> &storeData, TCuda2DPtr<TRes> &resBuf, int startResRow) const
    {
        TTileCoord tc;
        typedef TStoreMatMulWarpResult<TStoreFunc> TStoreResult;
        int xSizeRoundUp = (XSize + MM_TILE - 1) & ~(MM_TILE - 1);
        // one tile per warp group
        int storeWGCount = min(wgCount, (xSizeRoundUp - X) / MM_TILE);
        if (wgId < storeWGCount) {
            TStoreResult::StoreWG(params, storeData, tc, storeWGCount, sumBuf, 1.0f, X, Y + startResRow, resBuf);
        }
        BarSync(0, wgCount * 128);
    }

    __device__ int GetX() const { return X; }
    __device__ int GetY() const { return Y; }
};

}
