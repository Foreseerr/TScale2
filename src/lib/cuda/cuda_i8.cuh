#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "cuda_matmul.cuh"
#include "cuda_sort.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int I8_TRANSPOSE_WARPS = 8;

inline __device__ void TransposeKernelImpl(int xBlock, int yBlock, TCuda2DPtr<i8> src, TCuda2DPtr<i8> dst)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    __shared__ i8 buf[128][128];

    int h = threadIdx.x;
    int warpId = threadIdx.y;
    for (int yBase = 0; yBase < 128; yBase += I8_TRANSPOSE_WARPS) {
        int y = yBase + warpId;
        int xOffset = h * 4;
        int xorAddr = y & ~3;
        int *pSrc = (int *)&src[yBlock + y][xBlock + xOffset];
        int *pDst = (int *)&buf[y][xOffset ^ xorAddr];
        *pDst = *pSrc;
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += I8_TRANSPOSE_WARPS) {
        int y = yBase + warpId;
        int xOffset = h * 4;
        union {
            int column;
            i8 columnBytes[4];
        };
        for (int k = 0; k < 4; ++k) {
            int readX = y;
            int readY = xOffset + k;
            int xorAddr = readY & ~3;
            columnBytes[k] = buf[readY][readX ^ xorAddr];
        }
        int *pDst = (int *)&dst[xBlock + y][yBlock + xOffset];
        *pDst = column;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void TransposeI8Matrix(TCuda2DPtr<i8> src, TCuda2DPtr<i8> dst);
KERNEL_BLOCK_SIZE(TransposeI8Matrix, WARP_SIZE, I8_TRANSPOSE_WARPS);

// xTiles & yTiles are src dimensions
template <class TXSize, class TYSize>
void Transpose(TPtrArg<TGraph> c, const TCuda2DArray<i8> &src, TXSize &&xSize, TYSize &&ySize, TCuda2DArray<i8> *pDst)
{
    CudaCall(c, TransposeI8Matrix).Grid(MMTiles(xSize), MMTiles(ySize)).Read(src).Write(pDst);
}

template <class TXSize, class TYSize>
void Transpose(TPtrArg<TGraph> c, const TCuda2DArray<e4m3> &src, TXSize &&xSize, TYSize &&ySize, TCuda2DArray<e4m3> *pDst)
{
    CudaCall(c, TransposeI8Matrix).Grid(MMTiles(xSize), MMTiles(ySize)).Read(src).Write(pDst);
}

template <class TXSize, class TYSize>
void Transpose(TPtrArg<TGraph> c, const TCuda2DArray<e5m2> &src, TXSize &&xSize, TYSize &&ySize, TCuda2DArray<e5m2> *pDst)
{
    CudaCall(c, TransposeI8Matrix).Grid(MMTiles(xSize), MMTiles(ySize)).Read(src).Write(pDst);
}

template <class TXSize, class TYSize, class T1, class T2>
void Transpose(TPtrArg<TGraph> c, const TCuda2DArray<T1> &src, TXSize &&xSize, TYSize &&ySize, TCuda2DArray<T2> *pDst)
{
    Y_VERIFY(0 && "unsupported");
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreData>
struct TInt8MatMulData
{
    union {
        struct {
            T8SMemI8Tile aFrag[8];
            T8SMemI8Tile bFrag[8];
        };
        float RowScaleArr[128];
        TStoreData StoreData;
    };
};

struct TInt8MatMulCtx
{
    union {
        TMatMulWarpResult<int> Res;
        TMatMulWarpResult<float> ResScaled;
    };
    int aWarpOffset;
    int bWarpOffset;
    int aStride;
    int bStride;
    i8 *aPtr;
    i8 *bPtr;

    __device__ TInt8MatMulCtx(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, int warpId)
    {
        Res.Clear();
        aStride = aMatr.GetStrideInBytes();
        bStride = bMatr.GetStrideInBytes();
        aPtr = aMatr[blockIdx.y * MM_TILE];
        bPtr = bMatr[blockIdx.x * MM_TILE];

        aWarpOffset = (warpId & 1) * 4;
        bWarpOffset = (warpId >> 1) * 4;
    }

    template <class TStoreData>
    __device__ void LoadData(TInt8MatMulData<TStoreData> *pData, int warpId)
    {
        Copy8TileArray(pData->aFrag, warpId, TCuda2DPtr<i8>(aPtr, aStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8);
        Copy8TileArray(pData->bFrag, warpId, TCuda2DPtr<i8>(bPtr, bStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8);
    }

    template <class TStoreData>
    __device__ void Add(TInt8MatMulData<TStoreData> *pData, int warpId)
    {
        for (int k = 0; k < 8; ++k) {
            TRegTile<i8> b[4];
            b[0] = LoadTile(pData->bFrag[bWarpOffset + 0], k);
            b[1] = LoadTile(pData->bFrag[bWarpOffset + 1], k);
            b[2] = LoadTile(pData->bFrag[bWarpOffset + 2], k);
            b[3] = LoadTile(pData->bFrag[bWarpOffset + 3], k);
            for (int ty = 0; ty < 4; ++ty) {
                TRegTile<i8> a;
                a = LoadTile(pData->aFrag[aWarpOffset + ty], k);
                for (int tx = 0; tx < 4; ++tx) {
                    MMA(&Res.Sum[ty][tx], a, b[tx]);
                }
            }
        }
    }

    __device__ void NextTile()
    {
        aPtr += I8_TILE_GROUP_SIZE;
        bPtr += I8_TILE_GROUP_SIZE;
    }

    __device__ int GetResultOffsetX() { return bWarpOffset * TILE; }
    __device__ int GetResultOffsetY() { return aWarpOffset * TILE; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class TRes>
__global__ void Int8MatMulKernel(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, int kSize, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);

    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int warpId = threadIdx.y;

    TInt8MatMulCtx ctx(aMatr, bMatr, warpId);
    __shared__ TInt8MatMulData<typename TStoreFunc::TShmem<TStoreFunc>> data;

    for (int k = 0; k < kSize; k += 128) {
        __syncthreads();
        ctx.LoadData(&data, warpId);
        __syncthreads();
        ctx.Add(&data, warpId);
        ctx.NextTile();
    }
    // save result
    typedef TStoreMatMulWarpResult<TStoreFunc> TStoreResult;
    int mmTileX = blockIdx.x * MM_TILE;
    int mmTileY = blockIdx.y * MM_TILE;
    TStoreResult::Store(params, data.StoreData, tc, ctx.Res, 1.0f, ctx.GetResultOffsetX(), ctx.GetResultOffsetY(), mmTileX, mmTileY, resBuf);
}
KERNEL_BLOCK_SIZE(Int8MatMulKernel, WARP_SIZE, 4);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Int8MatMul(TPtrArg<TGraph> c, const T1 &aMatr, const T2 &bMatr, TRes *pResMatr, TXSize &&xSize, TYSize &&ySize, TZSize &&zSize)
{
    return CudaCall(c, Int8MatMulKernel<TStoreFunc, typename TRes::TElem>)
        .Grid(MMTiles(zSize), MMTiles(xSize))
        .Read(aMatr, bMatr, ySize)
        .Write(pResMatr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class TRes>
__global__ void Int8MatMulKernelRowScale(TCuda2DPtr<i8> aMatr, TCuda1DPtr<float> aRowScale, TCuda2DPtr<i8> bMatr, int kSize, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);

    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int h = threadIdx.x;
    int warpId = threadIdx.y;

    TInt8MatMulCtx ctx(aMatr, bMatr, warpId);
    __shared__ TInt8MatMulData<typename TStoreFunc::TShmem<TStoreFunc>> data;

    for (int k = 0; k < kSize; k += 128) {
        __syncthreads();
        ctx.LoadData(&data, warpId);
        __syncthreads();
        ctx.Add(&data, warpId);
        ctx.NextTile();
    }
    __syncthreads();
    {
        int y = warpId * WARP_SIZE + h;
        data.RowScaleArr[y] = aRowScale[blockIdx.y * MM_TILE + y];
    }
    __syncthreads();

    // inplace rescale result
    for (int ty = 0; ty < 4; ++ty) {
        TRegTileRow<float> scale;
        scale.Load(tc, data.RowScaleArr + (ctx.aWarpOffset + ty) * TILE);
        for (int tx = 0; tx < 4; ++tx) {
            TRegTile<float> &sumScaled = ctx.ResScaled.Sum[ty][tx];
            const TRegTile<int> &sum = ctx.Res.Sum[ty][tx];
            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                sumScaled.x[elem] = sum.x[elem] * scale.x[rowIndex];
                });
        }
    }
    // save scaled result
    typedef TStoreMatMulWarpResult<TStoreFunc> TStoreResult;
    int mmTileX = blockIdx.x * MM_TILE;
    int mmTileY = blockIdx.y * MM_TILE;
    TStoreResult::Store(params, data.StoreData, tc, ctx.ResScaled, 1.0f, ctx.GetResultOffsetX(), ctx.GetResultOffsetY(), mmTileX, mmTileY, resBuf);
}
KERNEL_BLOCK_SIZE(Int8MatMulKernelRowScale, WARP_SIZE, 4);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Int8MatMulRowScale(TPtrArg<TGraph> c, const T1 &aMatr, TCudaVector<float> &aRowScale, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xSize, TYSize &&ySize, TZSize &&zSize)
{
    return CudaCall(c, Int8MatMulKernelRowScale<TStoreFunc, typename TRes::TElem>)
        .Grid(MMTiles(zSize), MMTiles(xSize))
        .Read(aMatr, aRowScale, bMatr, ySize)
        .Write(pResMatr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class TRes>
__global__ void Int8MatMulYScaleKernel(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, TCuda1DPtr<float> yTileScale, int kSize, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);

    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int warpId = threadIdx.y;

    TInt8MatMulCtx ctx(aMatr, bMatr, warpId);
    __shared__ TInt8MatMulData<typename TStoreFunc::TShmem<TStoreFunc>> data;

    float prevTileScale = (kSize > 0) ? yTileScale[0] : 0;
    for (int k = 0; k < kSize; k += 128) {
        __syncthreads();
        {
            float scale = yTileScale[k / 128];
            if (scale == 0) {
                continue;
            }
            float mult = prevTileScale / scale;
            if (mult != 1) {
                for (int ty = 0; ty < 4; ++ty) {
                    for (int tx = 0; tx < 4; ++tx) {
                        ctx.Res.Sum[ty][tx].Scale(mult);
                    }
                }
            }
            prevTileScale = scale;
        }
        ctx.LoadData(&data, warpId);
        __syncthreads();
        ctx.Add(&data, warpId);
        ctx.NextTile();
    }
    // save result
    typedef TStoreMatMulWarpResult<TStoreFunc> TStoreResult;
    int mmTileX = blockIdx.x * MM_TILE;
    int mmTileY = blockIdx.y * MM_TILE;
    TStoreResult::Store(params, data.StoreData, tc, ctx.Res, prevTileScale, ctx.GetResultOffsetX(), ctx.GetResultOffsetY(), mmTileX, mmTileY, resBuf);
}
KERNEL_BLOCK_SIZE(Int8MatMulYScaleKernel, WARP_SIZE, 4);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Int8MatMulYScale(TPtrArg<TGraph> c, const T1 &aMatr, const T2 &bMatr, TCudaVector<float> &yTileScale, TRes *pResMatr,
    TXSize &&xSize, TYSize &&ySize, TZSize &&zSize)
{
    return CudaCall(c, Int8MatMulYScaleKernel<TStoreFunc, typename TRes::TElem>)
        .Grid(MMTiles(zSize), MMTiles(xSize))
        .Read(aMatr, bMatr, yTileScale, ySize)
        .Write(pResMatr);
}

}
