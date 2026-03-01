#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "vec_util.cuh"


namespace NCuda
{
constexpr int MM_TILE = 128;

template <class T>
auto MMTiles(T &&sz)
{
    return DivCeil(sz, MM_TILE);
}


///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////
// use in EnumArgs() below
template <class TMatMulCtx, class TShmem>
struct TCallComputeMatMul
{
    TMatMulCtx &Ctx;
    TShmem &Data;

    __device__ TCallComputeMatMul(TMatMulCtx &ctx, TShmem &data) : Ctx(ctx), Data(data) {}
    template <int TRANSPOSE_A, int TRANSPOSE_B, class T1, class T2>
    __device__ void ComputeMatMul(T1 &a, int ax, int ay, T2 &b, int bx, int by, int kSize)
    {
        Ctx.ComputeMatMul<TRANSPOSE_A, TRANSPOSE_B>(Data, a, ax, ay, b, bx, by, kSize);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct TMakeKernelParam
{
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// root matmul arg list
struct TMatMulArgsRoot
{
    template <class TElem>
    bool TypeMatch(TElem *) const
    {
        return true;
    }
};

inline TMatMulArgsRoot MultiArg()
{
    return TMatMulArgsRoot();
}

struct TMatMulKernelCallRoot
{
    template <class TFunc>
    __device__ void EnumArgs(TFunc func, int x, int y) const
    {
    }
};

template <>
struct TMakeKernelParam<TMatMulArgsRoot>
{
    template <class TAdapter>
    using Result = TMatMulKernelCallRoot;
};

template <class TAdapter>
void PassKernelParams(TAdapter &ad, TKernelOp &op, const TMatMulArgsRoot &arg)
{
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// matmul arg list
template <class TPrev, class TArgs>
struct TMatMulArgsList
{
    typedef TMatMulArgsList<TPrev, TArgs> TSelf;
    TPrev Prev;
    TArgs Args;

    TMatMulArgsList(const TPrev &prev, const TArgs &args) : Prev(prev), Args(args) {}
    template <class TElem>
    bool TypeMatch(TElem *p) const {
        return Prev.TypeMatch(p) && Args.TypeMatch(p);
    }
    template <class TNextArgs>
    TMatMulArgsList<TSelf, TNextArgs> operator+(const TNextArgs &args)
    {
        return TMatMulArgsList<TSelf, TNextArgs>(*this, args);
    }
};


template <class TArgs>
inline TMatMulArgsList<TMatMulArgsRoot, TArgs> operator+(const TMatMulArgsRoot &prev, const TArgs &args)
{
    return TMatMulArgsList<TMatMulArgsRoot, TArgs>(prev, args);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// regular matmul
template <class T1, class T2, class TKSize, int TRANSPOSE_A, int TRANSPOSE_B>
struct TMatMulArgs
{
    void *Owner;
    T1 &A;
    T2 &B;
    TKSize KSize;

    TMatMulArgs(void *owner, T1 &a, T2 &b, TKSize kSize) : Owner(owner), A(a), B(b), KSize(kSize) {}
    template <class TElem>
    bool TypeMatch(TElem *p) const
    {
        return typeid(typename T1::TElem) == typeid(TElem) && typeid(typename T2::TElem) == typeid(TElem);
    }
};


template <int TRANSPOSE_A, int TRANSPOSE_B, class T1, class T2, class TKSize>
inline TMatMulArgs<T1, T2, TKSize, TRANSPOSE_A, TRANSPOSE_B> MakeMatMulArgs(void *owner, T1 &a, T2 &b, TKSize kSize)
{
    return TMatMulArgs<T1, T2, TKSize, TRANSPOSE_A, TRANSPOSE_B>(owner, a, b, kSize);
}


template <class TAdapter, class TPrev, class T1, class T2, int TRANSPOSE_A, int TRANSPOSE_B>
struct TMatMulKernelCall : public TPrev
{
    typename TAdapter::TMatrixArg<T1> A;
    typename TAdapter::TMatrixArg<T2> B;
    int KSize;

    template <class TFunc>
    __device__ void EnumArgs(TFunc func, int x, int y) const
    {
        TPrev::EnumArgs(func, x, y);
        func.ComputeMatMul<TRANSPOSE_A, TRANSPOSE_B>(A, 0, y, B, 0, x, KSize);
    }
};


template <class TPrev, class T1, class T2, class TKSize, int TRANSPOSE_A, int TRANSPOSE_B>
struct TMakeKernelParam<TMatMulArgsList<TPrev, TMatMulArgs<T1, T2, TKSize, TRANSPOSE_A, TRANSPOSE_B>>>
{
    template <class TAdapter>
    using Result = TMatMulKernelCall<TAdapter, typename TMakeKernelParam<TPrev>::Result<TAdapter>, T1, T2, TRANSPOSE_A, TRANSPOSE_B>;
};


template <class TAdapter, class TPrev, class T1, class T2, class TKSize, int TRANSPOSE_A, int TRANSPOSE_B>
inline void PassKernelParams(
    TAdapter &ad, TKernelOp &op, const TMatMulArgsList<TPrev, TMatMulArgs<T1, T2, TKSize, TRANSPOSE_A, TRANSPOSE_B>> &argList)
{
    PassKernelParams(ad, op, argList.Prev);
    auto &args = argList.Args;
    ad.PassMatrixA(op, args.A);
    ad.PassMatrixB(op, args.B);
    op.Read(args.KSize);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TPassMatMulKernelArguments
{
    template <class T>
    using TMatrixArg = TCuda2DPtr<typename T::TElem>;

    template <class T>
    void PassMatrixA(TKernelOp &op, T &matr)
    {
        op.Read(matr);
    }
    template <class T>
    void PassMatrixB(TKernelOp &op, T &matr)
    {
        op.Read(matr);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TMatMulWarpResult
{
    TRegTile<T> Sum[4][4];

    __device__ void Clear()
    {
        for (int ty = 0; ty < 4; ++ty) {
            for (int tx = 0; tx < 4; ++tx) {
                Sum[ty][tx].Clear();
            }
        }
    }
};


template <class T>
struct TMatMulWarpGroupResult
{
    TRegTile<T> Sum00[4];
    TRegTile<T> Sum01[4];
    TRegTile<T> Sum10[4];
    TRegTile<T> Sum11[4];

    __device__ void Clear()
    {
        for (int k = 0; k < 4; ++k) {
            Sum00[k].Clear();
            Sum01[k].Clear();
            Sum10[k].Clear();
            Sum11[k].Clear();
        }
    }
    template <class TStoreRows>
    __device__ void StoreHalf0(const TTileCoord &tc, int yHalf, int warpId, TStoreRows &storeRows, float sumScale) const
    {
        for (int k = 0; k < 4; ++k) {
            Sum00[k].StoreHalfScaled(tc, yHalf, storeRows.GetNewBuf(warpId, k * TILE), sumScale);
            Sum01[k].StoreHalfScaled(tc, yHalf, storeRows.GetNewBuf(warpId, k * TILE + 64) , sumScale);
        }
    }
    template <class TStoreRows>
    __device__ void StoreHalf1(const TTileCoord &tc, int yHalf, int warpId, TStoreRows &storeRows, float sumScale) const
    {
        for (int k = 0; k < 4; ++k) {
            Sum10[k].StoreHalfScaled(tc, yHalf, storeRows.GetNewBuf(warpId, k * TILE), sumScale);
            Sum11[k].StoreHalfScaled(tc, yHalf, storeRows.GetNewBuf(warpId, k * TILE + 64), sumScale);
        }
    }
};


template <class T>
struct TMatMulWarpGroup128Result
{
    TRegTile<T> Sum0[8];
    TRegTile<T> Sum1[8];

    __device__ void Clear()
    {
        for (int k = 0; k < 8; ++k) {
            Sum0[k].Clear();
            Sum1[k].Clear();
        }
    }
    template <class TStoreRows>
    __device__ void StoreHalf0(const TTileCoord &tc, int yHalf, int warpId, TStoreRows &storeRows, float sumScale) const
    {
        for (int k = 0; k < 8; ++k) {
            Sum0[k].StoreHalfScaled(tc, yHalf, storeRows.GetNewBuf(warpId, k * TILE), sumScale);
        }
    }
    template <class TStoreRows>
    __device__ void StoreHalf1(const TTileCoord &tc, int yHalf, int warpId, TStoreRows &storeRows, float sumScale) const
    {
        for (int k = 0; k < 8; ++k) {
            Sum1[k].StoreHalfScaled(tc, yHalf, storeRows.GetNewBuf(warpId, k * TILE), sumScale);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// store result tile from 4 warps
struct TStoreRowBase
{
    struct TParams
    {
    };

    struct TWgShmem
    {
    };

    template <class TRowProcess>
    struct TShmem
    {
        enum {
            MAX_WG_COUNT = 2,
        };

        union {
            float ScaledRes[2][TILE][MM_TILE];
            float NewScaledRes[8][8][MM_TILE];
        };
        typename TRowProcess::TWgShmem WgShmem[MAX_WG_COUNT];

        __device__ TCuda2DPtr<float> GetNewBuf(int warpId, int xOffset)
        {
            return TCuda2DPtr<float>(NewScaledRes[warpId][0] + xOffset, MM_TILE * sizeof(float), TILE, 8);
        }
    };

    __device__ static float GetScale(TParams &params) { return 1; }

    template <class TParams>
    __device__ static void Init(TParams &params, TWgShmem &shmem, int wgThreadId, int dstX, int dstY)
    {
    }
};


template <class TRowProcess>
struct TStoreMatMulWarpResult
{
    typedef typename TRowProcess::TParams TParams;
    typedef typename TRowProcess::TShmem<TRowProcess> TShmem;

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, const TMatMulWarpResult<T> &mmRes, float resultScale,
        int offsetX, int offsetY, int mmTileX, int mmTileY, TCuda2DPtr<TRes> resBuf)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);
        int tile = mmTileX / MM_TILE;
        int h = threadIdx.x;
        int warpId = threadIdx.y;
        float sumScale = TRowProcess::GetScale(params) * resultScale;

        __syncthreads();
        TRowProcess::Init(params, shmem.WgShmem[0], warpId * WARP_SIZE + h, mmTileX, mmTileY);

        // process 2 rows at at time
        for (int ty = 0; ty < 4; ++ty) {
            __syncthreads();
            for (int tx = 0; tx < 4; ++tx) {
                mmRes.Sum[ty][tx].StoreScaled(tc, TCuda2DPtr<float>(&shmem.ScaledRes[offsetY / 64][0][offsetX + tx * TILE], MM_TILE * sizeof(float), 16, 16), sumScale);
            }
            __syncthreads();
            for (int base = 0; base < TILE; base += 2) {
                int rowId = base + (offsetX / 64);
                int dstY = mmTileY + offsetY + ty * TILE + rowId;
                TRowProcess::StoreRow(params, shmem.WgShmem[0], tile, mmTileX, dstY, shmem.ScaledRes[offsetY / 64][rowId], resBuf);
            }
        }
    }

    template <class TRes, class TMatMulResult>
    __device__ static void StoreWG(TParams &params, TShmem &shmem, const TTileCoord &tc, int wgCount, const TMatMulResult &mmRes,
        float resultScale, int mmTileX, int mmTileY, TCuda2DPtr<TRes> resBuf)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);
        int h = threadIdx.x;
        int warpId = threadIdx.y;
        int wgId = warpId / 4;
        int wgOffset = warpId & 3;
        mmTileX += wgId * MM_TILE;
        int tile = mmTileX / MM_TILE;
        float sumScale = TRowProcess::GetScale(params) * resultScale;
        CUDA_ASSERT(warpId < ARRAY_SIZE(shmem.NewScaledRes));
        CUDA_ASSERT(wgId < TShmem::MAX_WG_COUNT);

        BarSync(8, wgCount * 128);
        TRowProcess::Init(params, shmem.WgShmem[wgId], wgOffset * WARP_SIZE + h, mmTileX, mmTileY);
        BarSync(8, 128 * wgCount);

#pragma unroll
        for (int yHalf = 0; yHalf < 16; yHalf += 8) {
            mmRes.StoreHalf0(tc, yHalf, warpId, shmem, sumScale);
            __syncwarp();
            for (int y = 0; y < 8; ++y) {
                int dstX = mmTileX;
                int dstY = mmTileY + wgOffset * TILE + yHalf + y;
                TRowProcess::StoreRow(params, shmem.WgShmem[wgId], tile, dstX, dstY, shmem.NewScaledRes[warpId][y], resBuf);
            }
            __syncwarp();
        }
#pragma unroll
        for (int yHalf = 0; yHalf < 16; yHalf += 8) {
            mmRes.StoreHalf1(tc, yHalf, warpId, shmem, sumScale);
            __syncwarp();
            for (int y = 0; y < 8; ++y) {
                int dstX = mmTileX;
                int dstY = mmTileY + wgOffset * TILE + yHalf + y + 64;
                TRowProcess::StoreRow(params, shmem.WgShmem[wgId], tile, dstX, dstY, shmem.NewScaledRes[warpId][y], resBuf);
            }
            __syncwarp();
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class TRes, class TMultiArgs, class TMatMulCtx>
__global__ void GenericMatMulKernel(int mSize, int nSize,
    TCuda2DPtr<TRes> resBuf, __grid_constant__ const TMultiArgs multiArgs, typename TStoreFunc::TParams params)
{
    typedef typename TMatMulCtx::TShmem TShmem;
    TMatMulCtx ctx;
    __shared__ TShmem data;

    int mmTileX = blockIdx.x * MM_TILE;
    int mmTileY = blockIdx.y * MM_TILE;

    TCallComputeMatMul<TMatMulCtx, TShmem> cc(ctx, data);

    ctx.Clear();
    multiArgs.EnumArgs(cc, mmTileX, mmTileY);

    TTileCoord tc;
    int resFirstRow = blockIdx.z * mSize;
    ctx.SaveResult(tc, data, resBuf, mmTileX, mmTileY + resFirstRow, params);
}
KERNEL_BLOCK_SIZE(GenericMatMulKernel, WARP_SIZE, 4);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreScaledBase : public TStoreRowBase
{
    struct TParams
    {
        float *ScalePtr;
        float Scale;
    };

    __device__ static float GetScale(TParams &params)
    {
        float res = params.Scale;
        if (params.ScalePtr) {
            res *= *params.ScalePtr;
        }
        return res;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStore : public TStoreRowBase
{
    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 vec = LoadWarpVecSmem(mmRes);
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreAdd : public TStoreRowBase
{
    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 vec = LoadWarpVecSmem(mmRes);
        vec = vec + LoadWarpVec(resBuf[dstY] + dstX);
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreScaled : public TStoreScaledBase
{
    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 vec = LoadWarpVecSmem(mmRes);
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStoreAddScaled : public TStoreScaledBase
{
    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 vec = LoadWarpVecSmem(mmRes);
        vec = vec + LoadWarpVec(resBuf[dstY] + dstX);
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// store normalized result
struct TStoreRowTileNormalize : public TStoreRowBase
{
    struct TParams
    {
        float Scale;
        float VecScale; // for normalization
        TCuda2DPtr<float> ScaleBuf;
    };

    __device__ static float GetScale(TParams &params)
    {
        return params.Scale;
    }

    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 vec = LoadWarpVecSmem(mmRes);
        float sum2 = CalcWarpVecSum2(vec);
        float discrScale = 0;
        if (sum2 > 0) {
            discrScale = sqrt(sum2 / MM_TILE) * params.VecScale;
            vec = Scale(vec, 1 / discrScale);
        }
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        if (threadIdx.x == 0) {
            params.ScaleBuf[tile][dstY] = discrScale;
        }
        resBuf.CheckCoords(dstX, dstY);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// store max normalized result
struct TStoreRowTileMaxNormalize : public TStoreRowBase
{
    struct TParams
    {
        float *ScalePtr;
        TCuda2DPtr<float> TileScale;
    };

    __device__ static float GetScale(TParams &params)
    {
        return params.ScalePtr ? *params.ScalePtr : 1;
    }

    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        float4 vec = LoadWarpVecSmem(mmRes);
        float maxVal = CalcWarpVecMaxAbsValue(vec);
        float discrScale = 0;
        if (maxVal > 0) {
            discrScale = GetMaxDiscrScale(maxVal, (TRes *)0);
            vec = Scale(vec, 1 / discrScale);
        }
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        if (threadIdx.x == 0) {
            params.TileScale[tile][dstY] = discrScale;
        }
        resBuf.CheckCoords(dstX, dstY);
    }
};

}
