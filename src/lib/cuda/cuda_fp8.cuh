#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "cuda_matmul.cuh"
#include "sm90_fp8.cuh"


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline __device__ void MMAe4e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, const TRegTile<float> &c)
{
#if (__CUDA_ARCH__ >= 890)
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[0]), "=f"(pD->x[1]), "=f"(pD->x[2]), "=f"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[0]), "r"(b2.nx[0]),
        "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[4]), "=f"(pD->x[5]), "=f"(pD->x[6]), "=f"(pD->x[7])
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[1]), "r"(b2.nx[1]),
        "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
        );
#else
    printf("sm89 required\n");
#endif
}

__forceinline __device__ void MMAe4e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2)
{
    MMAe4e4(pD, a1, a2, b1, b2, *pD);
}


__forceinline __device__ void MMAe5e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, const TRegTile<float> &c)
{
#if (__CUDA_ARCH__ >= 890)
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[0]), "=f"(pD->x[1]), "=f"(pD->x[2]), "=f"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[0]), "r"(b2.nx[0]),
        "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[4]), "=f"(pD->x[5]), "=f"(pD->x[6]), "=f"(pD->x[7])
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[1]), "r"(b2.nx[1]),
        "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
        );
#else
    printf("sm89 required\n");
#endif
}

__forceinline __device__ void MMAe5e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2)
{
    MMAe5e4(pD, a1, a2, b1, b2, *pD);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TFloatA, class TFloatB>
inline __device__ void MMA(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, TFloatA*, TFloatB*)
{
    CUDA_ASSERT(0 && "unsupported combination");
}

inline __device__ void MMA(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, e4m3 *, e4m3 *)
{
    MMAe4e4(pD, a1, a2, b1, b2);
}

inline __device__ void MMA(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, e5m2 *, e4m3 *)
{
    MMAe5e4(pD, a1, a2, b1, b2);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreData>
struct TFp8MatMulData
{
    union {
        struct
        {
            T8SMemI8Tile aFrag[8];
            T8SMemI8Tile bFrag[8];
        };
        TStoreData StoreData;
    };
};


template <class TStoreFunc>
struct TFp8MatMulCtx
{
    typedef TFp8MatMulData<typename TStoreFunc::TShmem<TStoreFunc>> TShmem;

    TMatMulWarpResult<float> Res;
    int warpId;
    int aWarpOffset;
    int bWarpOffset;

    __device__ TFp8MatMulCtx()
    {
        warpId = threadIdx.y;
        aWarpOffset = (warpId & 1) * 4;
        bWarpOffset = (warpId >> 1) * 4;
    }

    __device__ void Clear() { Res.Clear(); }

    template <int TRANSPOSE_A, int TRANSPOSE_B, class TFloatA, class TFloatB>
    inline __device__ void ComputeMatMul(TShmem &data, TCuda2DPtr<TFloatA> aMatr, int ax, int ay, TCuda2DPtr<TFloatB> bMatr, int bx, int by, int kSize)
    {
        // we ignore passed TRANSPOSE_*, matrices should be transposed beforehand
        // CUDA_ASSERT(TRANSPOSE_A == 0);
        // CUDA_ASSERT(TRANSPOSE_B == 0);
        int aStride = aMatr.GetStrideInBytes();
        int bStride = bMatr.GetStrideInBytes();
        TFloatA *aPtr = aMatr[0] + ay * aStride + ax * sizeof(TFloatA);
        TFloatB *bPtr = bMatr[0] + by * bStride + bx * sizeof(TFloatB);
        for (int k = 0; k < kSize; k += 128) {
            __syncthreads();
            Copy8TileArray(data.aFrag, warpId, TCuda2DPtr<TFloatA>(aPtr, aStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8);
            Copy8TileArray(data.bFrag, warpId, TCuda2DPtr<TFloatB>(bPtr, bStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8);
            __syncthreads();
            for (int k = 0; k < 8; k += 2) {
                // run out of registers if using single loop with preloading b1[4] & b2[4], so split into 2 loops
                TRegTile<i8> b1[2];
                TRegTile<i8> b2[2];
                b1[0] = LoadTile(data.bFrag[bWarpOffset + 0], k);
                b1[1] = LoadTile(data.bFrag[bWarpOffset + 1], k);
                b2[0] = LoadTile(data.bFrag[bWarpOffset + 0], k + 1);
                b2[1] = LoadTile(data.bFrag[bWarpOffset + 1], k + 1);
                for (int ty = 0; ty < 4; ++ty) {
                    TRegTile<i8> a1 = LoadTile(data.aFrag[aWarpOffset + ty], k);
                    TRegTile<i8> a2 = LoadTile(data.aFrag[aWarpOffset + ty], k + 1);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&Res.Sum[ty][tx], a1, a2, b1[tx], b2[tx], aPtr, bPtr);
                    }
                }
                b1[0] = LoadTile(data.bFrag[bWarpOffset + 2], k);
                b1[1] = LoadTile(data.bFrag[bWarpOffset + 3], k);
                b2[0] = LoadTile(data.bFrag[bWarpOffset + 2], k + 1);
                b2[1] = LoadTile(data.bFrag[bWarpOffset + 3], k + 1);
                for (int ty = 0; ty < 4; ++ty) {
                    TRegTile<i8> a1 = LoadTile(data.aFrag[aWarpOffset + ty], k);
                    TRegTile<i8> a2 = LoadTile(data.aFrag[aWarpOffset + ty], k + 1);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&Res.Sum[ty][tx + 2], a1, a2, b1[tx], b2[tx], aPtr, bPtr);
                    }
                }
            }
            // next tile
            aPtr += I8_TILE_GROUP_SIZE;
            bPtr += I8_TILE_GROUP_SIZE;
        }
    }

    template <class TRes>
    inline __device__ void SaveResult(
        const TTileCoord &tc, TShmem &data, TCuda2DPtr<TRes> resBuf, int mmTileX, int mmTileY, typename TStoreFunc::TParams params)
    {
        typedef TStoreMatMulWarpResult<TStoreFunc> TStoreResult;
        int offsetX = bWarpOffset * TILE;
        int offsetY = aWarpOffset * TILE;
        TStoreResult::Store(params, data.StoreData, tc, Res, 1.0f, offsetX, offsetY, mmTileX, mmTileY, resBuf);
    }
};


template <class TStoreFunc, class TRes, class TMSize, class TNSize, class TMultiArgs>
TKernelOp &Fp8MatMulMulti(TPtrArg<TGraph> c, TRes *pResMatr, TMSize &&mSize, TNSize &&nSize, const TMultiArgs &multiArgs)
{
    if (UseSm90Kernels() && multiArgs.TypeMatch((e4m3 *)0)) {
        typedef NSm90Fp8MatMul::TPassMatMulKernelArguments TAdapter;
        typedef typename TMakeKernelParam<TMultiArgs>::Result<TAdapter> TKernelParams;
        using NSm90Fp8MatMul::Sm90Fp8MatMulKernel; // otherwise KERNEL_BLOCK_SIZE() is not picked up
        TAdapter ad;
        TKernelOp &res = CudaCall(c, Sm90Fp8MatMulKernel<TStoreFunc, typename TRes::TElem, TKernelParams>);
        res.FullGrid(c);
        res.Shmem(sizeof(NSm90Fp8MatMul::TMatMulData<typename TStoreFunc::TShmem<TStoreFunc>>));
        res.Read(mSize, nSize).Write(pResMatr).Struct();
        PassKernelParams(ad, res, multiArgs);
        return res;
    } else {
        typedef TPassMatMulKernelArguments TAdapter;
        typedef typename TMakeKernelParam<TMultiArgs>::Result<TAdapter> TKernelParams;
        typedef TFp8MatMulCtx<TStoreFunc> TMatMulCtx;
        TAdapter ad;
        TKernelOp &res = CudaCall(c, GenericMatMulKernel<TStoreFunc, typename TRes::TElem, TKernelParams, TMatMulCtx>);
        res.Grid(MMTiles(nSize), MMTiles(mSize));
        res.Read(mSize, nSize);
        res.Write(pResMatr).Struct();
        PassKernelParams(ad, res, multiArgs);
        return res;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class T1, class T2, class TRes, class TMSize, class TNSize, class TKSize>
TKernelOp &Fp8MatMul(TPtrArg<TGraph> c, const T1 &aMatr, const T2 &bMatr, TRes *pResMatr, TMSize &&mSize, TNSize &&nSize, TKSize &&kSize)
{
    auto mm = MakeMatMulArgs<0, 0>(nullptr, aMatr, bMatr, kSize);
    return Fp8MatMulMulti<TStoreFunc>(c, pResMatr, mSize, nSize, MultiArg() + mm);
}
}
