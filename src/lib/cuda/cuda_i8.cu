#include <util/pch.h>
#define KERNEL_UNIT "cuda_i8/"
#include "cuda_i8.cuh"
#include "cuda_graph.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


namespace NCuda
{
__global__ void TransposeI8Matrix(TCuda2DPtr<i8> src, TCuda2DPtr<i8> dst)
{
    int xBlock = blockIdx.x * MM_TILE;
    int yBlock = blockIdx.y * MM_TILE;
    TransposeKernelImpl(xBlock, yBlock, src, dst);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// copy 16x128 blocks
template <int N_BLOCKS>
struct TAsyncCopy16x128
{
    int SrcOffset;
    int DstOffset;
    int RowOffset;

    __device__ void Init(int srcDataStride, int warpId)
    {
        int blockId = warpId / 4;
        int blockWarp = warpId & 3;
        int h = threadIdx.x;
        int x = h & 7;
        int y = (h / 8) + blockWarp * 4;
        int y7 = y & 7;
        SrcOffset = y * srcDataStride + x * 16;
        DstOffset = ((y * 8 + x) ^ y7) * 16;
        RowOffset = srcDataStride * TILE;
        SrcOffset += RowOffset * blockId;
        DstOffset += sizeof(T8SMemI8Tile) * blockId;
    }
    __device__ void Copy(void *srcPtr, void *dstPtr, int rowCount)
    {
        ui8 *src = (((ui8*)srcPtr) + SrcOffset);
        ui8 *dst = (((ui8*)dstPtr) + DstOffset);
        for (int k = 0; k < rowCount / N_BLOCKS; ++k) {
            AsyncCopy16(dst, src);
            src = AdvancePtr(src, RowOffset * N_BLOCKS);
            dst = AdvancePtr(dst, sizeof(T8SMemI8Tile) * N_BLOCKS);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TRowProcess>
struct TStoreMatMulRowResult
{
    typedef typename TRowProcess::TParams TParams;
    typedef typename TRowProcess::TShmem<TRowProcess> TShmem;

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, int warpId, const TRegTile<T> *mmRes,
        float resultScale, int tile, int mmTileX, int mmTileY, TCuda2DPtr<TRes> resBuf)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);
        float sumScale = TRowProcess::GetScale(params) * resultScale;
        int h = threadIdx.x;

        __syncthreads();
        if (warpId < 4) {
            TRowProcess::Init(params, shmem.WgShmem[0], warpId * WARP_SIZE + h, mmTileX, mmTileY);
        }
        __syncthreads();

        for (int yHalf = 0; yHalf < TILE; yHalf += 8) {
            __syncwarp();
            for (int k = 0; k < MM_TILE / TILE; ++k) {
                mmRes[k].StoreHalfScaled(tc, yHalf, shmem.GetNewBuf(warpId, k * TILE), sumScale);
            }
            __syncwarp();
            for (int y = 0; y < 8; ++y) {
                yint dstY = mmTileY + warpId * TILE + yHalf + y;
                TRowProcess::StoreRow(params, shmem.WgShmem[0], tile, mmTileX, dstY, shmem.NewScaledRes[warpId][y], resBuf);
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline __device__ void LoadDoubleTile(TRegTile<i8> *p1, TRegTile<i8> *p2, const T8SMemI8Tile &ht, int tileId)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int ty = h & 15;
    int rowAddr = ty * 8 + tileId + (h / 16);
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(p1->nx[0]), "=r"(p1->nx[1]), "=r"(p2->nx[0]), "=r"(p2->nx[1]) : "r"(sharedAddr));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline __device__ void MMA(TRegTile<int> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1,
    const TRegTile<i8> &b2, const TRegTile<int> &c)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        : "=r"(pD->x[0]), "=r"(pD->x[1]), "=r"(pD->x[2]), "=r"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        : "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]), "r"(b1.nx[0]), "r"(b2.nx[0]), "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]),
        "r"(c.x[3]));
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        : "=r"(pD->x[4]), "=r"(pD->x[5]), "=r"(pD->x[6]), "=r"(pD->x[7])
        : "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]), "r"(b1.nx[1]), "r"(b2.nx[1]), "r"(c.x[4]), "r"(c.x[5]), "r"(c.x[6]),
        "r"(c.x[7]));
}

__forceinline __device__ void MMA(
    TRegTile<int> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2)
{
    MMA(pD, a1, a2, b1, b2, *pD);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TInt8DoubleRegTile
{
    int nx[4];
};

struct TLoadDoubleRegTileAddr
{
    int Y7;
    int RowAddr;

    __device__ void InitA()
    {
        int h = threadIdx.x;
        Y7 = h & 7;
        int ty = h & 15;
        RowAddr = ty * 8 + (h / 16);
    }
    __device__ void InitB()
    {
        int h = threadIdx.x;
        Y7 = h & 7;
        int ty = Y7 + (h / 16) * 8;
        RowAddr = ty * 8 + ((h / 8) & 1);
    }
};


__forceinline __device__ void LoadDoubleRegTile(const TLoadDoubleRegTileAddr &addr, TInt8DoubleRegTile *p, const T8SMemI8Tile &ht, int tileId)
{
    ui32 sharedAddr = GetSharedAddress(&ht.Data[(addr.RowAddr + tileId) ^ addr.Y7]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[1]), "=r"(p->nx[2]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}


__forceinline __device__ void MMA(TRegTile<int> *pD, const TInt8DoubleRegTile &a, const TInt8DoubleRegTile &b, const TRegTile<int> &c)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        : "=r"(pD->x[0]), "=r"(pD->x[1]), "=r"(pD->x[2]), "=r"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        : "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]), "r"(b.nx[0]), "r"(b.nx[1]), "r"(c.x[0]), "r"(c.x[1]),
        "r"(c.x[2]), "r"(c.x[3]));
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        : "=r"(pD->x[4]), "=r"(pD->x[5]), "=r"(pD->x[6]), "=r"(pD->x[7])
        : "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]), "r"(b.nx[2]), "r"(b.nx[3]), "r"(c.x[4]), "r"(c.x[5]),
        "r"(c.x[6]), "r"(c.x[7]));
}

__forceinline __device__ void MMA(TRegTile<int> *pD, const TInt8DoubleRegTile &a, const TInt8DoubleRegTile &b)
{
    MMA(pD, a, b, *pD);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreData>
struct TInt8MatMulDataExp
{
    enum {
        QSIZE = 2
    };
    union {
        struct {
            T8SMemI8Tile aFrag[QSIZE][8];
            T8SMemI8Tile bFrag[QSIZE][16];
        };
        TStoreData StoreData;
    };
};


struct TInt8MatMulSumCtx
{
    TLoadDoubleRegTileAddr AddrA;
    TLoadDoubleRegTileAddr AddrB;
    TRegTile<int> Sum[16];
    int MyRow;

    __device__ TInt8MatMulSumCtx(int row)
    {
        AddrA.InitA();
        AddrB.InitB();
        for (int k = 0; k < ARRAY_SIZE(Sum); ++k) {
            Sum[k].Clear();
        }
        MyRow = row;
    }

    template <class TStoreData>
    __device__ void Compute(TInt8MatMulDataExp<TStoreData> *pData, int q)
    {
        // // ref
        // for (int k = 0; k < 8; k += 2) {
        //     TRegTile<i8> a1, a2;
        //     LoadDoubleTile(&a1, &a2, pData->aFrag[q][MyRow], k);
        //     for (int n = 0; n < ARRAY_SIZE(Sum); ++n) {
        //         TRegTile<i8> b1, b2;
        //         LoadDoubleTile(&b1, &b2, pData->bFrag[q][n], k);
        //         MMA(&Sum[n], a1, a2, b1, b2);
        //     }
        // }
        // real fast
        TInt8DoubleRegTile a[4];
        for (int k = 0; k < 4; ++k) {
            LoadDoubleRegTile(AddrA, &a[k], pData->aFrag[q][MyRow], k * 2);
        }
        for (int n = 0; n < ARRAY_SIZE(Sum); ++n) {
            for (int k = 0; k < 4; ++k) {
                TInt8DoubleRegTile b;
                LoadDoubleRegTile(AddrB, &b, pData->bFrag[q][n], k * 2);
                MMA(&Sum[n], a[k], b);
            }
        }
    }
};


struct TInt8MatMulLoadCtx
{
    TAsyncCopy16x128<2> CopyA;
    TAsyncCopy16x128<2> CopyB;
    i8 *aPtr;
    i8 *bPtr;

    __device__ TInt8MatMulLoadCtx(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, int warpId)
    {
        CopyA.Init(aMatr.GetStrideInBytes(), warpId);
        CopyB.Init(bMatr.GetStrideInBytes(), warpId);
        aPtr = &aMatr[blockIdx.y * MM_TILE][0];
        bPtr = &bMatr[blockIdx.x * MM_TILE * 2][0];
    }

    template <class TStoreData>
    __device__ void LoadData(TInt8MatMulDataExp<TStoreData> *pData, int q)
    {
        CopyA.Copy(aPtr, pData->aFrag[q], 8);
        CopyB.Copy(bPtr, pData->bFrag[q], 16);
        AsyncCommitGroup();
    }

    __device__ void NextTile()
    {
        aPtr += MM_TILE;
        bPtr += MM_TILE;
    }
};


template <class TStoreFunc, class TRes>
__global__ void
__launch_bounds__(8 * WARP_SIZE, 1)
Int8MatMulKernelExp(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, int yTiles, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);

    TTileCoord tc;

    extern __shared__ char shmem[];
    typedef TInt8MatMulDataExp<typename TStoreFunc::TShmem<TStoreFunc>> TShmem;
    TShmem &data = *(TShmem*)shmem;

    TInt8MatMulLoadCtx loadCtx(aMatr, bMatr, threadIdx.y);
    TInt8MatMulSumCtx sum(threadIdx.y);
    int q = 0;
    loadCtx.LoadData(&data, q);
    AsyncWaitGroup<0>();
#pragma unroll(4)
    for (int yTile = 1; yTile < yTiles; ++yTile) {
        loadCtx.NextTile();
        q ^= 1;
        loadCtx.LoadData(&data, q);
        __syncthreads(); // data is ready on whole sm
        sum.Compute(&data, q ^ 1);
        __syncthreads(); // data is no longer used
        AsyncWaitGroup<0>(); // thread has read new data
    }
    q ^= 1;
    __syncthreads();
    sum.Compute(&data, q ^ 1);
    __syncthreads();

    int tile = blockIdx.x * 2;
    typedef TStoreMatMulRowResult<TStoreFunc> TStoreResult;
    TStoreResult::Store(params, data.StoreData, tc, threadIdx.y, sum.Sum, 1.0f, tile, tile * MM_TILE, blockIdx.y * MM_TILE, resBuf);
    tile += 1;
    TStoreResult::Store(params, data.StoreData, tc, threadIdx.y, sum.Sum + 8, 1.0f, tile, tile * MM_TILE, blockIdx.y * MM_TILE, resBuf);
}
KERNEL_BLOCK_SIZE(Int8MatMulKernelExp, WARP_SIZE, 8);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Int8MatMulExp(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&m, TYSize &&k, TZSize &&n)
{
    return CudaCall(c, Int8MatMulKernelExp<TStoreFunc, typename TRes::TElem>)
        .Grid(DivCeil(n, 2 * MM_TILE), MMTiles(m))
        .Shmem(sizeof(TInt8MatMulDataExp<typename TStoreFunc::TShmem<TStoreFunc>>))
        .Read(aMatr, bMatr, MMTiles(k))
        .Write(pResMatr);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

}


using namespace NCuda;

void TestMatMulInt8()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<i8> aMatr;
    TCuda2DArray<i8> bMatr;
    TCuda2DArray<float> resMatr;
    TCuda2DArray<float> resMatrRef;
    TCudaVector<float> aMatrRowScale;
    TCudaVector<float> yTileScale;

#ifndef NDEBUG
    const int ITER_COUNT = 1;
#else
    const int ITER_COUNT = 100;
#endif

    // int m = 256;
    // int n = 256;
    // int k = 256;
    // higher K for max flops
    // int m = 16 * 1024; // sample count
    // int n = 1024; // combiner width of tt=256
    // int k = 4096; // state dim
    // result fits in L2 cache
    int m = 8 * 1024; // sample count
    int n = 1024;
    int k = 512; // state dim
    aMatr.Allocate(k, m);
    bMatr.Allocate(k, n);
    resMatr.Allocate(n, m);
    resMatrRef.Allocate(n, m);
    aMatrRowScale.AllocateCuda(m);
    aMatrRowScale.ClearDeviceMem(stream);
    yTileScale.Allocate(k / MM_TILE);

    TVector<float> yScale;
    ClearPodArray(&yScale, yTileScale.GetSize());
    for (yint i = 0; i < YSize(yScale); ++i) {
        yScale[i] = i + 1;
        //yScale[i] = 1;
    }
    Put(stream, &yTileScale, yScale);


    // { // cublas
    // #include <cublas_v2.h>
    //     int8_t *A, *B;
    //     int32_t *C;
    //     cudaMalloc(&A, m * k * sizeof(int8_t));
    //     cudaMalloc(&B, k * n * sizeof(int8_t));
    //     cudaMalloc(&C, m * n * sizeof(int32_t));

    //     // cudaMemcpy(A, h_A.data(), m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
    //     // cudaMemcpy(B, h_B.data(), k * n * sizeof(int8_t), cudaMemcpyHostToDevice);

    //     const int32_t alpha = 1;
    //     const int32_t beta = 0;

    //     cublasHandle_t handle;
    //     cublasCreate(&handle);

    //     // Matmul using cublasGemmEx
    //     {
    //         NHPTimer::STime tStart;
    //         NHPTimer::GetTime(&tStart);
    //         for (yint z = 0;; ++z) {
    //             cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_8I, k, B, CUDA_R_8I, k, &beta, C, CUDA_R_32I, m,
    //                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    //         }
    //         cudaDeviceSynchronize();
    //         double tPassed = NHPTimer::GetTimePassed(&tStart);
    //         double tFlops = 2. * ITER_COUNT * m * k * n / tPassed / 1e12;
    //         DebugPrintf("%g TFlops\n", tFlops);
    //     }
    // }

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            //Int8MatMul<TStore>(c, aMatr, bMatr, &resMatr, m, k, n).Struct();
            Int8MatMulExp<TStore>(c, aMatr, bMatr, &resMatrRef, m, k, n).Struct();
            // Int8MatMulRowScale<TStore>(c, aMatr, aMatrRowScale, bMatr, &resMatr, m, k, n).Struct();
            // Int8MatMulYScale<TStore>(c, aMatr, bMatr, yTileScale, &resMatr, m, k, n).Struct();
        }
    }

    FillRandom(rng, stream, &aMatr);
    FillRandom(rng, stream, &bMatr);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = 2. * ITER_COUNT * m * k * n / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops, %g\n", maxTFlops, tFlops);

#ifndef NDEBUG
        resMatr.CopyToHost(stream);
        resMatrRef.CopyToHost(stream);
        stream.Sync();
        TArray2D<float> ra, rb;
        GetAllData(resMatr, &ra);
        GetAllData(resMatrRef, &rb);
        for (yint y = 0; y < ra.GetYSize(); ++y) {
            for (yint x = 0; x < ra.GetXSize(); ++x) {
                Y_ASSERT(ra[y][x] == rb[y][x]);
            }
        }
#endif
    }
}
