#include <util/pch.h>
#define KERNEL_UNIT "cuda_fp16/"
#include "cuda_fp16.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


using namespace NCuda;

//__global__ void TestLoadTile(TCuda2DPtr<half> data, TCuda2DPtr<half> dst)
//{
//    TTileCoord tc;
//    __shared__ T4x4SMemHalfTile frag;
//    int warpId = threadIdx.y;
//
//    Copy4x4Tile(&frag, warpId, data[0], data.GetStride());
//    __syncthreads();
//
//    if (warpId == 0) {
//        for (int x = 0; x < 4; ++x) {
//            for (int y = 0; y < 4; ++y) {
//                TRegTile<half> tt;
//                //LoadTile(&tt, frag, x, y);
//                LoadTileTransposed(&tt, frag, x, y);
//                tt.Store(tc, dst[y * 16] + x * 16, dst.GetStride());
//            }
//        }
//    }
//}


// ///////////////////////////////////////////////////////////////////////////////////////////////////
// // test warp vec load bw
// // typedef float TDataType;
// // typedef half TDataType;
// typedef i8 TDataType;
// __global__ void LoadTest(int count, TCuda1DPtr<TDataType> data)
// {
//     float4 sum = ZeroWarpVec();
//     int blk = threadIdx.y + blockIdx.x * blockDim.y;
//     for (int k = blk; k < count; k += blockDim.y * gridDim.x) {
//         sum = sum + LoadWarpVec(&data[k * MM_TILE]);
//     }
//     if (blk < count) {
//         StoreWarpVec(&data[blk * MM_TILE], sum);
//     }
// }


// void TestMemRead()
// {
//     const int ITER_COUNT = 100;
//     constexpr int SIZE = 1 << 22;
//     //constexpr int SIZE = (1 << 19) / sizeof(TDataType); // L2 fit
//     TCudaVector<TDataType> buf;
//     buf.AllocateCuda(SIZE * MM_TILE);

//     TIntrusivePtr<TGraph> computer = new TGraph;
//     {
//         TGraph *c = computer.Get();
//         for (yint iter = 0; iter < ITER_COUNT; ++iter) {
//             CudaCall(c, LoadTest).Block(32, 32).Grid(128).Read((int)SIZE).Write(&buf);
//         }
//     }
//     TStream stream;
//     for (;;) {
//         NHPTimer::STime tStart;
//         NHPTimer::GetTime(&tStart);
//         computer->Run(stream);
//         stream.Sync();
//         double tPassed = NHPTimer::GetTimePassed(&tStart);
//         double gbPerSec = sizeof(TDataType) * SIZE * MM_TILE * ITER_COUNT / tPassed / 1e9;
//         DebugPrintf("%g GB/sec\n", gbPerSec);
//     }
// }


///////////////////////////////////////////////////////////////////////////////////////////////////
void TestMatMulFp16()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<half> aMatr;
    TCuda2DArray<half> bMatr;
    TCuda2DArray<float> resMatr;
    TCuda2DArray<float> resMatrRef;

#ifndef NDEBUG
    const int ITER_COUNT = 1;
#else
    const int ITER_COUNT = 100;
#endif

    // int m = 128;
    // int n = 128;
    // int k = 128;
    int m = 4096;
    int n = 2048;
    int k = 8192;
    // int m = 32 * 1024;
    // int n = 4096;
    // int k = 2048;

    // constexpr bool TRANSPOSE_A = false;
    // constexpr bool TRANSPOSE_B = false;
    // constexpr bool TRANSPOSE_A = false;
    // constexpr bool TRANSPOSE_B = true;
    constexpr bool TRANSPOSE_A = true;
    constexpr bool TRANSPOSE_B = true;

    if (TRANSPOSE_A) {
        aMatr.Allocate(m, k);
    } else {
        aMatr.Allocate(k, m);
    }
    if (TRANSPOSE_B) {
        bMatr.Allocate(n, k);
    } else {
        bMatr.Allocate(k, n);
    }
    resMatr.Allocate(n, m);
    resMatrRef.Allocate(n, m);

    TIntrusivePtr<TGraph> c = new TGraph;
    {
        //CudaCall(c, TestLoadTile).Block(WARP_SIZE, MM_BATCH)(aMatr).Write(&aChk);
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            IgnoreSm90Kernels = false;
            Fp16MatMul<TRANSPOSE_A, TRANSPOSE_B, TStore>(c, aMatr, bMatr, &resMatr, m, n, k).Struct();
            IgnoreSm90Kernels = true;
            // Fp16MatMul<TRANSPOSE_A, TRANSPOSE_B, TStore>(c, aMatr, bMatr, &resMatrRef, m, n, k).Struct();
            auto bm = MakeMatMulArgs<TRANSPOSE_A, TRANSPOSE_B>(nullptr, aMatr, bMatr, k);
            Fp16MatMulMulti<TStore>(c, &resMatrRef, m, n, MultiArg() + bm).Struct();
        }
    }

    // TArray2D<float> axx;
    // axx.SetSizes(ySize, xSize);
    // axx.FillZero();
    // axx[0][0] = 1;
    // Put(stream, &aMatr, axx);
    // TArray2D<float> bxx;
    // bxx.SetSizes(ySize, zSize);
    // bxx.FillZero();
    // //bxx[0][80] = 1;
    // for (yint x = 0; x < 16; ++x) {
    //     for (yint y = 0; y < 16; ++y) {
    //         bxx[0 + y][0 + x] = 1;
    //     }
    // }
    // Put(stream, &bMatr, bxx);

    FillRandom(rng, stream, &aMatr);
    FillRandom(rng, stream, &bMatr);
    stream.Sync();
    //TestNewKernelCorrectness<TRANSPOSE_A, TRANSPOSE_B>(aMatr, bMatr, &resMatrRef);
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        c->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = 2. * ITER_COUNT * m * n * k / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops\n", maxTFlops);

#ifndef NDEBUG
        resMatr.CopyToHost(stream);
        resMatrRef.CopyToHost(stream);
        stream.Sync();
        TArray2D<float> ra, rb;
        GetAllData(resMatr, &ra);
        GetAllData(resMatrRef, &rb);
        for (yint y = 0; y < ra.GetYSize(); ++y) {
            for (yint x = 0; x < ra.GetXSize(); ++x) {
                //Y_VERIFY(ra[y][x] == rb[y][x]);
                if (ra[y][x] != rb[y][x]) {
                    DebugPrintf("[%g][%g] val = %g, ref = %g\n", y * 1., x * 1., ra[y][x], rb[y][x]);
                }
            }
        }
#endif
    }
}
