#include <util/pch.h>
#define KERNEL_UNIT "cuda_fp8/"
#include "cuda_fp8.cuh"
#include "cuda_graph.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


using namespace NCuda;

void TestMatMulFp8()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<e4m3> aMatr;
    TCuda2DArray<e4m3> bMatr;
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
    aMatr.Allocate(k, m);
    bMatr.Allocate(k, n);
    resMatr.Allocate(n, m);
    resMatrRef.Allocate(n, m);

    TIntrusivePtr<TGraph> c = new TGraph;
    {
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            IgnoreSm90Kernels = false;
            Fp8MatMul<TStore>(c, aMatr, bMatr, &resMatr, m, n, k).Struct();
#ifndef NDEBUG
            IgnoreSm90Kernels = true;
            //Fp8MatMul<TStore>(c, aMatr, bMatr, &resMatrRef, m, n, k).Struct();
            auto bm = MakeMatMulArgs<0, 0>(nullptr, aMatr, bMatr, k);
            Fp8MatMulMulti<TStore>(c, &resMatrRef, m, n, MultiArg() + bm).Struct();
#endif
        }
    }

    FillRandom(rng, stream, &aMatr);
    FillRandom(rng, stream, &bMatr);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        c->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = 2. * ITER_COUNT * m * n * k / tPassed / 1e12;
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
                // Y_VERIFY(ra[y][x] == rb[y][x]);
                if (fabs(ra[y][x] - rb[y][x]) > 120) {
                    DebugPrintf("[%g][%g] val = %g, ref = %g\n", y * 1., x * 1., ra[y][x], rb[y][x]);
                }
            }
        }
#endif
    }
}
