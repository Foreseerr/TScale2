#pragma once
#include "cuda_util.cuh"
#include "cuda_arrays.h"
#include "cuda_mma.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// kernel selector
extern bool IgnoreSm90Kernels;

bool UseSm90Kernels();


///////////////////////////////////////////////////////////////////////////////////////////////////
template <uint32_t RegCount>
__device__ void warp_reg_alloc()
{
#if (__CUDA_ARCH__ >= 900)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

template <uint32_t RegCount>
__device__ void warp_reg_free()
{
#if (__CUDA_ARCH__ >= 900)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// mbarrier
__device__ static __forceinline__ void mbarrier_init(ui64 *bar, int arrival_count)
{
#if (__CUDA_ARCH__ >= 900)
    ui32 bar_ptr = GetSharedAddress(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr), "r"(arrival_count));
#endif
}

__device__ static __forceinline__ void mbarrier_expect_bytes(ui64 *bar, ui32 bytes)
{
#if (__CUDA_ARCH__ >= 900)
    ui32 bar_ptr = GetSharedAddress(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr), "r"(bytes));
#endif
}

__device__ static __forceinline__ void mbarrier_arrive(ui64 *bar, ui32 count)
{
#if (__CUDA_ARCH__ >= 900)
    ui32 mbar_ptr = GetSharedAddress(bar);
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" : : "r"(mbar_ptr), "r"(count) : "memory");
#endif
}

__device__ static __forceinline__ void mbarrier_wait(ui64 *bar, int kPhaseBit)
{
#if (__CUDA_ARCH__ >= 900)
    ui32 mbar_ptr = GetSharedAddress(bar);
    asm volatile( //
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(mbar_ptr),
        "r"(kPhaseBit));
#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// handles array of barrirers bar[QSIZE] for queued processing
template <int QSIZE>
struct TQueueIndex
{
    int Q = 0;
    int BarrierPhase = 0;

    __device__ void Next()
    {
        if (++Q == QSIZE) {
            Q = 0;
            BarrierPhase ^= 1;
        }
    }
    __device__ int Wait(ui64 *barArr)
    {
        int res = Q;
        mbarrier_wait(&barArr[Q], BarrierPhase);
        Next();
        return res;
    }
    __device__ void Arrive(ui64 *barArr)
    {
        mbarrier_arrive(&barArr[Q], 1);
        Next();
    }
    __device__ int Allocate() // for tma copy and other async arrive usage
    {
        int res = Q;
        Next();
        return res;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// tensor map
template <class T>
inline CUtensorMapDataType GetTensorDataType()
{
    Y_VERIFY(0);
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

template <>
inline CUtensorMapDataType GetTensorDataType<half>()
{
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
}

template <>
inline CUtensorMapDataType GetTensorDataType<i8>()
{
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

template <>
inline CUtensorMapDataType GetTensorDataType<e4m3>()
{
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

template <>
inline CUtensorMapDataType GetTensorDataType<e5m2>()
{
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

template <ui32 smemHeight, ui32 smemWidth, class TArray>
CUtensorMap GetTensorMap(const TArray &arr2D)
{
    typedef typename TArray::TElem T;
    TCuda2DPtr<T> buf = arr2D.GetDevicePtr();
    ui32 globalWidth = arr2D.GetXSize();
    ui32 globalHeight = arr2D.GetYSize();
    ui64 globalStride = buf.GetStrideInBytes();

    uint64_t gmem_shape[] = {globalWidth, globalHeight}; // fastest dimension first
    uint64_t gmem_stride[] = {globalStride, 0};
    uint32_t smem_box_shape[] = {smemWidth, smemHeight};
    uint32_t smem_box_increment[] = {1, 1}; // (1,1) fill every item, (2,1) fills every other, like only reals for complex numbers

    CUtensorMap tma_map;
    CUresult result = cuTensorMapEncodeTiled(&tma_map, GetTensorDataType<T>(), 2, buf.Data, gmem_shape, gmem_stride,
        smem_box_shape, smem_box_increment, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    Y_VERIFY(result == CUDA_SUCCESS);
    return tma_map;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__device__ inline void tma_copy(void *dst, const CUtensorMap &tensorMap, ui64 *bar, int globalColumn, int globalRow)
{
#if (__CUDA_ARCH__ >= 900)
    ui32 mbar_ptr = GetSharedAddress(bar);
    ui32 dst_ptr = GetSharedAddress(dst);
    const void *tmap_addr = __cvta_constant_to_generic((size_t)&tensorMap);

    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];\n"
        :
        : "r"(dst_ptr), "l"(tmap_addr), "r"(mbar_ptr), "r"(globalColumn), "r"(globalRow)
        : "memory");
#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// wgmma
__device__ inline ui64 matrix_descriptor_encode(ui64 x)
{
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ inline ui64 make_smem_desc(void *ptr)
{
    ui32 addr = GetSharedAddress(ptr);
    ui64 desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((ui64)16) << 16;
    desc |= matrix_descriptor_encode((ui64)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

__device__ __forceinline__ void warpgroup_arrive()
{
#if (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void warpgroup_commit_batch()
{
#if (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}


template <int N>
__device__ __forceinline__ void warpgroup_wait()
{
#if (__CUDA_ARCH__ >= 900)
    CUDA_STATIC_ASSERT(N >= 0 && N <= 7); // WGMMA wait: N must be in range [0, 7]
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}


template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma64(TRegTile<float> *d, half *sA, half *sB)
{
#if (__CUDA_ARCH__ >= 900)
    ui64 desc_a = make_smem_desc(sA);
    ui64 desc_b = make_smem_desc(sB);
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
        : "+f"(d[0].x[0]), "+f"(d[0].x[1]), "+f"(d[0].x[2]), "+f"(d[0].x[3]), "+f"(d[0].x[4]), "+f"(d[0].x[5]), "+f"(d[0].x[6]), "+f"(d[0].x[7]),
          "+f"(d[1].x[0]), "+f"(d[1].x[1]), "+f"(d[1].x[2]), "+f"(d[1].x[3]), "+f"(d[1].x[4]), "+f"(d[1].x[5]), "+f"(d[1].x[6]), "+f"(d[1].x[7]),
          "+f"(d[2].x[0]), "+f"(d[2].x[1]), "+f"(d[2].x[2]), "+f"(d[2].x[3]), "+f"(d[2].x[4]), "+f"(d[2].x[5]), "+f"(d[2].x[6]), "+f"(d[2].x[7]),
          "+f"(d[3].x[0]), "+f"(d[3].x[1]), "+f"(d[3].x[2]), "+f"(d[3].x[3]), "+f"(d[3].x[4]), "+f"(d[3].x[5]), "+f"(d[3].x[6]), "+f"(d[3].x[7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(1), "n"(1), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#endif
}


// template<int ScaleD, int TransA, int TransB>
// __device__ void wgmma128(TRegTile<float> *d, half* sA, half* sB) {
//     ui64 desc_a = make_smem_desc(&sA[0]);
//     ui64 desc_b = make_smem_desc(&sB[0]);
//     asm volatile(
//         "{\n"
//         "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
//         "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,  "
//         " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15, "
//         " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
//         " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31, "
//         " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39, "
//         " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
//         " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
//         " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
//         " %64,"
//         " %65,"
//         " %66, %67, %68, %69, %70;\n"
//         "}\n"
//         : "+f"(d[0].x[0]), "+f"(d[0].x[1]), "+f"(d[0].x[2]), "+f"(d[0].x[3]), "+f"(d[0].x[4]), "+f"(d[0].x[5]), "+f"(d[0].x[6]), "+f"(d[0].x[7]),
//           "+f"(d[1].x[0]), "+f"(d[1].x[1]), "+f"(d[1].x[2]), "+f"(d[1].x[3]), "+f"(d[1].x[4]), "+f"(d[1].x[5]), "+f"(d[1].x[6]), "+f"(d[1].x[7]),
//           "+f"(d[2].x[0]), "+f"(d[2].x[1]), "+f"(d[2].x[2]), "+f"(d[2].x[3]), "+f"(d[2].x[4]), "+f"(d[2].x[5]), "+f"(d[2].x[6]), "+f"(d[2].x[7]),
//           "+f"(d[3].x[0]), "+f"(d[3].x[1]), "+f"(d[3].x[2]), "+f"(d[3].x[3]), "+f"(d[3].x[4]), "+f"(d[3].x[5]), "+f"(d[3].x[6]), "+f"(d[3].x[7]),
//           "+f"(d[4].x[0]), "+f"(d[4].x[1]), "+f"(d[4].x[2]), "+f"(d[4].x[3]), "+f"(d[4].x[4]), "+f"(d[4].x[5]), "+f"(d[4].x[6]), "+f"(d[4].x[7]),
//           "+f"(d[5].x[0]), "+f"(d[5].x[1]), "+f"(d[5].x[2]), "+f"(d[5].x[3]), "+f"(d[5].x[4]), "+f"(d[5].x[5]), "+f"(d[5].x[6]), "+f"(d[5].x[7]),
//           "+f"(d[6].x[0]), "+f"(d[6].x[1]), "+f"(d[6].x[2]), "+f"(d[6].x[3]), "+f"(d[6].x[4]), "+f"(d[6].x[5]), "+f"(d[6].x[6]), "+f"(d[6].x[7]),
//           "+f"(d[7].x[0]), "+f"(d[7].x[1]), "+f"(d[7].x[2]), "+f"(d[7].x[3]), "+f"(d[7].x[4]), "+f"(d[7].x[5]), "+f"(d[7].x[6]), "+f"(d[7].x[7])
//         : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(1), "n"(1), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
// }

template <int ScaleD>
__device__ __forceinline__ void wgmma64e4e4(TRegTile<float> *d, i8 *sA, i8 *sB)
{
#if (__CUDA_ARCH__ >= 900)
    ui64 desc_a = make_smem_desc(sA);
    ui64 desc_b = make_smem_desc(sB);
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36;\n"
                 "}\n"
        : "+f"(d[0].x[0]), "+f"(d[0].x[1]), "+f"(d[0].x[2]), "+f"(d[0].x[3]), "+f"(d[0].x[4]), "+f"(d[0].x[5]), "+f"(d[0].x[6]), "+f"(d[0].x[7]),
          "+f"(d[1].x[0]), "+f"(d[1].x[1]), "+f"(d[1].x[2]), "+f"(d[1].x[3]), "+f"(d[1].x[4]), "+f"(d[1].x[5]), "+f"(d[1].x[6]), "+f"(d[1].x[7]),
          "+f"(d[2].x[0]), "+f"(d[2].x[1]), "+f"(d[2].x[2]), "+f"(d[2].x[3]), "+f"(d[2].x[4]), "+f"(d[2].x[5]), "+f"(d[2].x[6]), "+f"(d[2].x[7]),
          "+f"(d[3].x[0]), "+f"(d[3].x[1]), "+f"(d[3].x[2]), "+f"(d[3].x[3]), "+f"(d[3].x[4]), "+f"(d[3].x[5]), "+f"(d[3].x[6]), "+f"(d[3].x[7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(1), "n"(1));
#endif
}

template <int ScaleD>
__device__ __forceinline__ void wgmma128e4e4(TRegTile<float> *d, i8 *sA, i8 *sB)
{
#if (__CUDA_ARCH__ >= 900)
    ui64 desc_a = make_smem_desc(sA);
    ui64 desc_b = make_smem_desc(sB);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,  "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15, "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31, "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39, "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66, %67, %68;\n"
        "}\n"
        : "+f"(d[0].x[0]), "+f"(d[0].x[1]), "+f"(d[0].x[2]), "+f"(d[0].x[3]), "+f"(d[0].x[4]), "+f"(d[0].x[5]), "+f"(d[0].x[6]), "+f"(d[0].x[7]),
          "+f"(d[1].x[0]), "+f"(d[1].x[1]), "+f"(d[1].x[2]), "+f"(d[1].x[3]), "+f"(d[1].x[4]), "+f"(d[1].x[5]), "+f"(d[1].x[6]), "+f"(d[1].x[7]),
          "+f"(d[2].x[0]), "+f"(d[2].x[1]), "+f"(d[2].x[2]), "+f"(d[2].x[3]), "+f"(d[2].x[4]), "+f"(d[2].x[5]), "+f"(d[2].x[6]), "+f"(d[2].x[7]),
          "+f"(d[3].x[0]), "+f"(d[3].x[1]), "+f"(d[3].x[2]), "+f"(d[3].x[3]), "+f"(d[3].x[4]), "+f"(d[3].x[5]), "+f"(d[3].x[6]), "+f"(d[3].x[7]),
          "+f"(d[4].x[0]), "+f"(d[4].x[1]), "+f"(d[4].x[2]), "+f"(d[4].x[3]), "+f"(d[4].x[4]), "+f"(d[4].x[5]), "+f"(d[4].x[6]), "+f"(d[4].x[7]),
          "+f"(d[5].x[0]), "+f"(d[5].x[1]), "+f"(d[5].x[2]), "+f"(d[5].x[3]), "+f"(d[5].x[4]), "+f"(d[5].x[5]), "+f"(d[5].x[6]), "+f"(d[5].x[7]),
          "+f"(d[6].x[0]), "+f"(d[6].x[1]), "+f"(d[6].x[2]), "+f"(d[6].x[3]), "+f"(d[6].x[4]), "+f"(d[6].x[5]), "+f"(d[6].x[6]), "+f"(d[6].x[7]),
          "+f"(d[7].x[0]), "+f"(d[7].x[1]), "+f"(d[7].x[2]), "+f"(d[7].x[3]), "+f"(d[7].x[4]), "+f"(d[7].x[5]), "+f"(d[7].x[6]), "+f"(d[7].x[7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(1), "n"(1));
#endif
}
}
