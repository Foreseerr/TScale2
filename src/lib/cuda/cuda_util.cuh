#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <util/fp8.h>


#ifdef NDEBUG
#define CUDA_ASSERT( B )
#else
#define CUDA_ASSERT( B ) if (!(B)) { printf("assert failed " #B "\n"); }
#endif


namespace NCuda
{
constexpr int WARP_SIZE = 32;
constexpr int MAX_WARPS = 32;


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace staticassert
{
    template <bool x> struct CheckStruct;
    template <> struct CheckStruct<true> { int X; };// enum { value = 1 };
    template<int x> struct test {};
};

#define CUDA_STATIC_ASSERT( B )  typedef staticassert::test<sizeof(staticassert::CheckStruct< (bool)(B) >) > static_assert_chk_ ## __LINE__


///////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline __device__ char *AlignDevicePtr(char *p, yint x)
{
    char *np = (char *)nullptr;
    return np + (((p - np) + x - 1) & ~(x - 1));
}

__forceinline __device__ ui32 GetSharedAddress(const void *ptr)
{
    return static_cast<ui32>(__cvta_generic_to_shared(ptr));
}

template <class T>
inline __device__ void DoSwapDevice(T &a, T &b)
{
    T k = a;
    a = b;
    b = k;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
inline __device__ i32 CvtToI32(float x)
{
    int32_t res;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(res) : "f"(x));
    return res;
}

inline __device__ i8 CvtToI8(float x)
{
    int32_t res;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(x));
    return res;
}

inline __device__ int CvtToHalf2(float x, float y)
{
    int res;
    asm volatile("cvt.rn.satfinite.f16x2.f32 %0, %2, %1;" : "=r"(res) : "f"(x), "f"(y));
    return res;
}

inline __device__ int CvtToI8(float4 val)
{
    union {
        i8 val8[4];
        int val32;
    };
    val8[0] = CvtToI8(val.x);
    val8[1] = CvtToI8(val.y);
    val8[2] = CvtToI8(val.z);
    val8[3] = CvtToI8(val.w);
    return val32;
}

inline __device__ int CvtToE4m3(float4 val)
{
#if (__CUDA_ARCH__ >= 890)
    union {
        ui16 val16[2];
        int val32;
    };
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(val16[0]) : "f"(val.x), "f"(val.y));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(val16[1]) : "f"(val.z), "f"(val.w));
    return val32;
#else
    printf("sm89 required\n");
    return 0;
#endif
}

inline __device__ int CvtToE5m2(float4 val)
{
#if (__CUDA_ARCH__ >= 890)
    union {
        ui16 val16[2];
        int val32;
    };
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;\n" : "=h"(val16[0]) : "f"(val.x), "f"(val.y));
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;\n" : "=h"(val16[1]) : "f"(val.z), "f"(val.w));
    return val32;
#else
    printf("sm89 required\n");
    return 0;
#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst>
inline __device__ void StoreConvertedFloat(float x, TDst *p)
{
    *p = (TDst)x;
}

inline __device__ void StoreConvertedFloat(float x, half *p)
{
    asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(*(ui16 *)p) : "f"(x));
}

inline __device__ void StoreConvertedFloat(float x, i8 *p)
{
    *p = CvtToI8(x);
}

inline __device__ void StoreConvertedFloat(float x, e4m3 *p)
{
    p->Data = CvtToE4(x);
}

inline __device__ void StoreConvertedFloat(float x, e5m2 *p)
{
    p->Data = CvtToE5(x);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst>
inline __device__ void StoreZero(TDst *p)
{
    *p = 0;
}

inline __device__ void StoreZero(e4m3 *p)
{
    p->Data = 0;
}

inline __device__ void StoreZero(e5m2 *p)
{
    p->Data = 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void StoreConvertedFloat2(float2 val, float *p)
{
    *(float2 *)p = val;
}

inline __device__ void StoreConvertedFloat2(float2 val, half *p)
{
    asm volatile("cvt.rn.satfinite.f16x2.f32 %0, %2, %1;" : "=r"(*(ui32*)p) : "f"(val.x), "f"(val.y));
}

inline __device__ void StoreConvertedFloat2(float2 val, i8 *p)
{
    union {
        i8 val8[2];
        ui16 val16;
    };
    val8[0] = CvtToI8(val.x);
    val8[1] = CvtToI8(val.y);
    *(ui16 *)p = val16;
}

inline __device__ void StoreConvertedFloat2(float2 val, e4m3 *p)
{
#if (__CUDA_ARCH__ >= 890)
    float fHigh = val.y;
    float fLow = val.x;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(val16) : "f"(fLow), "f"(fHigh));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}

inline __device__ void StoreConvertedFloat2(float2 val, e5m2 *p)
{
#if (__CUDA_ARCH__ >= 890)
    float fHigh = val.y;
    float fLow = val.x;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;\n" : "=h"(val16) : "f"(fLow), "f"(fHigh));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}


inline __device__ void StoreScaledFloat2(float2 val, float *p, float mult)
{
    *(float2 *)p = make_float2(val.x * mult, val.y * mult);
}

inline __device__ void StoreScaledFloat2(float2 val, half *p, float mult)
{
    *(half2 *)p = make_half2(val.x * mult, val.y * mult);
}


inline __device__ void StoreAddScaledFloat2(float2 val, float *p, float mult)
{
    float2 old = *(float2 *)p;
    *(float2 *)p = make_float2(old.x + val.x * mult, old.y + val.y * mult);
}

inline __device__ void StoreAddScaledFloat2(float2 val, half *p, float mult)
{
    half2 old = *(half2 *)p;
    *(half2 *)p = old + make_half2(half(val.x * mult), half(val.y * mult));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void StoreConvertedFloat2(half2 val, float *p)
{
    *(float2 *)p = make_float2(val.x, val.y);
}

inline __device__ void StoreConvertedFloat2(half2 val, half *p)
{
    *(half2 *)p = val;
}

inline __device__ void StoreConvertedFloat2(half2 val, i8 *p)
{
    union {
        i8 val8[2];
        ui16 val16;
    };
    val8[0] = CvtToI8(val.x);
    val8[1] = CvtToI8(val.y);
    *(ui16 *)p = val16;
}

inline __device__ void StoreConvertedFloat2(half2 val, e4m3 *p)
{
#if (__CUDA_ARCH__ >= 890)
    int valSrc = *(int *)&val;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(val16) : "r"(valSrc));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}

inline __device__ void StoreConvertedFloat2(half2 val, e5m2 *p)
{
#if (__CUDA_ARCH__ >= 890)
    int valSrc = *(int *)&val;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;\n" : "=h"(val16) : "r"(valSrc));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}


inline __device__ void StoreScaledFloat2(half2 val, float *p, float mult)
{
    *(float2 *)p = make_float2(float(val.x) * mult, float(val.y) * mult);
}

inline __device__ void StoreScaledFloat2(half2 val, half *p, float mult)
{
    *(half2 *)p = val * make_half2(mult, mult);
}


inline __device__ void StoreAddScaledFloat2(half2 val, float *p, float mult)
{
    float2 old = *(float2 *)p;
    *(float2 *)p = make_float2(old.x + float(val.x) * mult, old.y + float(val.y) * mult);
}

inline __device__ void StoreAddScaledFloat2(half2 val, half *p, float mult)
{
    half2 old = *(half2 *)p;
    *(half2 *)p = old + val * make_half2(mult, mult);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void StoreConvertedFloat2(int2 val, int *p)
{
    *(int2 *)p = val;
}

inline __device__ void StoreConvertedFloat2(int2 val, float *p)
{
    *(float2 *)p = make_float2(val.x, val.y);
}

inline __device__ void StoreConvertedFloat2(int2 val, half *p)
{
    *(half2 *)p = make_half2(val.x, val.y);
}


inline __device__ void StoreScaledFloat2(int2 val, int *p, float mult)
{
    *(int2 *)p = make_int2(CvtToI32(val.x * mult), CvtToI32(val.y * mult));
}

inline __device__ void StoreScaledFloat2(int2 val, float *p, float mult)
{
    *(float2 *)p = make_float2(val.x * mult, val.y * mult);
}

inline __device__ void StoreScaledFloat2(int2 val, half *p, float mult)
{
    *(half2 *)p = make_half2(val.x * mult, val.y * mult);
}


inline __device__ void StoreAddScaledFloat2(int2 val, int *p, float mult)
{
    int2 old = *(int2*)p;
    *(int2 *)p = make_int2(old.x + CvtToI32(val.x * mult), old.y + CvtToI32(val.y * mult));
}

inline __device__ void StoreAddScaledFloat2(int2 val, float *p, float mult)
{
    float2 old = *(float2 *)p;
    *(float2 *)p = make_float2(old.x + val.x * mult, old.y + val.y * mult);
}

inline __device__ void StoreAddScaledFloat2(int2 val, half *p, float mult)
{
    half2 old = *(half2 *)p;
    *(half2 *)p = old + make_half2(half(val.x * mult), half(val.y * mult));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// discr scale for normalizations to max value
template <class T>
static __device__ float GetMaxDiscrScale(float maxValue, T *)
{
    Y_VERIFY(0);
}
static __device__ float GetMaxDiscrScale(float maxValue, i8 *)
{
    return maxValue / 127;
}
static __device__ float GetMaxDiscrScale(float maxValue, e4m3 *)
{
    return maxValue / 256;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float RoundFloatUp(float x)
{
    // round scale to avoid precision loss in I8MatMulXYoZYeXZlarge() due to long chain of sum multiplication by close to 1 numbers
    float tail = __int_as_float(__float_as_int(x) | 0xfffff) - __int_as_float(__float_as_int(x) & 0xfff00000);
    float res = __int_as_float(__float_as_int(x + tail) & 0xfff00000); // round up
    return res;
}

inline __device__ float TruncateToPow2(float x)
{
    return __int_as_float(__float_as_int(x) & 0xff800000);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// warp sum
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
constexpr unsigned ALL_WARPS_MASK = 0xffffffff;

inline __device__ float WarpSum(float val)
{
    CUDA_STATIC_ASSERT(WARP_SIZE == 32);
    float sum = val;
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 16);
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 8);
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 4);
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 2);
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 1);
    return sum;
}

inline __device__ int WarpIntSum(int val)
{
    return __reduce_add_sync(ALL_WARPS_MASK, val);
}

inline __device__ float HalfWarpSum(float val)
{
    CUDA_STATIC_ASSERT(WARP_SIZE == 32);
    float sum = val;
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 8);
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 4);
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 2);
    sum += __shfl_xor_sync(ALL_WARPS_MASK, sum, 1);
    return sum;
}

inline __device__ float WarpMax(float val)
{
    CUDA_STATIC_ASSERT(WARP_SIZE == 32);
    float res = val;
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 16));
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 8));
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 4));
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 2));
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 1));
    return res;
}

inline __device__ float HalfWarpMax(float val)
{
    CUDA_STATIC_ASSERT(WARP_SIZE == 32);
    float res = val;
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 8));
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 4));
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 2));
    res = fmaxf(res, __shfl_xor_sync(ALL_WARPS_MASK, res, 1));
    return res;
}

inline __device__ int WarpMinInt(int val)
{
    return __reduce_min_sync(ALL_WARPS_MASK, val);
}

inline __device__ int WarpMaxInt(int val)
{
    return __reduce_max_sync(ALL_WARPS_MASK, val);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ int WarpMaxIdx(float val, int idx)
{
    CUDA_STATIC_ASSERT(WARP_SIZE == 32);
    float res = val;
    int resIdx = idx;
    for (int xx = 16; xx > 0; xx /= 2) {
        float chk = __shfl_xor_sync(ALL_WARPS_MASK, res, xx);
        float chkIdx = __shfl_xor_sync(ALL_WARPS_MASK, resIdx, xx);
        if (chk > res) {
            res = chk;
            resIdx = chkIdx;
        } else if (chk == res && chkIdx < resIdx) {
            resIdx = chkIdx;
        }
    }
    return resIdx;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float BlockSum(float x)
{
    CUDA_ASSERT(blockDim.y <= MAX_WARPS);
    CUDA_ASSERT(MAX_WARPS <= WARP_SIZE);
    __shared__ float val[MAX_WARPS];
    __syncthreads();
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    float sum = WarpSum(x);
    if (h == 0) {
        val[warpId] = sum;
    }
    __syncthreads();
    sum = (h < blockDim.y) ? val[h] : 0;
    sum = WarpSum(sum);
    return sum;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// synchronize multiple warps, uses same bar ID, modify to use different IDs
__forceinline__ __device__ void BarSync(int id, int count)
{
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count));
}

__forceinline__ __device__ void BarArrive(int id, int count)
{
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void atomicAddExact(float *pDst, float val)
{
    int *p = (int*)pDst;
    for (;;) {
        int assumed = *p;// assumed = old;
        if (atomicCAS(p, assumed, __float_as_int(val + __int_as_float(assumed))) == assumed) {
            return;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline __device__ T *AdvancePtr(T *p, int offset)
{
    return (T *)(((char *)p) + offset);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TCudaRngLCG
{
    ui32 State;

    __device__ TCudaRngLCG(ui32 a, ui32 b, ui32 c)
    {
        State = a * 0x398bf183 + b * 0x1affc3df + c * 0x9023a049;
    }
    __device__ ui32 Gen()
    {
        State = State * 0x448e3079 + 0xc484ef10;
        return State;
    }
    __device__ float GenUniformFloat()
    {
        return Gen() * (1.0f / float(1ll << 32));
    }
    __device__ float GenRandReal3() // (0;1)
    {
        return min(max(GenUniformFloat(), 1e-10f), 0.999999f);
    }
};
}
