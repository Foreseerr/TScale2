#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"

namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float4 Fabs(float4 a)
{
    return make_float4(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w));
}

inline __device__ float4 Exp2f(float4 a)
{
    return make_float4(exp2f(a.x), exp2f(a.y), exp2f(a.z), exp2f(a.w));
}

inline __device__ float4 Log2f(float4 a)
{
    return make_float4(log2f(a.x), log2f(a.y), log2f(a.z), log2f(a.w));
}

inline __device__ float4 Scale(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ float4 Max(float4 a, float4 b)
{
    return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __device__ float4 AddScaled(float4 a, float4 b, float scale)
{
    return make_float4(a.x + b.x * scale, a.y + b.y * scale, a.z + b.z * scale, a.w + b.w * scale);
}

inline __device__ float HorizontalSum(float4 a)
{
    return a.x + a.y + a.z + a.w;
}

inline __device__ float HorizontalMax(float4 a)
{
    return max(max(a.x, a.y), max(a.z, a.w));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// vector utils, 128 dim vector per warp
constexpr int WARP_VEC_DIM = 128;

inline __device__ float4 ZeroWarpVec()
{
    return make_float4(0, 0, 0, 0);
}

inline __device__ float4 InitWarpVec(float x)
{
    return make_float4(x, x, x, x);
}

template <class Func>
inline __device__ void EnumWarpVecElements(float4 *vec, Func f)
{
    int thrOffset = threadIdx.x * 4;
    f(thrOffset + 0, &vec->x);
    f(thrOffset + 1, &vec->y);
    f(thrOffset + 2, &vec->z);
    f(thrOffset + 3, &vec->w);
}

inline __device__ float CalcWarpVecSum2(float4 vec)
{
    return WarpSum(HorizontalSum(vec * vec));
}

inline __device__ float CalcWarpVecSum(float4 vec)
{
    return WarpSum(HorizontalSum(vec));
}

inline __device__ float CalcWarpVecMax(float4 vec)
{
    return WarpMax(HorizontalMax(vec));
}

inline __device__ float CalcWarpVecMaxAbsValue(float4 vec)
{
    return WarpMax(HorizontalMax(Fabs(vec)));
}

inline __device__ float DotProductWarpVec(float4 a, float4 b)
{
    float dp = HorizontalSum(a * b);
    return WarpSum(dp);
}

// src - not normalized source vector, grad - array gradient, returns gradient of pre normalization vector
inline __device__ float4 TileNormalizeBackpropWarpVec(float4 src, float4 grad)
{
    float sum2 = CalcWarpVecSum2(src);
    if (sum2 == 0) {
        return ZeroWarpVec();
    } else {
        float dp = DotProductWarpVec(src, grad);

        float sigma = dp / sum2;
        float scale = sqrtf(WARP_VEC_DIM / sum2);
        float4 projectedGrad = grad - Scale(src, sigma);
        return Scale(projectedGrad, scale);
    }
}


inline __device__ float NormalizeWarpVec(float4 *vec, float mult)
{
    float sum2 = CalcWarpVecSum2(*vec);
    float discrScale = 0;
    if (sum2 > 0) {
        discrScale = sqrt(sum2 / WARP_VEC_DIM);
        Scale(*vec, mult / discrScale);
    }
    return discrScale;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// load warp vec
inline __device__ int4 LoadGlobal16BytesBypassL1(const void *p)
{
    int4 res;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w) : "l"(p));
    return res;
}

inline __device__ int2 LoadGlobal8BytesBypassL1(const void *p)
{
    int2 res;
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(res.x), "=r"(res.y) : "l"(p));
    return res;
}

inline __device__ int LoadGlobal4BytesBypassL1(const void *p)
{
    int res;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(res) : "l"(p));
    return res;
}


inline __device__ float4 LoadWarpVec(const float *src)
{
    union {
        int4 ival;
        float4 fval;
    };
    int h = threadIdx.x;
    ival = LoadGlobal16BytesBypassL1(src + h * 4);
    return fval;
}


inline __device__ float4 LoadWarpVec(const half *src)
{
    int h = threadIdx.x;
    union {
        int2 data;
        half res[4];
    };
    data = LoadGlobal8BytesBypassL1(src + h * 4);
    return make_float4(res[0], res[1], res[2], res[3]);
}


template <class T>
inline __device__ float4 LoadWarpVecByteWidth(const T *src)
{
    int h = threadIdx.x;
    union {
        int data;
        T res[4];
    };
    data = LoadGlobal4BytesBypassL1(src + h * 4);
    return make_float4(res[0], res[1], res[2], res[3]);
}

inline __device__ float4 LoadWarpVec(const i8 *src)
{
    return LoadWarpVecByteWidth(src);
}

inline __device__ float4 LoadWarpVec(const e4m3 *src)
{
    return LoadWarpVecByteWidth(src);
}

inline __device__ float4 LoadWarpVec(const e5m2 *src)
{
    return LoadWarpVecByteWidth(src);
}


template <class TSrc>
inline __device__ float4 LoadWarpVecSmem(const TSrc *src)
{
    int thrOffset = threadIdx.x * 4;
    return make_float4(src[thrOffset], src[thrOffset + 1], src[thrOffset + 2], src[thrOffset + 3]);
}


inline __device__ float4 LoadWarpVecCached(const float *src)
{
    int thrOffset = threadIdx.x * 4;
    return *(const float4 *)(src + thrOffset);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// store warp vec
inline __device__ void StoreWarpVec(float *dst, float4 src)
{
    int thrOffset = threadIdx.x * 4;
    *(float4 *)(dst + thrOffset) = src;
}

inline __device__ void StoreWarpVec(half *dst, float4 src)
{
    int h = threadIdx.x;
    int2 grp;
    grp.x = CvtToHalf2(src.x, src.y);
    grp.y = CvtToHalf2(src.z, src.w);
    *(int2 *)(dst + h * 4) = grp;
}

template <class T>
inline __device__ void StoreWarpVecPacked(T *dst, int packed)
{
    CUDA_STATIC_ASSERT(sizeof(T) == 1);
    int h = threadIdx.x;
    *(int *)(dst + h * 4) = packed;
}

inline __device__ void StoreWarpVec(i8 *dst, float4 src)
{
    StoreWarpVecPacked(dst, CvtToI8(src));
}

inline __device__ void StoreWarpVec(e4m3 *dst, float4 src)
{
    StoreWarpVecPacked(dst, CvtToE4m3(src));
}

inline __device__ void StoreWarpVec(e5m2 *dst, float4 src)
{
    StoreWarpVecPacked(dst, CvtToE5m2(src));
}

template <class TDst>
inline __device__ void StoreZeroWarpVec(TDst *dst)
{
    StoreWarpVec(dst, ZeroWarpVec());
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// vector utils, vectors reside in smem
template <class TSrc>
inline __device__ void LoadSmemVec(int dim, TSrc *src, float *buf)
{
    // could use async loads to fully utilize bandwidth with single warp?
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        buf[d] = src[d];
    }
}


inline __device__ float CalcMaxAbsValueSmem(int dim, float *buf)
{
    int h = threadIdx.x;
    float maxVal = 0;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        float val = buf[d];
        maxVal = max(maxVal, fabs(val));
    }
    return WarpMax(maxVal);
}


template <class TDst>
inline __device__ void StoreZeroVec(int dim, TDst *dst)
{
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        StoreConvertedFloat(0, &dst[d]);
    }
}


template <class TDst>
inline __device__ void StoreScaledSmemVec(int dim, float *buf, float mult, TDst *dst)
{
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        StoreConvertedFloat(buf[d] * mult, &dst[d]);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// debug kernels
template <int STATE_DIM>
__global__ void TestNan(int stepId, int id, TCuda2DPtr<float> vec)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    for (int d = h; d < STATE_DIM; ++d) {
        float val = vec[t][d];
        if (isnan(val) || !isfinite(val)) {
            printf("TestNan(%d / %d), t = %d, [%d], %g, %x\n", stepId, id, t, d, val, __float_as_int(val));
            return;
        }
    }
}


template <int STATE_DIM>
__global__ void TestNanHalf(int stepId, int id, TCuda2DPtr<half> vec)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    for (int d = h; d < STATE_DIM; ++d) {
        float val = vec[t][d];
        if (isnan(val) || !isfinite(val)) {
            printf("TestNanHalf(%d / %d), t = %d, [%d], %g, %x\n", stepId, id, t, d, val, (int)__half_as_ushort(vec[t][d]));
            return;
        }
    }
}


template <int STATE_DIM>
__global__ void TestNanE4M3(int stepId, int id, TCuda2DPtr<e4m3> vec)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    for (int d = h; d < STATE_DIM; ++d) {
        float val = vec[t][d];
        if (isnan(val) || !isfinite(val)) {
            printf("TestNanFp8(%d / %d), t = %d, [%d], %g, %x\n", stepId, id, t, d, val, (ui8)vec[t][d].Data);
            return;
        }
    }
}


template <int STATE_DIM, class T>
__global__ void VecsCheckSum(int len, TCuda2DPtr<T> vecs)
{
    int h = threadIdx.x;
    int chkSum = 0;
    for (int t = 0; t < len; ++t) {
        for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
            int d = k * WARP_SIZE + threadIdx.x;
            float val = vecs[t][d];
            chkSum += __float_as_int(val);
        }
    }
    chkSum = WarpIntSum(chkSum);
    if (h == 0) {
        printf("vecs %p, chksum %d\n", &vecs[0][0], chkSum);
    }
}


template <class T>
__global__ void PrintValue(T *p)
{
    if (threadIdx.x == 0) {
        printf("Value = %g\n", float(*p));
    }
}


template <int STATE_DIM, class T>
__global__ void PrintVec(int offset, TCuda1DPtr<T> vec)
{
    for (int k = threadIdx.x; k < STATE_DIM; k += WARP_SIZE) {
        int d = k + offset;
        float val = vec[d];
        printf("gpu vec[%g] = %g\n", d * 1., val);
    }
}


template <int STATE_DIM, class T>
__global__ void PrintArr(int t, TCuda2DPtr<T> vecs)
{
    for (int k = threadIdx.x; k < STATE_DIM; k += WARP_SIZE) {
        float val = vecs[t][k];
        printf("gpu vec[%g] = %g\n", k * 1., val);
    }
}

}
