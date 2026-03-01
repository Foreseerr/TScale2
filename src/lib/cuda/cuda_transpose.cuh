#pragma once
#include "cuda_util.cuh"
#include "cuda_arrays.h"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TCopy4Type;

template <>
struct TCopy4Type<half>
{
    typedef int2 TCopyType;
};

template <>
struct TCopy4Type<e4m3>
{
    typedef int TCopyType;
};

template <>
struct TCopy4Type<i8>
{
    typedef int TCopyType;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TVec4
{
    T x, y, z, w;
};

template <class T>
inline __device__ TVec4<T> LoadPackedVec4(T *p)
{
    typedef typename TCopy4Type<T>::TCopyType TCopy;
    union {
        TCopy data;
        TVec4<T> vec;
    };
    TCopy *p4 = (TCopy *)p;
    data = __ldg(p4 + threadIdx.x); // bypass L1
    return vec;

}

template <class T>
inline __device__ void StorePackedVec4(T *p, TVec4<T> vecArg)
{
    typedef typename TCopy4Type<T>::TCopyType TCopy;
    union {
        TCopy data;
        TVec4<T> vec;
    };
    TCopy *p4 = (TCopy *)p;
    vec = vecArg;
    p4[threadIdx.x] = data;
}

template <class T>
inline __device__ TVec4<T> ZeroPackedVec4()
{
    TVec4<T> zero = {};
    return zero;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TTransposeBuf
{
    T Buf[128][128 + 16];

    __device__ void CopyToGmem(TCuda2DPtr<T> dst, int offsetX, int offsetY)
    {
        int x = threadIdx.x * 4;
        for (int y = threadIdx.y; y < 128; y += blockDim.y) {
            typedef typename TCopy4Type<T>::TCopyType TCopy;
            TCopy *srcPtr = (TCopy *)&Buf[y][x];
            TCopy *dstPtr = (TCopy *)&dst[offsetY + y][offsetX + x];
            *dstPtr = *srcPtr;
        }
    }

    __device__ void StoreVecTransposed(int y, float4 vec)
    {
        int x = threadIdx.x * 4;
        StoreConvertedFloat(vec.x, &Buf[x + 0][y]);
        StoreConvertedFloat(vec.y, &Buf[x + 1][y]);
        StoreConvertedFloat(vec.z, &Buf[x + 2][y]);
        StoreConvertedFloat(vec.w, &Buf[x + 3][y]);
    }

    __device__ void StoreVecTransposed(int y, TVec4<T> vec)
    {
        int x = threadIdx.x * 4;
        Buf[x + 0][y] = vec.x;
        Buf[x + 1][y] = vec.y;
        Buf[x + 2][y] = vec.z;
        Buf[x + 3][y] = vec.w;
    }
};
}
