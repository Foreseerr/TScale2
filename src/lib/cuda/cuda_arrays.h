#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <util/fp8.h>
#include "cuda_memory.h"

#ifdef __CUDACC__
#include "cuda_util.cuh"
#else
#define CUDA_ASSERT(x)
#endif


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
enum EMemType
{
    MT_HOST,
    MT_DEVICE,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda memory blob
struct TMemoryBlob
{
    void *Ptr = 0;
    yint Stride = 0; // in bytes
    yint YSize = 0;

    TMemoryBlob() {}
    TMemoryBlob(void *p, yint stride, yint ySize) : Ptr(p), Stride(stride), YSize(ySize) {}
    TMemoryBlob(void *p, yint stride) : Ptr(p), Stride(stride), YSize(1) {}
    yint GetSize() const { return YSize * Stride; }
    bool IsSameSize(const TMemoryBlob &x) const
    {
        return Stride == x.Stride && YSize == x.YSize;
    }
    template <class T>
    T *GetElementAddress(yint x, yint y) const
    {
        Y_ASSERT(y >= 0 && y < YSize);
        char *buf = (char *)Ptr;
        return (T *)(buf + y * Stride + x * sizeof(T));
    }
    yint GetStrideInBytes() const { return Stride; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda 1D array
template<class T>
class TCudaPOD
{
    const void *Owner; // pointer to owner of this pod (array/vector for example)
    TIntrusivePtr<TCudaMemoryPool> Pool;
    void *Data;

public:
    TCudaPOD(const void *owner, TPtrArg<TCudaMemoryPool> pool, void *pData) : Owner(owner), Pool(pool.Get()), Data(pData) {}
    T *GetDevicePtr() const { return (T*)Data; }
    const void *GetOwner() const { return Owner; }
    TPtrArg<TCudaMemoryPool> GetMemPool() const { return Pool; }
};


template<class T>
struct TCuda1DPtr
{
    T *Data;
#ifndef NDEBUG
    int Size;
#endif

    __host__ __device__ TCuda1DPtr(void *data, int size) : Data((T *)data)
    {
#ifndef NDEBUG
        Size = size;
#endif
    }
#ifndef NDEBUG
    __forceinline __device__ T &operator[](int x) const
    {
        CUDA_ASSERT(x >= 0 && x < Size);
        return Data[x];
    }
#else
    __forceinline __device__ operator T*() const
    {
        return Data;
    }
#endif
};


template <class T>
class TCudaVectorFragment
{
    const void *Owner; // pointer to owner of this pod (array/vector for example)
    TIntrusivePtr<TCudaMemoryPool> Pool;
    yint Size;
    void *DeviceData;

public:
    typedef T TElem;

    TCudaVectorFragment(const void *owner, TPtrArg<TCudaMemoryPool> pool, yint sz, void *pData)
        : Owner(owner), Pool(pool.Get()), Size(sz), DeviceData(pData)
    {
    }
    TCuda1DPtr<T> GetDevicePtr() const { return TCuda1DPtr<T>(DeviceData, Size); }
    TMemoryBlob GetDeviceMem() const { return TMemoryBlob(DeviceData, sizeof(T) * Size, 1); }
    const void *GetOwner() const { return Owner; }
    TPtrArg<TCudaMemoryPool> GetMemPool() const { return Pool; }
    yint GetSize() const { return Size; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class TCudaVector : public TNonCopyable
{
    yint Size = 0;
    TCudaMemory Mem;

public:
    typedef T TElem;

    yint GetSize() const { return Size; }
    const void *GetOwner() const { return this; }

    // allocate
    void Allocate(yint count)
    {
        Size = count;
        Mem.Allocate(count * sizeof(T), TCudaMemory::CUDA_ALLOC, cudaHostAllocDefault);
    }
    void AllocateHost(yint count)
    {
        Size = count;
        Mem.Allocate(count * sizeof(T), TCudaMemory::CUDA_MAP, cudaHostAllocDefault);
    }
    void AllocateCuda(yint count)
    {
        Size = count;
        Mem.Allocate(count * sizeof(T), TCudaMemory::CUDA_ALLOC, -1);
    }
    void AllocateCuda(yint count, TPtrArg<TCudaMemoryPool> pool)
    {
        Size = count;
        Mem.AllocateFromCudaPool(count * sizeof(T), pool);
    }
    TPtrArg<TCudaMemoryPool> GetMemPool() const
    {
        return Mem.GetMemPool();
    }

    // allocate and assign
    void Init(const TVector<T> &vec)
    {
        if (!vec.empty()) {
            AllocateCuda(YSize(vec));
            Mem.CopyToDevice(vec.data(), YSize(vec) * sizeof(T));
        }
    }

    // ops
    void CopyToDevice(const TStream &stream, yint elemCount) { Mem.CopyToDevice(stream, elemCount * sizeof(T)); }
    void CopyToDevice(const TStream &stream) { CopyToDevice(stream, Size); }
    void CopyToHost(const TStream &stream, yint elemCount) { Mem.CopyToHost(stream, elemCount * sizeof(T)); }
    void CopyToHost(const TStream &stream) { CopyToHost(stream, Size); }
    void ClearDeviceMem(const TStream &stream) { Mem.ClearDeviceMem(stream); }
    void ClearHostMem() { Mem.ClearHostMem(); }
    void ClearDeviceMem() { Mem.ClearDeviceMem(); }

    // data access
    TMemoryBlob GetDeviceMem() const
    {
        return TMemoryBlob(Mem.GetDevicePtr(), Mem.GetSizeInBytes());
    }
    TMemoryBlob GetHostMem() const
    {
        return TMemoryBlob(Mem.GetHostPtr(), Mem.GetSizeInBytes());
    }
    TMemoryBlob GetMem(EMemType mt) const
    {
        return (mt == MT_HOST) ? GetHostMem() : GetDeviceMem();
    }
    TCuda1DPtr<T> GetDevicePtr() const
    {
        return TCuda1DPtr<T>(Mem.GetDevicePtr(), Size);
    }
    T* GetHostPtr() const
    {
        return (T*)Mem.GetHostPtr();
    }
    TCudaPOD<T> GetElement(yint idx) const
    {
        Y_ASSERT(idx >= 0 && idx < Size);
        return TCudaPOD<T>(this, GetMemPool(), ((T*)Mem.GetDevicePtr()) + idx);
    }

    // fragments
    TCudaVectorFragment<T> MakeFragment(yint offset, yint sz)
    {
        Y_ASSERT(offset + sz <= GetSize());
        char *pFragmentData = ((char *)Mem.GetDevicePtr()) + offset * sizeof(T);
        return TCudaVectorFragment<T>(this, GetMemPool(), sz, pFragmentData);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda 2D array
template<class T>
struct TCuda2DPtr
{
    ui8 *Data;
    int StrideInBytes; // int or size_t?
#ifndef NDEBUG
    int XSize, YSize;
#endif

    __host__ __device__ TCuda2DPtr(void *data, int strideInBytes, int xSize, int ySize) : Data((ui8*)data), StrideInBytes(strideInBytes)
    {
#ifndef NDEBUG
        XSize = xSize;
        YSize = ySize;
#endif
    }
    __forceinline __device__ ui8 *GetRawData() const { return Data; }
    __forceinline __device__ T *operator[](int y) const
    {
        CUDA_ASSERT(y >= 0 && y < YSize);
        return (T *)(Data + y * StrideInBytes);
    }
    __forceinline __device__ void CheckCoords(int x, int y) const
    {
        CUDA_ASSERT(y >= 0 && y < YSize && x >= 0 && x < XSize);
    }
    __forceinline __device__ TCuda2DPtr<T> Fragment(int xOffset, int yOffset) const
    {
#ifndef NDEBUG
        CUDA_ASSERT(xOffset < XSize && yOffset < YSize);
        return TCuda2DPtr<T>(Data + yOffset * StrideInBytes + xOffset * sizeof(T), StrideInBytes, XSize - xOffset, YSize - yOffset);
#else
        return TCuda2DPtr<T>(Data + yOffset * StrideInBytes + xOffset * sizeof(T), StrideInBytes, 0, 0);
#endif
    }
    __forceinline __host__ __device__ int GetStrideInBytes() const
    {
        return StrideInBytes;
    }
};


template<class T>
class TCuda2DArrayFragment
{
    const void *Owner; // pointer to owner of this pod (array/vector for example)
    TIntrusivePtr<TCudaMemoryPool> Pool;
    yint Stride = 0;
    yint XSize = 0;
    yint YSize = 0;
    void *DeviceData = 0;
public:
    typedef T TElem;

    TCuda2DArrayFragment() {}
    TCuda2DArrayFragment(const void *owner, TPtrArg<TCudaMemoryPool> pool, yint stride, yint xSize, yint ySize, void *pData)
        : Owner(owner), Pool(pool.Get()), Stride(stride), XSize(xSize), YSize(ySize), DeviceData(pData)
    {
    }
    TCuda2DPtr<T> GetDevicePtr() const
    {
        return TCuda2DPtr<T>(DeviceData, Stride, XSize, YSize);
    }
    TMemoryBlob GetDeviceMem() const { return TMemoryBlob(DeviceData, Stride, YSize); }
    const void *GetOwner() const { return Owner; }
    TPtrArg<TCudaMemoryPool> GetMemPool() const { return Pool; }
    yint GetXSize() const { return XSize; }
    yint GetYSize() const { return YSize; }
};


template<class T>
class TCuda2DArray : public TNonCopyable
{
    yint Stride = 0; // stride in bytes
    yint XSize = 0;
    yint YSize = 0;
    TCudaMemory Mem;

    yint SelectStride(yint widthInBytes)
    {
        if ((widthInBytes & (widthInBytes - 1)) == 0) {
            return widthInBytes; // pow2
        } else {
            if (widthInBytes < 128) {
                yint res = 1 * sizeof(T);
                while (res < widthInBytes) {
                    res *= 2;
                }
                return res;
            } else {
                return DivCeil(widthInBytes, 128) * 128;
            }
        }
    }
    void SetSizes(yint xSize, yint ySize)
    {
        XSize = xSize;
        YSize = ySize;
        Stride = SelectStride(xSize * sizeof(T));
    }

public:
    typedef T TElem;

    yint GetXSize() const { return XSize; }
    yint GetYSize() const { return YSize; }
    const void *GetOwner() const { return this; }
    bool IsEmpty() const { return YSize == 0; }

    // allocate
    void Allocate(yint xSize, yint ySize)
    {
        SetSizes(xSize, ySize);
        Mem.Allocate(YSize * Stride, TCudaMemory::CUDA_ALLOC, cudaHostAllocDefault);
    }
    void AllocateHost(yint xSize, yint ySize)
    {
        SetSizes(xSize, ySize);
        Mem.Allocate(YSize * Stride, TCudaMemory::CUDA_MAP, cudaHostAllocDefault);
    }
    void AllocateCuda(yint xSize, yint ySize)
    {
        SetSizes(xSize, ySize);
        Mem.Allocate(YSize * Stride, TCudaMemory::CUDA_ALLOC, -1);
    }
    void AllocateCuda(yint xSize, yint ySize, TPtrArg<TCudaMemoryPool> pool)
    {
        SetSizes(xSize, ySize);
        Mem.AllocateFromCudaPool(YSize * Stride, pool);
    }
    TPtrArg<TCudaMemoryPool> GetMemPool() const { return Mem.GetMemPool(); }

    // ops
    void CopyToDevice(const TStream &stream, yint ySize)
    {
        Y_ASSERT(ySize <= YSize);
        Mem.CopyToDevice(stream, ySize * Stride);
    }
    void CopyToDevice(const TStream &stream) { CopyToDevice(stream, YSize); }
    void CopyToHost(const TStream &stream, yint ySize)
    {
        Y_ASSERT(ySize <= YSize);
        Mem.CopyToHost(stream, ySize * Stride);
    }
    void CopyToHost(const TStream &stream) { CopyToHost(stream, YSize); }
    void ClearDeviceMem(const TStream &stream) { Mem.ClearDeviceMem(stream); }
    void ClearHostMem() { Mem.ClearHostMem(); }

    // data access
    TMemoryBlob GetDeviceMem() const { return TMemoryBlob(Mem.GetDevicePtr(), Stride, YSize); }
    TMemoryBlob GetHostMem() const { return TMemoryBlob(Mem.GetHostPtr(), Stride, YSize); }
    TMemoryBlob GetMem(EMemType mt) const { return (mt == MT_HOST) ? GetHostMem() : GetDeviceMem(); }
    TCuda2DPtr<T> GetDevicePtr() const { return TCuda2DPtr<T>(Mem.GetDevicePtr(), Stride, XSize, YSize); }
    THost2DPtr<T> GetHostPtr() const { return THost2DPtr<T>(Mem.GetHostPtr(), Stride, XSize, YSize); }
    bool IsHostMem() const { return Mem.GetHostPtr() != 0; }

    // fragments
    TCuda2DArrayFragment<T> MakeFragment(yint xOffset, yint xSize, yint yOffset, yint ySize)
    {
        Y_ASSERT(xOffset + xSize <= GetXSize());
        Y_ASSERT(yOffset + ySize <= GetYSize());
        char *pFragmentData = ((char *)Mem.GetDevicePtr()) + yOffset * Stride + xOffset * sizeof(T);
        return TCuda2DArrayFragment<T>(this, GetMemPool(), Stride, xSize, ySize, pFragmentData);
    }
    TCuda2DArrayFragment<T> MakeFragment(yint xOffset, yint yOffset)
    {
        Y_ASSERT(xOffset < GetXSize());
        Y_ASSERT(yOffset < GetYSize());
        char *pFragmentData = ((char *)Mem.GetDevicePtr()) + yOffset * Stride + xOffset * sizeof(T);
        return TCuda2DArrayFragment<T>(this, GetMemPool(), Stride, GetXSize() - xOffset, GetYSize() - yOffset, pFragmentData);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// span [beg; fin)
struct TCudaSpan
{
    int Beg = 0;
    int Fin = 0;

    TCudaSpan() {}
    TCudaSpan(int b, int f) : Beg(b), Fin(f) {}
    yint GetSize() const { return Fin - Beg; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// CudaVector data copy
template <class T>
void GetData(TCudaVector<T> &arr, TVector<T> *res, yint sz)
{
    Y_ASSERT(sz <= arr.GetSize());
    res->resize(sz);
    if (sz > 0) {
        TMemoryBlob hostMem = arr.GetHostMem();
        memcpy(res->data(), hostMem.Ptr, sizeof(T) * sz);
    }
}

template <class T>
void GetAllData(TCudaVector<T> &arr, TVector<T> *res)
{
    GetData(arr, res, arr.GetSize());
}

template <class T>
inline yint PutHost(TCudaVector<T> *arr, const TVector<T> &data)
{
    yint sz = YSize(data);
    Y_VERIFY(sz <= arr->GetSize());
    if (sz > 0) {
        TMemoryBlob hostMem = arr->GetHostMem();
        memcpy(hostMem.Ptr, &data[0], sz * sizeof(T));
    }
    return sz;
}

template <class T>
inline void Put(const TStream &stream, TCudaVector<T> *arr, const TVector<T> &data)
{
    yint sz = PutHost(arr, data);
    if (sz > 0) {
        arr->CopyToDevice(stream, sz);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Cuda2DArray data copy
template <class T>
inline void GetAllData(const TCuda2DArray<T> &arr, yint rowCount, TArray2D<T> *res)
{
    TMemoryBlob hostMem = arr.GetHostMem();
    yint xSize = arr.GetXSize();
    yint ySize = arr.GetYSize();
    Y_VERIFY(rowCount <= ySize);
    res->SetSizes(xSize, rowCount);
    for (yint y = 0; y < rowCount; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*res)[y][x] = *hostMem.GetElementAddress<T>(x, y);
        }
    }
}

template <class T>
inline void GetAllData(const TCuda2DArray<T> &arr, TArray2D<T> *res)
{
    GetAllData(arr, arr.GetYSize(), res);
}

template <class T>
inline void GetAllData(const TCuda2DArray<T> &arr, TVector<TVector<T>> *res)
{
    TMemoryBlob hostMem = arr.GetHostMem();
    yint xSize = arr.GetXSize();
    yint ySize = arr.GetYSize();
    res->resize(ySize);
    for (yint y = 0; y < ySize; ++y) {
        TVector<T> &dst = (*res)[y];
        dst.resize(xSize);
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = *hostMem.GetElementAddress<T>(x, y);
        }
    }
}

template <class T, class TMatrix2D>
inline void PutHost(TCuda2DArray<T> *arr, const TMatrix2D &data)
{
    TMemoryBlob hostMem = arr->GetHostMem();
    yint xSize = data.GetXSize();
    yint ySize = data.GetYSize();
    Y_ASSERT(ySize <= arr->GetYSize());
    Y_ASSERT(xSize <= arr->GetXSize());
    yint widthInBytes = xSize * sizeof(T);
    yint tail = hostMem.Stride - widthInBytes;
    Y_ASSERT(tail >= 0);
    if (widthInBytes == 0) {
        return;
    }
    for (yint y = 0; y < ySize; ++y) {
        char *dst = (char*)hostMem.GetElementAddress<T>(0, y);
        memcpy(dst, data.GetRow(y), widthInBytes);
        if (tail > 0) {
            // write combined memory works faster with fully written cache lines
            memset(dst + widthInBytes, 0, tail);
        }
    }
}

template <class T>
inline void Put(const TStream &stream, TCuda2DArray<T> *arr, const TArray2D<T> &data)
{
    PutHost(arr, data);
    arr->CopyToDevice(stream, data.GetYSize());
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// half floats data copy 
void GetAllData(const TCuda2DArray<half> &arr, TArray2D<float> *p);
void GetAllData(const TCuda2DArray<half> &arr, TVector<TVector<float>> *p);

void Put(TStream &stream, TCuda2DArray<half> *arr, const TArray2D<float> &src);
void Put(TStream &stream, TCudaVector<half> *arr, const TVector<float> &src);


///////////////////////////////////////////////////////////////////////////////////////////////////
// fill random data for tests
template <class TRng>
void FillRandom(TRng &rng, TStream &stream, TCuda2DArray<i8> *p)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    TArray2D<i8> data;
    data.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
#ifdef NDEBUG
            data[y][x] = 0; // measure peak performance, not limited by TDP
#else
            data[y][x] = (i8)rng.Uniform(256);
#endif
        }
    }
    Put(stream, p, data);
}


template <class TRng>
void FillRandom(TRng &rng, TStream &stream, TCuda2DArray<e4m3> *p)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    TArray2D<e4m3> data;
    data.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
#ifdef NDEBUG
            data[y][x].Data = 0; // measure peak performance, not limited by TDP
#else
            data[y][x].Data = (i8)(rng.Uniform(127) + rng.Uniform(2) * 128);
#endif
        }
    }
    Put(stream, p, data);
}


template <class TRng, class T>
void FillRandom(TRng &rng, TStream &stream, TCuda2DArray<T> *p)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    TArray2D<T> data;
    data.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
#ifdef NDEBUG
            data[y][x] = 0; // measure peak performance, not limited by TDP
#else
            data[y][x] = rng.GenRandReal3();
#endif
        }
    }
    Put(stream, p, data);
}

}
