#pragma once
#include <cuda_runtime.h>

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda stream
class TStream : public TNonCopyable
{
    cudaStream_t Stream;
public:
    TStream() { cudaStreamCreate(&Stream); }
    ~TStream() { cudaStreamDestroy(Stream); }
    void Sync() { cudaStreamSynchronize(Stream); }
    operator cudaStream_t() const { return Stream; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
char *CudaHugeAlloc(yint sz);
void CudaHugeFree(void *p);
char *CudaHugeHostAlloc(yint sz);
void CudaHugeHostFree(void *p);


///////////////////////////////////////////////////////////////////////////////////////////////////
// reusable cuda memory pool
class TCudaMemoryPool : public TThrRefBase
{
    TIntrusivePtr<TCudaMemoryPool> ParentSelect;
    TIntrusivePtr<TCudaMemoryPool> ParentTaken;
    TVector<TIntrusivePtr<TCudaMemoryPool>> SelectArr; // one is active
    TVector<TIntrusivePtr<TCudaMemoryPool>> TakenArr; // every can be active
    bool AsyncIO = false;
    yint TotalSize = 0;
    char *Ptr = 0;

    char *SetBaseDevicePtr(char *base)
    {
        char *mem = base;
        Ptr = mem;
        mem += TotalSize;
        for (yint k = 0; k < YSize(TakenArr); ++k) {
            mem = TakenArr[k]->SetBaseDevicePtr(mem);
        }
        char *res = mem;
        for (yint k = 0; k < YSize(SelectArr); ++k) {
            char *next = SelectArr[k]->SetBaseDevicePtr(mem);
            res = Max(res, next);
        }
        return res;
    }

    void Reset()
    {
        *this = TCudaMemoryPool();
    }

    TCudaMemoryPool *GetParent() const
    {
        return ParentSelect.Get() ? ParentSelect.Get() : ParentTaken.Get();
    }

public:
    yint Allocate(yint sz)
    {
        Y_VERIFY(Ptr == 0 && "can not allocate after memory buffer is assigned");
        const yint ROUND_SIZE = 512;
        yint res = TotalSize;
        yint szRound = RoundUp(sz, ROUND_SIZE);
        TotalSize += szRound;
        return res;
    }

    yint GetMemSize()
    {
        yint fixed = TotalSize;
        for (yint k = 0; k < YSize(TakenArr); ++k) {
            fixed += TakenArr[k]->GetMemSize();
        }
        yint maxShared = 0;
        for (yint k = 0; k < YSize(SelectArr); ++k) {
            maxShared = Max(maxShared, SelectArr[k]->GetMemSize());
        }
        return fixed + maxShared;
    }

    void *GetDevicePtr(yint offset) const
    {
        Y_VERIFY(Ptr != 0);
        return Ptr + offset;
    }

    void UseForAsyncIO(bool b) { AsyncIO = b; }
    bool IsUsedForAsyncIO() const { return AsyncIO; }

    friend class TCudaMemoryAllocator;
    friend class TCudaActivePoolTracker;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaMemoryAllocator : public TThrRefBase
{
private:
    TIntrusivePtr<TCudaMemoryPool> RootPool;
    TVector<TIntrusivePtr<TCudaMemoryPool>> AllPools;
    void *DeviceBuf = 0;

    ~TCudaMemoryAllocator()
    {
        for (auto &p : AllPools) {
            p->Reset(); // clear all pointers to avoid circular references
        }
        if (DeviceBuf) {
            CudaHugeFree(DeviceBuf);
            DeviceBuf = 0;
        }
    }
public:
    TCudaMemoryAllocator()
    {
        RootPool = new TCudaMemoryPool;
        AllPools.push_back(RootPool);
    }
    TIntrusivePtr<TCudaMemoryPool> CreatePool(TPtrArg<TCudaMemoryPool> parent)
    {
        TIntrusivePtr<TCudaMemoryPool> p = new TCudaMemoryPool;
        p->ParentSelect = parent.Get();
        parent->SelectArr.push_back(p);
        AllPools.push_back(p);
        return p;
    }
    TIntrusivePtr<TCudaMemoryPool> CreateNonsharedPool(TPtrArg<TCudaMemoryPool> parent)
    {
        TIntrusivePtr<TCudaMemoryPool> p = new TCudaMemoryPool;
        p->ParentTaken = parent.Get();
        parent->TakenArr.push_back(p);
        AllPools.push_back(p);
        return p;
    }
    TIntrusivePtr<TCudaMemoryPool> CreatePool() { return CreatePool(RootPool); }
    TIntrusivePtr<TCudaMemoryPool> CreateNonsharedPool() { return CreateNonsharedPool(RootPool); }
    void AllocateMemory()
    {
        Y_VERIFY(DeviceBuf == 0 && "can allocate only once");
        yint maxSize = RootPool->GetMemSize();
        if (maxSize > 0) {
            void *ptr = CudaHugeAlloc(maxSize);
            RootPool->SetBaseDevicePtr((char*)ptr);
            DeviceBuf = ptr;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaActivePoolTracker
{
public:
    typedef THashMap<TCudaMemoryPool*, bool, TPtrHash> TPools;
private:
    THashMap<TIntrusivePtr<TCudaMemoryPool>, bool> Active;
    THashMap<TIntrusivePtr<TCudaMemoryPool>, TIntrusivePtr<TCudaMemoryPool>> Selected; // parent -> selected pool

    void Deactivate(TCudaMemoryPool *p, TPools *pDel)
    {
        auto tt = Active.find(p);
        if (tt != Active.end()) {
            for (TIntrusivePtr<TCudaMemoryPool> &taken : p->TakenArr) {
                Deactivate(taken.Get(), pDel);
            }
            Unselect(p, pDel);
            (*pDel)[p];
            Active.erase(tt);
        }
    }

    void Unselect(TCudaMemoryPool *parent, TPools *pDel)
    {
        auto it = Selected.find(parent);
        if (it != Selected.end()) {
            TCudaMemoryPool *p = it->second.Get();
            Y_ASSERT(Active.find(p) != Active.end());
            Deactivate(p, pDel);
            Selected.erase(it);
        }
    }

public:
    void ActivatePool(TCudaMemoryPool *pool, TPools *pAdd, TPools *pDel, TPools *pRequiredPools)
    {
        for (TCudaMemoryPool *p = pool; p;) {
            if (Active.find(p) != Active.end()) {
                break;
            }
            (*pAdd)[p];
            Active[p];
            if (p->ParentSelect.Get()) {
                TCudaMemoryPool *parent = p->ParentSelect.Get();
                Unselect(parent, pDel);
                Selected[parent] = p;
                p = parent;
            } else if (p->ParentTaken.Get()) {
                p = p->ParentTaken.Get();
            } else {
                p = nullptr;
            }
        }
        // walk up to the root
        for (TCudaMemoryPool *p = pool->GetParent(); p; p = p->GetParent()) {
            (*pRequiredPools)[p];
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaMemory : public TNonCopyable
{
    yint SizeInBytes = 0;
    char *HostBuf = 0;
    void *DeviceBuf = 0;
    void *DeviceData = 0;
    TIntrusivePtr<TCudaMemoryPool> Pool;
    yint PoolOffset = 0;

public:
    enum {
        CUDA_ALLOC,
        CUDA_MAP,
    };

private:
    void Free()
    {
        if (HostBuf) {
            CudaHugeHostFree(HostBuf);
            HostBuf = 0;
        }
        if (DeviceBuf) {
            CudaHugeFree(DeviceBuf);
            DeviceBuf = 0;
        }
        DeviceData = 0;
        Pool = 0;
    }
    void *GetDeviceBuf() const
    {
        if (DeviceBuf) {
            return DeviceBuf;
        }
        if (Pool.Get()) {
            return Pool->GetDevicePtr(PoolOffset);
        }
        return nullptr;
    }
    void *GetDeviceData() const
    {
        if (DeviceData) {
            return DeviceData;
        }
        if (Pool.Get()) {
            return Pool->GetDevicePtr(PoolOffset);
        }
        return nullptr;
    }

public:
    ~TCudaMemory()
    {
        Free();
    }
    yint GetSizeInBytes() const
    {
        return SizeInBytes;
    }

    // allocate
    void Allocate(yint sizeInBytesArg, int deviceAlloc, int hostAllocFlag)
    {
        Free();
        SizeInBytes = RoundUp(sizeInBytesArg, 128);
        Y_VERIFY(SizeInBytes <= 0x80000000ll); // we are using int offsets in kernels for perf
        if (deviceAlloc == CUDA_ALLOC) {
            DeviceBuf = CudaHugeAlloc(SizeInBytes);
            DeviceData = DeviceBuf;
        }
        if (hostAllocFlag != -1) {
            Y_ASSERT(hostAllocFlag == cudaHostAllocDefault);
            HostBuf = CudaHugeHostAlloc(SizeInBytes);
            if (DeviceData == 0) {
                Y_ASSERT(deviceAlloc == CUDA_MAP);
                DeviceData = HostBuf; // assume UVA
                //Y_VERIFY(cudaHostGetDevicePointer(&DeviceData, HostBuf, 0) == cudaSuccess);
            }
        }
    }
    void AllocateFromCudaPool(yint sizeInBytesArg, TPtrArg<TCudaMemoryPool> pool)
    {
        if (pool.Get() == nullptr) {
            Allocate(sizeInBytesArg, CUDA_ALLOC, -1);
            return;
        }
        Free();
        SizeInBytes = RoundUp(sizeInBytesArg, 128);
        Y_VERIFY(SizeInBytes <= 0x80000000ll); // we are using int offsets for perf
        Pool = pool.Get();
        PoolOffset = pool->Allocate(SizeInBytes);
    }
    TPtrArg<TCudaMemoryPool> GetMemPool() const { return Pool; }

    // mem ops
    void CopyToHost(const TStream &stream, yint sizeInBytes)
    {
        Y_ASSERT(sizeInBytes <= SizeInBytes);
        Y_ASSERT(HostBuf != 0);
        void *deviceBuf = GetDeviceBuf();
        if (deviceBuf) {
            cudaMemcpyAsync(HostBuf, deviceBuf, sizeInBytes, cudaMemcpyDeviceToHost, stream);
        } else {
            Y_VERIFY(0);
        }
    }
    void CopyToDevice(const TStream &stream, yint sizeInBytes)
    {
        Y_ASSERT(sizeInBytes <= SizeInBytes);
        Y_ASSERT(HostBuf != 0);
        void *deviceBuf = GetDeviceBuf();
        if (deviceBuf) {
            cudaMemcpyAsync(deviceBuf, HostBuf, sizeInBytes, cudaMemcpyHostToDevice, stream);
        } else {
            Y_VERIFY(0);
        }
    }
    void CopyToDevice(const void *data, yint sizeInBytes)
    {
        Y_ASSERT(sizeInBytes <= SizeInBytes);
        cudaMemcpy(GetDeviceBuf(), data, sizeInBytes, cudaMemcpyHostToDevice);
    }
    void ClearHostMem() { memset(HostBuf, 0, SizeInBytes); }
    void ClearDeviceMem()
    {
        Y_VERIFY(cudaMemset(GetDeviceBuf(), 0, SizeInBytes) == cudaSuccess);
    }
    void ClearDeviceMem(const TStream &stream)
    {
        Y_VERIFY(cudaMemsetAsync(GetDeviceBuf(), 0, SizeInBytes, stream) == cudaSuccess);
    }

    // get mem address
    void *GetDevicePtr() const
    {
        return GetDeviceData();
    }
    void *GetHostPtr() const
    {
        return HostBuf;
    }
};

}
