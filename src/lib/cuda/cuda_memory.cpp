#include "cuda_memory.h"
#include "cuda_init.h"
#include <cuda.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint PAGE_SIZE = 1 << 21;

class TDeviceMemoryBlock : public TThrRefBase
{
    CUmemGenericAllocationHandle Mem = 0;
    char *Ptr = 0;
    yint Size = 0;

    ~TDeviceMemoryBlock()
    {
        Y_VERIFY(CUDA_SUCCESS == cuMemUnmap(Ptr - (char *)0, Size));
        Y_VERIFY(CUDA_SUCCESS == cuMemRelease(Mem));
    }

public:
    TDeviceMemoryBlock(yint deviceId, yint deviceCount, yint sz) : Size(sz)
    {
        CUmemAllocationProp allocProp;
        Zero(allocProp);
        if (deviceId >= 0) {
            allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            allocProp.location.id = deviceId;
        } else {
            allocProp.location.type = CU_MEM_LOCATION_TYPE_HOST;
        }
        allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        size_t granularity = 0;
        Y_VERIFY(CUDA_SUCCESS == cuMemGetAllocationGranularity(&granularity, &allocProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        Y_VERIFY(granularity == PAGE_SIZE);
        // create
        Y_VERIFY(CUDA_SUCCESS == cuMemCreate(&Mem, Size, &allocProp, 0));
        // address reserve
        CUdeviceptr cuPtr = 0;
        Y_VERIFY(CUDA_SUCCESS == cuMemAddressReserve(&cuPtr, Size, PAGE_SIZE, 0, 0));
        // map
        Y_VERIFY(CUDA_SUCCESS == cuMemMap(cuPtr, Size, 0, Mem, 0));
        // give access
        TVector<CUmemAccessDesc> ad;
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            CUmemAccessDesc desc;
            desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            desc.location.id = deviceId;
            desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            ad.push_back(desc);
        }
        if (deviceId < 0) {
            CUmemAccessDesc desc;
            desc.location.type = CU_MEM_LOCATION_TYPE_HOST;
            desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            ad.push_back(desc);
        }
        Y_VERIFY(CUDA_SUCCESS == cuMemSetAccess(cuPtr, Size, ad.data(), YSize(ad)));
        Ptr = ((char *)0) + cuPtr;
    }

    char *GetPtr() const { return Ptr; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint LARGE_BLOCK_SIZE = 256 * (1 << 20);

typedef THashMap<void *, TIntrusivePtr<TDeviceMemoryBlock>, TPtrHash> TAllBlocksHash;

class TDeviceMemoryAllocator
{
    yint DeviceId = 0;
    yint DeviceCount = 0;
    TIntrusivePtr<TDeviceMemoryBlock> CurBlock;
    yint CurOffset = 0;

public:
    void Init(yint deviceId, yint deviceCount)
    {
        DeviceId = deviceId;
        DeviceCount = deviceCount;
    }

    char *Alloc(yint sz, TAllBlocksHash &allBlocks)
    {
        Y_VERIFY(DeviceCount != 0);
        if (sz > LARGE_BLOCK_SIZE / 8) {
            TIntrusivePtr<TDeviceMemoryBlock> blk = new TDeviceMemoryBlock(DeviceId, DeviceCount, RoundUp(sz, PAGE_SIZE));
            char *res = blk->GetPtr();
            allBlocks[res] = blk;
            return res;
        }
        if (CurBlock.Get() == 0 || CurOffset + sz > LARGE_BLOCK_SIZE) {
            CurBlock = new TDeviceMemoryBlock(DeviceId, DeviceCount, LARGE_BLOCK_SIZE);
            CurOffset = 0;
        }
        char *res = CurBlock->GetPtr() + CurOffset;
        allBlocks[res] = CurBlock;
        CurOffset += RoundUp(sz, 128);
        return res;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TGlobalCudaAllocator
{
    TAllBlocksHash AllBlocks;
    TDeviceMemoryAllocator DevArr[MAX_NUM_DEVICES];
    TDeviceMemoryAllocator Host;

    TGlobalCudaAllocator()
    {
        yint deviceCount = GetCudaDeviceCount();
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            DevArr[deviceId].Init(deviceId, deviceCount);
        }
        Host.Init(-1, deviceCount);
    }

    char *Alloc(yint deviceId, yint sz)
    {
        if (deviceId < 0) {
            return Host.Alloc(sz, AllBlocks);
        } else {
            return DevArr[deviceId].Alloc(sz, AllBlocks);
        }
    }

    void Free(void *p)
    {
        auto it = AllBlocks.find(p);
        Y_VERIFY(it != AllBlocks.end());
        AllBlocks.erase(it);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
static TAtomic CudaAllocLock;
static TGlobalCudaAllocator *GlobalAllocator;

static void InitAllocator()
{
    if (GlobalAllocator == 0) {
        GlobalAllocator = new TGlobalCudaAllocator;
    }
}

char *CudaHugeAlloc(yint sz)
{
    char *p = 0;
    Y_VERIFY(cudaSuccess == cudaMalloc(&p, sz));
    return p;
    // TGuard<TAtomic> lock(CudaAllocLock);
    // InitAllocator();
    // return GlobalAllocator->Alloc(CudaGetDevice(), sz);
}


void CudaHugeFree(void *p)
{
    Y_VERIFY(cudaSuccess == cudaFree(p));
    // TGuard<TAtomic> lock(CudaAllocLock);
    // GlobalAllocator->Free(p);
}


char *CudaHugeHostAlloc(yint sz)
{
    TGuard<TAtomic> lock(CudaAllocLock);
    InitAllocator();
    return GlobalAllocator->Alloc(-1, sz);
}


void CudaHugeHostFree(void *p)
{
    GlobalAllocator->Free(p);
}
}
