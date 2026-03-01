#include <util/pch.h>
#define KERNEL_UNIT "multi_device_chk/"
#include "cuda_graph.cuh"
#include "multi_device_buf.h"
#include "vec_util.cuh"
#include <lib/hp_timer/hp_timer.h>


using namespace NCuda;

// asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];" : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(ptr) :
// "memory");

struct TNVLMulticastCtx
{
    struct TMemBlock
    {
        yint Size = 0;
        CUmemGenericAllocationHandle Mem = 0;
        CUdeviceptr Ptr = 0;

        void Allocate(yint deviceId, yint sz)
        {
            Size = sz;
            CUmemAllocationProp allocProp;
            Zero(allocProp);
            allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            allocProp.location.id = deviceId;
            allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            Y_VERIFY(CUDA_SUCCESS == cuMemCreate(&Mem, sz, &allocProp, 0));
        }

        void CreateMulticast(yint sz, yint deviceCount, size_t *pGranularity)
        {
            CUmulticastObjectProp mop;
            Zero(mop);
            mop.numDevices = deviceCount;
            mop.size = sz;

            size_t &granularity = *pGranularity;
            Y_VERIFY(CUDA_SUCCESS == cuMulticastGetGranularity(&granularity, &mop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

            Size = RoundUp(sz, granularity);
            mop.size = Size;
            Y_VERIFY(CUDA_SUCCESS == cuMulticastCreate(&Mem, &mop));

            for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                Y_VERIFY(CUDA_SUCCESS == cuMulticastAddDevice(Mem, deviceId));
            }
        }

        void MapObject(yint deviceCount, yint granularity)
        {
            Y_VERIFY(Ptr == 0);
            Y_VERIFY(CUDA_SUCCESS == cuMemAddressReserve(&Ptr, Size, granularity, 0, 0));
            yint bufOffset = 0;
            Y_VERIFY(CUDA_SUCCESS == cuMemMap(Ptr, Size, bufOffset, Mem, 0));
            TVector<CUmemAccessDesc> ad;
            for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                CUmemAccessDesc desc;
                desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                desc.location.id = deviceId;
                desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                ad.push_back(desc);
            }
            Y_VERIFY(CUDA_SUCCESS == cuMemSetAccess(Ptr, Size, ad.data(), deviceCount));
        }
        
        void Free()
        {
            Y_VERIFY(CUDA_SUCCESS == cuMemUnmap(Ptr, Size));
            Y_VERIFY(CUDA_SUCCESS == cuMemRelease(Mem));
        }
    };

    TVector<TMemBlock> DeviceMem;
    TMemBlock MultiMem;

public:
    void Create(yint sz, yint deviceCount)
    {
        DeviceMem.resize(deviceCount);
        size_t granularity = 0;
        MultiMem.CreateMulticast(sz, deviceCount, &granularity);

        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TMemBlock &blk = DeviceMem[deviceId];
            blk.Allocate(deviceId, MultiMem.Size);
            blk.MapObject(deviceCount, granularity);
            yint mcOffset = 0;
            yint memOffset = 0;
            Y_VERIFY(CUDA_SUCCESS == cuMulticastBindMem(MultiMem.Mem, mcOffset, DeviceMem[deviceId].Mem, memOffset, MultiMem.Size, 0));
            // cuMulticastUnbind
        }
        MultiMem.MapObject(deviceCount, granularity);
    }
    yint GetDeviceCount() const { return YSize(DeviceMem); }
};


constexpr yint NUM_DEVICES = 8;
struct TNVLptrArr
{
    int *PtrArr[NUM_DEVICES];
};

__device__ __forceinline__ void multimem_st_global(float *ptr, float val)
{
#if (__CUDA_ARCH__ >= 900)
    asm volatile("multimem.st.global.f32 [%0], %1;" ::"l"(ptr), "f"(val) : "memory");
#endif
}
__device__ __forceinline__ void multimem_st_global(float2 *ptr, float2 val)
{
#if (__CUDA_ARCH__ >= 900)
    asm volatile("multimem.st.global.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "f"(val.x), "f"(val.y) : "memory");
#endif
}
__device__ __forceinline__ void multimem_st_global(float4 *ptr, float4 val)
{
#if (__CUDA_ARCH__ >= 900)
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
#endif
}

template <class T>
__global__ void TestMultimem(int myDeviceId, int ySize, int strideInBytes, int lenTiles, T *src, T *dst)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    for (int row = warpId + blockIdx.x * blockDim.y; row < ySize; row += blockDim.y * gridDim.x) {
        int rowOffset = row * strideInBytes;
        for (int x = 0; x < lenTiles; ++x) {
            int tileId = myDeviceId * lenTiles + x;
            T vec = ((T*)(AdvancePtr(src, rowOffset) + tileId * WARP_VEC_DIM))[h];
            T *dstPtr = ((T*)(AdvancePtr(dst, rowOffset) + tileId * WARP_VEC_DIM)) + h;
            multimem_st_global(dstPtr, vec);
        }
    }
}

template <class T, class THold>
void MultimemGather(
    TNVLMulticastCtx *pCtx, TIntrusivePtr<TGraph> c, TIntrusivePtr<TMultiDevice2DArray<T>> p, yint deviceId)
{
    yint deviceCount = pCtx->GetDeviceCount();
    TCuda2DArray<T> &buf = p->GetData(deviceId);
    int lenTiles = buf.GetXSize() / deviceCount / WARP_VEC_DIM;
    int ySize = buf.GetYSize();
    int stride = buf.GetDeviceMem().Stride;

    constexpr yint smCount = 64;
    //pSync->Sync(c, deviceId, buf);
    CudaCall(c, TestMultimem<THold>)
        .Block(WARP_SIZE, 32)
        .Grid(smCount)
        .DepWrite(buf)
        .Read((int)deviceId, ySize, stride, lenTiles, pCtx->DeviceMem[deviceId].Ptr, pCtx->MultiMem.Ptr);
    //pSync->Sync(c, deviceId, buf);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void RunMultideviceTest()
{
    constexpr int NUM_DEVICES = 8;
    constexpr int REP_COUNT = 100;
    typedef float T;

    constexpr int XSIZE = 1024;
    constexpr int YSIZE = 8192;

    TIntrusivePtr<TMultiDeviceBufferFabric> fabric = new TMultiDeviceBufferFabric();
    TDeviceGroup deviceGroup(0, NUM_DEVICES);
    TIntrusivePtr<TMultiDeviceBuffers> multi = new TMultiDeviceBuffers(fabric, deviceGroup);
    TIntrusivePtr<TMultiDevice2DArray<T>> buf = multi->Fab().Create2DArray<T>("buf");
    TIntrusivePtr<TGraph> computerArr[NUM_DEVICES];
    TStream streamArr[NUM_DEVICES];

    // allocate
    for (yint deviceId = 0; deviceId < NUM_DEVICES; ++deviceId) {
        cudaSetDevice(deviceId);
        buf->AllocateCuda(deviceId, XSIZE, YSIZE, null_ptr_arg);
        multi->InitSync(deviceId);
    }

    // create run graphs
    for (yint deviceId = 0; deviceId < NUM_DEVICES; ++deviceId) {
        cudaSetDevice(deviceId);
        TIntrusivePtr<TGraph> &c = computerArr[deviceId];
        c = new TGraph;
        for (yint iter = 0; iter < REP_COUNT; ++iter) {
            multi->Op().AllGatherXSplit(c, buf, deviceId);
            // multi->Op().ReduceXSplit(c, buf, deviceId);
            // MultimemGather<float, float4>(&nvl, c, buf, deviceId);
        }
    }

    // measure
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        for (yint deviceId = 0; deviceId < NUM_DEVICES; ++deviceId) {
            cudaSetDevice(deviceId);
            computerArr[deviceId]->Run(streamArr[deviceId]);
        }
        for (yint deviceId = 0; deviceId < NUM_DEVICES; ++deviceId) {
            streamArr[deviceId].Sync();
        }
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double bw = XSIZE * YSIZE * sizeof(T) * REP_COUNT / tPassed / 1e9;
        DebugPrintf("%g GB/sec\n", bw);
    }
}
