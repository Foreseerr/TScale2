#include <util/pch.h>
#define KERNEL_UNIT "multi_device_buf/"
#include "cuda_graph.cuh"
#include "cuda_init.h"
#include "cuda_util.cuh"
#include "multi_device_buf.h"
#include "vec_util.cuh"
#include <lib/cuda/cuda_init.h>
#include <lib/hp_timer/hp_timer.h>


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TMultiGPUptr
{
    T *PtrArr[MAX_NUM_DEVICES];
};


struct TSyncDeviceState
{
    ui32 Data[32]; // 128b cacheline
};


template <class TBuf>
static TMultiGPUptr<typename TBuf::TElem> MakeMultiPtr(TPtrArg<TBuf> p, const TDeviceGroup &deviceGroup)
{
    typedef typename TBuf::TElem T;
    yint dgSize = deviceGroup.GetSize();
    Y_VERIFY(dgSize <= MAX_NUM_DEVICES);
    TMultiGPUptr<T> stateArr;
    for (yint dgRank = 0; dgRank < dgSize; ++dgRank) {
        stateArr.PtrArr[dgRank] = (T *)p->GetData(deviceGroup.DeviceId(dgRank)).GetDevicePtr().Data;
    }
    return stateArr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void MultiGPUsyncKernel(int myDgRank, int numDevices, TMultiGPUptr<TSyncDeviceState> stateArr)
{
    int h = threadIdx.x;
    // increment our sync id and copy it to all devices
    ui32 syncId = stateArr.PtrArr[myDgRank][myDgRank].Data[h];
    ++syncId;
    for (int k = 0; k < numDevices; ++k) {
        stateArr.PtrArr[k][myDgRank].Data[h] = syncId;
    }
    __syncwarp();
    __threadfence_system();

    // wait all devices
    TSyncDeviceState *localState = stateArr.PtrArr[myDgRank];
    for (bool ok = false; !ok;) {
        ok = true;
        for (int k = 0; k < numDevices; ++k) {
            ui32 val;
            asm("ld.volatile.global.u32 %0 ,[%1];" : "=r"(val) : "l"(&localState[k].Data[h]));
            ok &= (val >= syncId);
        }
    }
}

__global__ void NopKernel()
{
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMultiGPUSync : public TThrRefBase
{
    struct TDeviceCtx : public TThrRefBase
    {
        TCudaVector<TSyncDeviceState> AllStates;
        yint SyncArrPtr = 0;

        void Allocate()
        {
            AllStates.AllocateCuda(MAX_NUM_DEVICES);
            AllStates.ClearDeviceMem();
        }
    };

private:
    TDeviceGroup DeviceGroup;
    TVector<TIntrusivePtr<TDeviceCtx>> DeviceArr;
    TVector<TIntrusivePtr<TKernelOp>> SimulatedSyncOpArr;

public:
    TMultiGPUSync(const TDeviceGroup &deviceGroup) : DeviceGroup(deviceGroup)
    {
        yint dgSize = deviceGroup.GetSize();
        Y_VERIFY(dgSize <= MAX_NUM_DEVICES);
        if (dgSize > 1) {
            DeviceArr.resize(dgSize);
            for (yint dgRank = 0; dgRank < dgSize; ++dgRank) {
                DeviceArr[dgRank] = new TDeviceCtx;
            }
        }
        CudaEnablePeerAccess();
    }

    const TDeviceGroup &GetDeviceGroup() const { return DeviceGroup; }

    void InitSync(yint deviceId)
    {
        if (NeedSync()) {
            yint dgRank = DeviceGroup.Rank(deviceId);
            DeviceArr[dgRank]->Allocate();
        }
    }

    bool NeedSync() const { return DeviceGroup.GetSize() > 1; }

    template <class T>
    void Sync(TPtrArg<TGraph> c, yint deviceId, T &buf)
    {
        Y_VERIFY(NeedSync());
        Y_VERIFY(DeviceArr[0]->AllStates.GetSize() > 0 && "sync is not initialized");
        yint dgRank = DeviceGroup.Rank(deviceId);
        TIntrusivePtr<TKernelOp> op;
        if (SIMULATE_MULTI_GPU) {
            // syncOp can not depend on buf, otherwise non-existing write deps arise
            TKernelOp &pre = CudaCall(c, NopKernel).DepWrite(buf);
            if (dgRank == 0) {
                SimulatedSyncOpArr.push_back(&CudaCall(c, NopKernel));
            }
            TKernelOp &syncOp = *SimulatedSyncOpArr[DeviceArr[dgRank]->SyncArrPtr++];
            syncOp.Dep(pre);
            TKernelOp &post = CudaCall(c, NopKernel).Dep(syncOp).DepWrite(buf);
        } else {
            int dgSize = YSize(DeviceArr);
            TMultiGPUptr<TSyncDeviceState> ptrs;
            for (yint k = 0; k < dgSize; ++k) {
                ptrs.PtrArr[k] = DeviceArr[k]->AllStates.GetDevicePtr().Data;
            }
            op = &CudaCall(c, MultiGPUsyncKernel).Read((int)dgRank, (int)dgSize, ptrs);
            op->DepWrite(DeviceArr[dgRank]->AllStates);
            op->DepWrite(buf);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMultiDeviceOpCtx
{
    TIntrusivePtr<TMultiGPUSync> Sync;
    TDeviceGroup DeviceGroup;
    yint SMCount = 1;
    int dgRank = 0; // int to pass to kenels as parameter
    int dgSize = 0; // int to pass to kenels as parameter

    TMultiDeviceOpCtx(TPtrArg<TGraph> c, EReducerPriority pr, TPtrArg<TMultiGPUSync> pSync, int deviceId) : Sync(pSync)
    {
        SMCount = (pr == RP_LOW_PRIORITY) ? BG_SM_COUNT : Min<yint>(64, c->GetFullGridSize());
        Sync = pSync.Get();
        DeviceGroup = Sync->GetDeviceGroup();
        dgRank = DeviceGroup.Rank(deviceId);
        dgSize = DeviceGroup.GetSize();
    }
    
    bool NeedSync() const { return Sync->NeedSync(); }
    
    template <class TBuf>
    TMultiGPUptr<typename TBuf::TElem> MakeMultiPtr(TPtrArg<TBuf> p)
    {
        return NCuda::MakeMultiPtr(p, DeviceGroup);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
TMultiDeviceReducer::TMultiDeviceReducer(const TDeviceGroup &deviceGroup, EReducerPriority pr) : DeviceGroup(deviceGroup), Priority(pr)
{
    Sync = new TMultiGPUSync(deviceGroup);
}

TMultiDeviceReducer::~TMultiDeviceReducer() {}

void TMultiDeviceReducer::InitSync(yint deviceId)
{
    Sync->InitSync(deviceId);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void AllReduceKernel(int myDgRank, int dgSize, int xSizeTiles, int ySize, TMultiGPUptr<T> stateArr)
{
    int warpId = threadIdx.y;
    int blk = (warpId + blockIdx.x * blockDim.y) * dgSize + myDgRank;
    int step = blockDim.y * gridDim.x * dgSize;
    int lenTiles = xSizeTiles * ySize;
    for (int tileId = blk; tileId < lenTiles; tileId += step) {
        float4 vec[MAX_NUM_DEVICES];
        for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
            vec[dgRank] = LoadWarpVec(stateArr.PtrArr[dgRank] + tileId * WARP_VEC_DIM);
        }
        float4 sum = ZeroWarpVec();
        for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
            sum = sum + vec[dgRank];
        }
        for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
            StoreWarpVec(stateArr.PtrArr[dgRank] + tileId * WARP_VEC_DIM, sum);
        }
    }
    __threadfence_system(); // ensure global visibility
}


template <class TBuf, class TYSize>
static void AllReduceImpl(TPtrArg<TGraph> c, TMultiDeviceOpCtx &ctx, TPtrArg<TBuf> p, TYSize &&ySize, yint deviceId)
{
    typedef typename TBuf::TElem T;
    if (!ctx.NeedSync()) {
        return;
    }

    auto &buf = p->GetData(deviceId);
    TMultiGPUptr<T> stateArr = ctx.MakeMultiPtr(p);
    int xSizeBytes = buf.GetDeviceMem().Stride;
    Y_VERIFY(xSizeBytes % (WARP_VEC_DIM * sizeof(T)) == 0);
    int xSizeTiles = xSizeBytes / sizeof(T) / WARP_VEC_DIM;

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, AllReduceKernel<T>)
        .Block(WARP_SIZE, 32)
        .Grid(ctx.SMCount)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, xSizeTiles, ySize, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


void TMultiDeviceReducer::AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllReduceImpl(c, ctx, p, 1, deviceId);
}


void TMultiDeviceReducer::AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    int ySize = p->GetData(deviceId).GetYSize();
    AllReduceImpl(c, ctx, p, ySize, deviceId);
}


void TMultiDeviceReducer::AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllReduceImpl(c, ctx, p, ySize, deviceId);
}

void TMultiDeviceReducer::AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllReduceImpl(c, ctx, p, ySize, deviceId);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void AllMaxKernel(int myDgRank, int dgSize, int lenTiles, TMultiGPUptr<float> stateArr)
{
    for (int tileId = myDgRank; tileId < lenTiles; tileId += dgSize) {
        float4 vec[MAX_NUM_DEVICES];
        for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
            vec[dgRank] = LoadWarpVec(stateArr.PtrArr[dgRank] + tileId * WARP_VEC_DIM);
        }
        float4 vecMax = vec[0];
        for (int dgRank = 1; dgRank < dgSize; ++dgRank) {
            vecMax = Max(vecMax, vec[dgRank]);
        }
        for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
            StoreWarpVec(stateArr.PtrArr[dgRank] + tileId * WARP_VEC_DIM, vecMax);
        }
    }
    __threadfence_system(); // ensure global visibility
}


void TMultiDeviceReducer::AllMax(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    if (!ctx.NeedSync()) {
        return;
    }

    TCudaVector<float> &buf = p->GetData(deviceId);
    TMultiGPUptr<float> stateArr = ctx.MakeMultiPtr(p);
    int lenTiles = buf.GetSize() / WARP_VEC_DIM;
    Y_VERIFY(lenTiles * WARP_VEC_DIM == buf.GetSize());

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, AllMaxKernel).DepWrite(buf).Read(ctx.dgRank, ctx.dgSize, lenTiles, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void ReduceRows2DKernel(int myDgRank, int dgSize, int ySize, int strideInBytes, int lenTiles, TMultiGPUptr<T> stateArr)
{
    int warpId = threadIdx.y;
    for (int row = warpId + blockIdx.x * blockDim.y; row < ySize; row += blockDim.y * gridDim.x) {
        int rowOffset = row * strideInBytes;
        for (int x = 0; x < lenTiles; ++x) {
            int tileId = myDgRank * lenTiles + x;
            float4 vec[MAX_NUM_DEVICES];
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                vec[dgRank] = LoadWarpVec(AdvancePtr(stateArr.PtrArr[dgRank], rowOffset) + tileId * WARP_VEC_DIM);
            }
            float4 sum = ZeroWarpVec();
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                sum = sum + vec[dgRank];
            }
            StoreWarpVec(AdvancePtr(stateArr.PtrArr[myDgRank], rowOffset) + tileId * WARP_VEC_DIM, sum);
        }
    }
}


template <class T, class TYSize>
void ReduceRowsXSplitImpl(TPtrArg<TGraph> c, TMultiDeviceOpCtx &ctx, TPtrArg<TMultiDevice2DArray<T>> p, TYSize &&ySize, yint deviceId)
{
    if (!ctx.NeedSync()) {
        return;
    }

    TCuda2DArray<T> &buf = p->GetData(deviceId);
    TMultiGPUptr<T> stateArr = ctx.MakeMultiPtr(p);
    int lenTiles = buf.GetXSize() / ctx.dgSize / WARP_VEC_DIM;
    int stride = buf.GetDeviceMem().Stride;
    Y_VERIFY(lenTiles * ctx.dgSize * WARP_VEC_DIM == buf.GetXSize());

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, ReduceRows2DKernel<T>)
        .Block(WARP_SIZE, 32)
        .Grid(ctx.SMCount)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, ySize, stride, lenTiles, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


void TMultiDeviceReducer::ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceRowsXSplitImpl(c, ctx, p, p->GetData(deviceId).GetYSize(), deviceId);
}

void TMultiDeviceReducer::ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceRowsXSplitImpl(c, ctx, p, p->GetData(deviceId).GetYSize(), deviceId);
}

void TMultiDeviceReducer::ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceRowsXSplitImpl(c, ctx, p, ySize, deviceId);
}

void TMultiDeviceReducer::ReduceXSplit(
    TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceRowsXSplitImpl(c, ctx, p, ySize, deviceId);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void ReduceCompressedRows2DKernel(int myDgRank, int dgSize, int ySize, int srcStride, int lenTiles,
    TCuda2DPtr<float> gatherTileScale, TMultiGPUptr<i8> srcArr, TCuda2DPtr<T> dst)
{
    int warpId = threadIdx.y;
    for (int row = warpId + blockIdx.x * blockDim.y; row < ySize; row += blockDim.y * gridDim.x) {
        for (int x = 0; x < lenTiles; ++x) {
            int dataTileId = myDgRank * lenTiles + x;
            float4 vec[MAX_NUM_DEVICES];
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                vec[dgRank] = LoadWarpVec(AdvancePtr(srcArr.PtrArr[dgRank], row * srcStride) + dataTileId * WARP_VEC_DIM);
            }
            float4 sum = ZeroWarpVec();
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                int scaleTileId = dgRank * lenTiles + x;
                sum = sum + Scale(vec[dgRank], gatherTileScale[scaleTileId][row]);
            }
            StoreWarpVec(dst[row] + dataTileId * WARP_VEC_DIM, sum);
        }
    }
}

constexpr int COMPRESS_BATCH = 32;
template <class T>
__global__ void CompressRowsKernel(TCuda2DPtr<T> src, TCuda2DPtr<i8> dst, TCuda2DPtr<float> tileScale)
{
    int h = threadIdx.x;
    int tileId = blockIdx.x;
    int rowBase = blockIdx.y * COMPRESS_BATCH;

    float4 vecArr[COMPRESS_BATCH];
    for (int k = 0; k < COMPRESS_BATCH; ++k) {
        int row = rowBase + k;
        vecArr[k] = LoadWarpVec(src[row] + tileId * WARP_VEC_DIM);
    }
    __shared__ float resTileScale[COMPRESS_BATCH];
    for (int k = 0; k < COMPRESS_BATCH; ++k) {
        int row = rowBase + k;
        float4 vec = vecArr[k];
        float maxVal = CalcWarpVecMaxAbsValue(vec);
        float discrScale = 0;
        if (maxVal > 0) {
            discrScale = GetMaxDiscrScale(maxVal, (i8 *)0);
            vec = Scale(vec, 1 / discrScale);
        }
        StoreWarpVec(dst[row] + tileId * WARP_VEC_DIM, vec);
        if (h == 0) {
            resTileScale[k] = discrScale;
        }
    }
    if (h < COMPRESS_BATCH) {
        tileScale[tileId][rowBase + h] = resTileScale[h];
    }
}

__global__ void GatherCompressedTileScaleKernel(
    int myDgRank, int dgSize, int lenTiles, TMultiGPUptr<float> allTileScale, TCuda2DPtr<float> gatherTileScale)
{
    int x = blockIdx.x;
    int rowBase = blockIdx.y * WARP_VEC_DIM;
    int stride = gatherTileScale.GetStrideInBytes();
    for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
        int srcTileId = myDgRank * lenTiles + x;
        int dstTileId = dgRank * lenTiles + x;
        float *src = AdvancePtr(allTileScale.PtrArr[dgRank], stride * srcTileId);
        float4 vec = LoadWarpVec(src + rowBase);
        StoreWarpVec(gatherTileScale[dstTileId] + rowBase, vec);
    }
}


template <class T, class TYSize>
void ReduceCompressedRowsXSplitImpl(
    TPtrArg<TGraph> c, TMultiDeviceOpCtx &ctx, TPtrArg<TMultiDeviceFRed2DArray<T>> p, TYSize &&ySize, yint deviceId)
{
    if (!ctx.NeedSync()) {
        return;
    }

    TCuda2DArray<T> &buf = p->GetData(deviceId);
    TCuda2DArray<i8> &buf8 = p->GetPackedData(deviceId);
    TCuda2DArray<float> &tileScale = p->GetTileScale(deviceId);
    TCuda2DArray<float> &gatherTileScale = p->GetGatherTileScale(deviceId);
    int lenTiles = buf.GetXSize() / ctx.dgSize / WARP_VEC_DIM;
    int stride8 = buf8.GetDeviceMem().Stride;
    Y_VERIFY(lenTiles * ctx.dgSize * WARP_VEC_DIM == buf.GetXSize());
    Y_VERIFY(TMultiDeviceFRed2DArray<T>::TILE_SIZE == WARP_VEC_DIM);

    // compress
    CudaCall(c, CompressRowsKernel<T>)
        .Grid(buf.GetXSize() / WARP_VEC_DIM, DivCeil(ySize, COMPRESS_BATCH))
        .DepWrite(buf) // for Sync() dependency
        .Read(buf)
        .Write(&buf8, &tileScale);
    ctx.Sync->Sync(c, deviceId, buf);

    // collect tile scale from all devices
    TMultiGPUptr<float> allTileScale;
    for (yint k = 0; k < ctx.dgSize; ++k) {
        allTileScale.PtrArr[k] = (float *)p->GetTileScale(ctx.DeviceGroup.DeviceId(k)).GetDevicePtr().Data;
    }
    CudaCall(c, GatherCompressedTileScaleKernel)
        .Grid(lenTiles, DivCeil(ySize, WARP_VEC_DIM))
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, lenTiles, allTileScale)
        .Write(&gatherTileScale);

    // reduce
    TMultiGPUptr<i8> allPacked;
    for (yint k = 0; k < ctx.dgSize; ++k) {
        allPacked.PtrArr[k] = (i8*)p->GetPackedData(ctx.DeviceGroup.DeviceId(k)).GetDevicePtr().Data;
    }
    CudaCall(c, ReduceCompressedRows2DKernel<T>)
        .Block(WARP_SIZE, 32)
        .Grid(ctx.SMCount)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, ySize, stride8, lenTiles, gatherTileScale, allPacked)
        .Write(&buf);
    ctx.Sync->Sync(c, deviceId, buf);
}


void TMultiDeviceReducer::ReduceXSplit(
    TPtrArg<TGraph> c, TPtrArg<TMultiDeviceFRed2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceCompressedRowsXSplitImpl(c, ctx, p, ySize, deviceId);
}

void TMultiDeviceReducer::ReduceXSplit(
    TPtrArg<TGraph> c, TPtrArg<TMultiDeviceFRed2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceCompressedRowsXSplitImpl(c, ctx, p, ySize, deviceId);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void AllGatherKernel(int myDgRank, int dgSize, int lenTiles, TMultiGPUptr<float> stateArr)
{
    for (int x = 0; x < lenTiles; ++x) {
        int tileId = myDgRank * lenTiles + x;
        float4 vec;
        vec = LoadWarpVec(stateArr.PtrArr[myDgRank] + tileId * WARP_VEC_DIM);
        for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
            if (dgRank != myDgRank) {
                StoreWarpVec(stateArr.PtrArr[dgRank] + tileId * WARP_VEC_DIM, vec);
            }
        }
    }
    __threadfence_system(); // ensure global visibility
}


void TMultiDeviceReducer::AllGather(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    if (!ctx.NeedSync()) {
        return;
    }

    TCudaVector<float> &buf = p->GetData(deviceId);
    TMultiGPUptr<float> stateArr = ctx.MakeMultiPtr(p);
    int lenTiles = buf.GetSize() / ctx.dgSize / WARP_VEC_DIM;
    Y_VERIFY(lenTiles * ctx.dgSize * WARP_VEC_DIM == buf.GetSize());

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, AllGatherKernel).DepWrite(buf).Read(ctx.dgRank, ctx.dgSize, lenTiles, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class THold>
__global__ void AllGatherRows2DKernel(int myDgRank, int dgSize, int ySize, int strideInBytes, int lenTiles, TMultiGPUptr<T> stateArr)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    for (int row = warpId + blockIdx.x * blockDim.y; row < ySize; row += blockDim.y * gridDim.x) {
        int rowOffset = row * strideInBytes;
        for (int x = 0; x < lenTiles; ++x) {
            int tileId = myDgRank * lenTiles + x;
            THold vec = ((THold *)(AdvancePtr(stateArr.PtrArr[myDgRank], rowOffset) + tileId * WARP_VEC_DIM))[h];
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                if (dgRank != myDgRank) {
                    ((THold *)(AdvancePtr(stateArr.PtrArr[dgRank], rowOffset) + tileId * WARP_VEC_DIM))[h] = vec;
                }
            }
        }
    }
    __threadfence_system(); // ensure global visibility
}


template <class T, class THold>
void AllGatherXSplitImpl(TPtrArg<TGraph> c, TMultiDeviceOpCtx &ctx, TPtrArg<TMultiDevice2DArray<T>> p, yint deviceId)
{
    if (!ctx.NeedSync()) {
        return;
    }

    TCuda2DArray<T> &buf = p->GetData(deviceId);
    TMultiGPUptr<T> stateArr = ctx.MakeMultiPtr(p);
    int lenTiles = buf.GetXSize() / ctx.dgSize / WARP_VEC_DIM;
    Y_ASSERT(lenTiles * ctx.dgSize * WARP_VEC_DIM == buf.GetXSize());
    int ySize = buf.GetYSize();
    int stride = buf.GetDeviceMem().Stride;

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, AllGatherRows2DKernel<T, THold>)
        .Block(WARP_SIZE, 32)
        .Grid(ctx.SMCount)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, ySize, stride, lenTiles, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


void TMultiDeviceReducer::AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherXSplitImpl<float, int4>(c, ctx, p, deviceId);
}

void TMultiDeviceReducer::AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherXSplitImpl<half, int2>(c, ctx, p, deviceId);
}

void TMultiDeviceReducer::AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<i8>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherXSplitImpl<i8, int>(c, ctx, p, deviceId);
}

void TMultiDeviceReducer::AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<e4m3>> p, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherXSplitImpl<e4m3, int>(c, ctx, p, deviceId);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void AllGatherMoeKernel(
    int myDgRank, int dgSize, int lenTiles, TCudaSpan expertSpan, TCuda1DPtr<int> tileExpert, TMultiGPUptr<float> stateArr)
{
    for (int tileId = 0; tileId < lenTiles; ++tileId) {
        int expertId = tileExpert[tileId];
        if (expertId < expertSpan.Beg || expertId >= expertSpan.Fin) {
            continue;
        }
        float4 vec;
        vec = LoadWarpVec(stateArr.PtrArr[myDgRank] + tileId * WARP_VEC_DIM);
        for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
            if (dgRank != myDgRank) {
                StoreWarpVec(stateArr.PtrArr[dgRank] + tileId * WARP_VEC_DIM, vec);
            }
        }
    }
    __threadfence_system(); // ensure global visibility
}


void TMultiDeviceReducer::AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, TKernelParameter<int> &len, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    if (!ctx.NeedSync()) {
        return;
    }

    TCudaVector<float> &buf = p->GetData(deviceId);
    TMultiGPUptr<float> stateArr = ctx.MakeMultiPtr(p);

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, AllGatherMoeKernel)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, DivCeil(len, WARP_VEC_DIM), expertSpan, tileExpert, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class THold>
__global__ void AllGather2DMoeKernel(int myDgRank, int dgSize, int ySize, int strideInBytes, int lenTiles, TCudaSpan expertSpan,
    TCuda1DPtr<int> tileExpert, TMultiGPUptr<T> stateArr)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    for (int row = warpId + blockIdx.x * blockDim.y; row < ySize; row += blockDim.y * gridDim.x) {
        int expertId = tileExpert[row / WARP_VEC_DIM];
        if (expertId < expertSpan.Beg || expertId >= expertSpan.Fin) {
            continue;
        }
        int rowOffset = row * strideInBytes;
        for (int tileId = 0; tileId < lenTiles; ++tileId) {
            THold vec = ((THold *)(AdvancePtr(stateArr.PtrArr[myDgRank], rowOffset) + tileId * WARP_VEC_DIM))[h];
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                if (dgRank != myDgRank) {
                    ((THold *)(AdvancePtr(stateArr.PtrArr[dgRank], rowOffset) + tileId * WARP_VEC_DIM))[h] = vec;
                }
            }
        }
    }
    __threadfence_system(); // ensure global visibility
}


template <class T, class THold, class TYSize>
void AllGatherMoeImpl(TPtrArg<TGraph> c, TMultiDeviceOpCtx &ctx, TPtrArg<TMultiDevice2DArray<T>> p, TYSize &&ySize, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    if (!ctx.NeedSync()) {
        return;
    }

    TCuda2DArray<T> &buf = p->GetData(deviceId);
    TMultiGPUptr<T> stateArr = ctx.MakeMultiPtr(p);
    int lenTiles = buf.GetXSize() / WARP_VEC_DIM;
    int stride = buf.GetDeviceMem().Stride;

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, AllGather2DMoeKernel<T, THold>)
        .Block(WARP_SIZE, 32)
        .Grid(ctx.SMCount)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, ySize, stride, lenTiles, expertSpan, tileExpert, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


void TMultiDeviceReducer::AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize,
    yint deviceId, const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherMoeImpl<float, int4>(c, ctx, p, ySize, deviceId, expertSpan, tileExpert);
}

void TMultiDeviceReducer::AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherMoeImpl<half, int2>(c, ctx, p, ySize, deviceId, expertSpan, tileExpert);
}

void TMultiDeviceReducer::AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<i8>> p, TKernelParameter<int> &ySize, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherMoeImpl<i8, int>(c, ctx, p, ySize, deviceId, expertSpan, tileExpert);
}

void TMultiDeviceReducer::AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<e4m3>> p, TKernelParameter<int> &ySize, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    AllGatherMoeImpl<e4m3, int>(c, ctx, p, ySize, deviceId, expertSpan, tileExpert);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void ReduceRowsMoeKernel(int myDgRank, int dgSize, int ySize, int strideInBytes, int lenTiles, TCudaSpan expertSpan,
    TCuda1DPtr<int> tileExpert, TMultiGPUptr<T> stateArr)
{
    int warpId = threadIdx.y;
    for (int row = warpId + blockIdx.x * blockDim.y; row < ySize; row += blockDim.y * gridDim.x) {
        int expertId = tileExpert[row / WARP_VEC_DIM];
        if (expertId < expertSpan.Beg || expertId >= expertSpan.Fin) {
            continue;
        }
        int rowOffset = row * strideInBytes;
        for (int tileId = 0; tileId < lenTiles; ++tileId) {
            float4 vec[MAX_NUM_DEVICES];
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                vec[dgRank] = LoadWarpVec(AdvancePtr(stateArr.PtrArr[dgRank], rowOffset) + tileId * WARP_VEC_DIM);
            }
            float4 sum = ZeroWarpVec();
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                sum = sum + vec[dgRank];
            }
            StoreWarpVec(AdvancePtr(stateArr.PtrArr[myDgRank], rowOffset) + tileId * WARP_VEC_DIM, sum);
        }
    }
}


template <class T, class TYSize>
void ReduceRowsMoeImpl(TPtrArg<TGraph> c, TMultiDeviceOpCtx &ctx, TPtrArg<TMultiDevice2DArray<T>> p, TYSize &&ySize, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    if (!ctx.NeedSync()) {
        return;
    }

    TCuda2DArray<T> &buf = p->GetData(deviceId);
    TMultiGPUptr<T> stateArr = ctx.MakeMultiPtr(p);
    int lenTiles = buf.GetXSize() / WARP_VEC_DIM;
    int stride = buf.GetDeviceMem().Stride;
    Y_VERIFY(lenTiles * WARP_VEC_DIM == buf.GetXSize());

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, ReduceRowsMoeKernel<T>)
        .Block(WARP_SIZE, 32)
        .Grid(ctx.SMCount)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, ySize, stride, lenTiles, expertSpan, tileExpert, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}


void TMultiDeviceReducer::ReduceMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceRowsMoeImpl(c, ctx, p, ySize, deviceId, expertSpan, tileExpert);
}

void TMultiDeviceReducer::ReduceMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId,
    const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    ReduceRowsMoeImpl(c, ctx, p, ySize, deviceId, expertSpan, tileExpert);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void AllGatherYSplitKernel(int myDgRank, int dgSize, int lenTiles, TCudaSpan ySpan, int stride, TMultiGPUptr<T> stateArr)
{
    for (int y = ySpan.Beg + blockIdx.x; y < ySpan.Fin; y += gridDim.x) {
        for (int tileId = threadIdx.y; tileId < lenTiles; tileId += blockDim.y) {
            int offset = y * stride + tileId * WARP_VEC_DIM * sizeof(T);
            float4 vec;
            vec = LoadWarpVec(AdvancePtr(stateArr.PtrArr[myDgRank], offset));
            for (int dgRank = 0; dgRank < dgSize; ++dgRank) {
                if (dgRank != myDgRank) {
                    StoreWarpVec(AdvancePtr(stateArr.PtrArr[dgRank], offset), vec);
                }
            }
        }
    }
    __threadfence_system(); // ensure global visibility
}


void TMultiDeviceReducer::AllGatherYSplit(
    TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, const TCudaSpan &ySpan, yint deviceId)
{
    TMultiDeviceOpCtx ctx(c, Priority, Sync, deviceId);
    if (!ctx.NeedSync()) {
        return;
    }

    TCuda2DArray<float> &buf = p->GetData(deviceId);
    TMultiGPUptr<float> stateArr = ctx.MakeMultiPtr(p);
    int lenTiles = buf.GetXSize() / WARP_VEC_DIM;
    int stride = buf.GetDeviceMem().Stride;
    Y_VERIFY(lenTiles * WARP_VEC_DIM == buf.GetXSize());

    ctx.Sync->Sync(c, deviceId, buf);
    CudaCall(c, AllGatherYSplitKernel<float>)
        .Block(WARP_SIZE, 32)
        .Grid(ctx.SMCount)
        .DepWrite(buf)
        .Read(ctx.dgRank, ctx.dgSize, lenTiles, ySpan, stride, stateArr);
    ctx.Sync->Sync(c, deviceId, buf);
}
}
