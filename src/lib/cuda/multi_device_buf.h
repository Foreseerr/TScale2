#pragma once
#include "cuda_arrays.h"
#include "cuda_init.h"


namespace NCuda
{
template <class T>
struct TKernelParameter;

class TGraph;
class TMultiGPUSync;

///////////////////////////////////////////////////////////////////////////////////////////////////
class TDeviceGroup
{
    TVector<int> RankArr;
    TVector<int> DevArr;
    int Size = 0;

public:
    TDeviceGroup() {}
    TDeviceGroup(int start, int fin)
    {
        Size = fin - start;
        RankArr.resize(MAX_NUM_DEVICES, -1);
        DevArr.resize(Size);
        for (yint k = start; k < fin; ++k) {
            RankArr[k] = k - start;
            DevArr[k - start] = k;
        }
    }
    TDeviceGroup(const TVector<int> &deviceArr)
    {
        Size = YSize(deviceArr);
        DevArr = deviceArr;
        RankArr.resize(MAX_NUM_DEVICES, -1);
        for (yint k = 0; k < Size; ++k) {
            yint deviceId = DevArr[k];
            Y_ASSERT(RankArr[deviceId] == -1);
            RankArr[deviceId] = k;
        }
    }
    int GetSize() const { return Size; }
    int Rank(yint deviceId) const { return RankArr[deviceId]; }
    int DeviceId(yint rank) const { return DevArr[rank]; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class TMultiDeviceVector : public TThrRefBase
{
    struct TBuffer : public TThrRefBase
    {
        TCudaVector<T> Data;
    };
    TVector<TIntrusivePtr<TBuffer>> DeviceArr;

public:
    typedef T TElem;

    TMultiDeviceVector() { DeviceArr.resize(MAX_NUM_DEVICES); }
    void AllocateCuda(yint deviceId, yint sz, TPtrArg<TCudaMemoryPool> pool)
    {
        DeviceArr[deviceId] = new TBuffer;
        DeviceArr[deviceId]->Data.AllocateCuda(sz, pool);
    }
    TCudaVector<T> &GetData(yint deviceId) { return DeviceArr[deviceId]->Data; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class TMultiDevice2DArray : public TThrRefBase
{
    struct TBuffer : public TThrRefBase
    {
        TCuda2DArray<T> Data;
    };
    TVector<TIntrusivePtr<TBuffer>> DeviceArr;

public:
    typedef T TElem;

    TMultiDevice2DArray() { DeviceArr.resize(MAX_NUM_DEVICES); }
    void AllocateCuda(yint deviceId, yint xSize, yint ySize, TPtrArg<TCudaMemoryPool> pool)
    {
        DeviceArr[deviceId] = new TBuffer;
        DeviceArr[deviceId]->Data.AllocateCuda(xSize, ySize, pool);
    }
    bool IsAllocated(yint deviceId) { return DeviceArr[deviceId].Get() != nullptr; }
    TCuda2DArray<T> &GetData(yint deviceId) { return DeviceArr[deviceId]->Data; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class TMultiDeviceFRed2DArray : public TThrRefBase
{
    struct TBuffer : public TThrRefBase
    {
        TCuda2DArray<T> Data;
        TCuda2DArray<i8> Data8; // compressed data for faster reduce
        TCuda2DArray<float> TileScale; // [tile][y]
        TCuda2DArray<float> GatherTileScale; // [device, tile][y]
    };
    TVector<TIntrusivePtr<TBuffer>> DeviceArr;

public:
    enum {
        TILE_SIZE = 128
    };

public:
    typedef T TElem;

    TMultiDeviceFRed2DArray() { DeviceArr.resize(MAX_NUM_DEVICES); }
    void AllocateCuda(yint deviceId, yint xSize, yint ySize, TPtrArg<TCudaMemoryPool> pool)
    {
        DeviceArr[deviceId] = new TBuffer;
        DeviceArr[deviceId]->Data.AllocateCuda(xSize, ySize, pool);
        DeviceArr[deviceId]->Data8.AllocateCuda(xSize, ySize, pool);
        DeviceArr[deviceId]->TileScale.AllocateCuda(RoundUp(ySize, TILE_SIZE), (xSize / TILE_SIZE), pool);
        DeviceArr[deviceId]->GatherTileScale.AllocateCuda(RoundUp(ySize, TILE_SIZE), (xSize / TILE_SIZE), pool);
    }
    TCuda2DArray<T> &GetData(yint deviceId) { return DeviceArr[deviceId]->Data; }
    TCuda2DArray<i8> &GetPackedData(yint deviceId) { return DeviceArr[deviceId]->Data8; }
    TCuda2DArray<float> &GetTileScale(yint deviceId) { return DeviceArr[deviceId]->TileScale; }
    TCuda2DArray<float> &GetGatherTileScale(yint deviceId) { return DeviceArr[deviceId]->GatherTileScale; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline TCuda2DArrayFragment<T> CalcXSplitWindow(TCuda2DArray<T> &data, yint xSize, yint deviceId, const TDeviceGroup &deviceGroup)
{
    Y_VERIFY(xSize <= data.GetXSize());
    yint dgSize = deviceGroup.GetSize();
    if (dgSize > 1) {
        Y_VERIFY(xSize % dgSize == 0);
        int winXSize = xSize / dgSize;
        return data.MakeFragment(winXSize * deviceGroup.Rank(deviceId), winXSize, 0, data.GetYSize());
    } else {
        return data.MakeFragment(0, 0);
    }
}

template <class T>
inline TCuda2DArrayFragment<T> CalcYSplitWindow(TCuda2DArray<T> &data, yint ySize, yint deviceId, const TDeviceGroup &deviceGroup)
{
    Y_VERIFY(ySize <= data.GetYSize());
    yint dgSize = deviceGroup.GetSize();
    if (dgSize > 1) {
        Y_VERIFY(ySize % dgSize == 0);
        int winYSize = ySize / dgSize;
        return data.MakeFragment(0, data.GetXSize(), winYSize * deviceGroup.Rank(deviceId), winYSize);
    } else {
        return data.MakeFragment(0, 0);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EReducerPriority {
    RP_FULL_GRID,
    RP_LOW_PRIORITY,
};

class TMultiDeviceReducer : public TThrRefBase
{
    TIntrusivePtr<TMultiGPUSync> Sync;
    TDeviceGroup DeviceGroup;
    EReducerPriority Priority = RP_FULL_GRID;

private:
    ~TMultiDeviceReducer();

public:
    TMultiDeviceReducer(const TDeviceGroup &deviceGroup, EReducerPriority pr);
    const TDeviceGroup &GetDeviceGroup() const { return DeviceGroup; }
    void InitSync(yint deviceId);

    // all reduce
    void AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, yint deviceId);
    void AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, yint deviceId);
    void AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId);
    void AllReduce(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId);
    // all max
    void AllMax(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, yint deviceId);
    
    // reduce
    void ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, yint deviceId);
    void ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, yint deviceId);
    void ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId);
    void ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId);

    // reduce with per tile compression
    void ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceFRed2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId);
    void ReduceXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceFRed2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId);

    // gather
    void AllGather(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, yint deviceId);
    void AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, yint deviceId);
    void AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, yint deviceId);
    void AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<i8>> p, yint deviceId);
    void AllGatherXSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<e4m3>> p, yint deviceId);

    // moe gather (select tiles)
    void AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceVector<float>> p, TKernelParameter<int> &len, yint deviceId,
        const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert);

    // moe gather (select rows)
    void AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId,
        const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert);
    void AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId,
        const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert);
    void AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<i8>> p, TKernelParameter<int> &ySize, yint deviceId,
        const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert);
    void AllGatherMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<e4m3>> p, TKernelParameter<int> &ySize, yint deviceId,
        const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert);

    // more reduce (select rows)
    void ReduceMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, TKernelParameter<int> &ySize, yint deviceId,
        const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert);
    void ReduceMoe(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<half>> p, TKernelParameter<int> &ySize, yint deviceId,
        const TCudaSpan &expertSpan, TCudaVector<int> &tileExpert);

    // horizontal stripes gather
    void AllGatherYSplit(TPtrArg<TGraph> c, TPtrArg<TMultiDevice2DArray<float>> p, const TCudaSpan &ySpan, yint deviceId);
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMultiDeviceBufferFabric : public TThrRefBase
{
    THashMap<TString, TIntrusivePtr<TThrRefBase>> BufSet;

    template <class T>
    TIntrusivePtr<T> CreateImpl(const TString &id)
    {
        TIntrusivePtr<T> res;
        auto it = BufSet.find(id);
        if (it != BufSet.end()) {
            return (T *)it->second.Get();
        } else {
            res = new T();
            BufSet[id] = res;
            return res;
        }
    }

public:
    template <class T>
    TIntrusivePtr<TMultiDeviceVector<T>> CreateVector(const TString &id)
    {
        return CreateImpl<TMultiDeviceVector<T>>(id);
    }
    template <class T>
    TIntrusivePtr<TMultiDevice2DArray<T>> Create2DArray(const TString &id)
    {
        return CreateImpl<TMultiDevice2DArray<T>>(id);
    }
    template <class T>
    TIntrusivePtr<TMultiDeviceFRed2DArray<T>> CreateFRed2DArray(const TString &id)
    {
        return CreateImpl<TMultiDeviceFRed2DArray<T>>(id);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMultiDeviceBuffers : public TThrRefBase
{
    TIntrusivePtr<TMultiDeviceReducer> Reducer;
    TIntrusivePtr<TMultiDeviceBufferFabric> FabricPtr;

public:
    TMultiDeviceBuffers(TPtrArg<TMultiDeviceBufferFabric> fabric, const TDeviceGroup &deviceGroup) : FabricPtr(fabric.Get())
    {
        Reducer = new TMultiDeviceReducer(deviceGroup, RP_FULL_GRID);
    }
    void InitSync(yint deviceId) { Reducer->InitSync(deviceId); }
    TMultiDeviceReducer &Op() { return *Reducer.Get(); }
    TMultiDeviceBufferFabric &Fab() { return *FabricPtr.Get(); }
    yint GetDgSize() const { return Reducer->GetDeviceGroup().GetSize(); }
    const TDeviceGroup &GetDeviceGroup() const { return Reducer->GetDeviceGroup(); }
};
}
