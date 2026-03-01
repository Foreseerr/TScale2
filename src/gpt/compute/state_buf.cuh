#pragma once
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/multi_device_buf.h>
#include "cfg_precision.h"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
class TBufferTransfer;
class TBufferTransferSync;

class TStateBuffer : public TThrRefBase
{
    TIntrusivePtr<TMultiDevice2DArray<TStateFloat>> State;
    THashMap<TString, TIntrusivePtr<TBufferTransfer>> AllTransfers;
    TIntrusivePtr<TBufferTransferSync> Sync;
    yint SyncIdCount = 0;
    bool IsPipelineParallel = false;

    ~TStateBuffer();
    TIntrusivePtr<TBufferTransfer> GetTransfer(TPtrArg<TGraph> c, const TString &name);
    yint AllocSyncId();

public:
    TStateBuffer(TPtrArg<TMultiDeviceBuffers> multiBuffers, yint stateBufId, bool isPipelineParallel);
    void AllocateCuda(yint deviceId, yint xSize, yint ySize);
    void RecvData(TPtrArg<TGraph> c, const TString &name, yint deviceId, yint prevDeviceId);
    void CopyData(TPtrArg<TGraph> c, const TString &name, TKernelParameter<int> &len, yint deviceId, yint nextDeviceId);
    TCuda2DArray<TStateFloat> &Get(yint deviceId)
    {
        return State->GetData(deviceId);
    }
};
}
