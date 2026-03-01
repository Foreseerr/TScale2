#include <util/pch.h>
#define KERNEL_UNIT "state_buf/"
#include "state_buf.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint MAX_SYNC_COUNT = 16384;

struct TBTSyncCounter
{
    ui64 Val[16];

    void Set(ui64 x)
    {
        for (yint k = 0; k < ARRAY_SIZE(Val); ++k) {
            Val[k] = x;
        }
    }
};

__global__ void BufferTransferArriveKernel(int syncId, TCuda1DPtr<TBTSyncCounter> myCounter, TCuda1DPtr<TBTSyncCounter> remoteBuf)
{
    int h = threadIdx.x;
    if (h < 16) {
        ui64 id = myCounter[syncId].Val[h] + 1;
        myCounter[syncId].Val[h] = id;
        remoteBuf[syncId].Val[h] = id;
    }
    __threadfence_system();
}

__global__ void BufferTransferSyncKernel(int syncId, TCuda1DPtr<TBTSyncCounter> prevCounter, TCuda1DPtr<TBTSyncCounter> buf)
{
    int h = threadIdx.x;
    if (h < 16) {
        volatile ui64 *bufPtr = &buf[syncId].Val[h];
        for (ui64 id = prevCounter[syncId].Val[h];;) {
            ui64 val = *bufPtr;
            if (val != id) {
                prevCounter[syncId].Val[h] = val;
                break;
            }
        }
    }
}

__global__ void BufferTransferNopKernel() {}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TBufferTransferSync : public TThrRefBase
{
    struct TBuf
    {
        TCudaVector<TBTSyncCounter> Buf;
        TCudaVector<TBTSyncCounter> SyncId;
        TCudaVector<TBTSyncCounter> ArriveId;
    };
    TBuf BufArr[MAX_NUM_DEVICES];

public:
    void AllocateCuda(yint deviceId)
    {
        TBuf &buf = BufArr[deviceId];
        buf.Buf.AllocateCuda(MAX_SYNC_COUNT);
        buf.Buf.ClearDeviceMem();
        buf.SyncId.AllocateCuda(MAX_SYNC_COUNT);
        buf.SyncId.ClearDeviceMem();
        TVector<TBTSyncCounter> ac;
        TBTSyncCounter init;
        init.Set((deviceId + 1ll) << 56);
        ac.resize(MAX_SYNC_COUNT, init);
        buf.ArriveId.Init(ac);
    }

    TKernelOp &Sync(TPtrArg<TGraph> c, yint id, yint deviceId)
    {
        TBuf &sb = BufArr[deviceId];
        return CudaCall(c, BufferTransferSyncKernel).Read(id).Write(&sb.SyncId, &sb.Buf);
    }

    TKernelOp &Arrive(TPtrArg<TGraph> c, yint id, yint deviceId, yint targetDeviceId)
    {
        TBuf &sb = BufArr[deviceId];
        TBuf &sbTarget = BufArr[targetDeviceId];
        return CudaCall(c, BufferTransferArriveKernel).Read(id).Write(&sb.ArriveId, &sbTarget.Buf);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TBufferTransfer : public TThrRefBase
{
    struct TSendOps
    {
        TIntrusivePtr<TKernelOp> SendOp;
        void *DstBuf = 0;
        TIntrusivePtr<TGraph> C;
    };
    struct TRecvOps
    {
        TIntrusivePtr<TKernelOp> RecvPre;
        TIntrusivePtr<TKernelOp> RecvPost;
        void *DstBuf = 0;
        TIntrusivePtr<TGraph> C;
    };

    TIntrusivePtr<TBufferTransferSync> Sync;
    TVector<TSendOps> SendOps;
    TVector<TRecvOps> RecvOps;
    yint SyncIdSendComplete = 0;
    yint SyncIdCanSend = 0;

private:
    void Link()
    {
        while (!SendOps.empty() && !RecvOps.empty()) {
            TSendOps s = SendOps[0];
            TRecvOps r = RecvOps[0];
            Y_ASSERT(s.DstBuf == r.DstBuf);
            Y_ASSERT(s.C == r.C);
            s.SendOp->Dep(*r.RecvPre);
            r.RecvPost->Dep(*s.SendOp);
            SendOps.erase(SendOps.begin());
            RecvOps.erase(RecvOps.begin());
        }
    }

public:
    TBufferTransfer(TPtrArg<TBufferTransferSync> sync, yint sendComplete, yint canSend)
        : Sync(sync), SyncIdSendComplete(sendComplete), SyncIdCanSend(canSend)
    {
    }

    template <class TDst>
    void Receive(TPtrArg<TGraph> c, TDst *dst, yint deviceId, yint prevDeviceId)
    {
        if (SIMULATE_MULTI_GPU) {
            TRecvOps &r = *RecvOps.insert(RecvOps.end());
            r.RecvPre = &CudaCall(c, BufferTransferNopKernel).DepWrite(*dst);
            r.RecvPost = &CudaCall(c, BufferTransferNopKernel).DepWrite(*dst);
            r.DstBuf = dst;
            r.C = c.Get();
            Link();
        } else {
            Sync->Arrive(c, SyncIdCanSend, deviceId, prevDeviceId).DepWrite(*dst);
            Sync->Sync(c, SyncIdSendComplete, deviceId).DepWrite(*dst);
        }
    }

    template <class TSrc, class TDst>
    void Send(TPtrArg<TGraph> c, TDst *dst, TSrc &src, TKernelParameter<int> &len, yint deviceId, yint targetDeviceId)
    {
        if (SIMULATE_MULTI_GPU) {
            TSendOps &s = *SendOps.insert(SendOps.end());
            s.SendOp = &c->KernelCopy(dst, src, len).FullGrid(c);
            s.DstBuf = dst;
            s.C = c.Get();
            Link();
        } else {
            TKernelOp &canSend = Sync->Sync(c, SyncIdCanSend, deviceId);
            TKernelOp &op = c->KernelCopy(dst, src, len).Grid(BG_SM_COUNT).Dep(canSend);
            Sync->Arrive(c, SyncIdSendComplete, deviceId, targetDeviceId).Dep(op);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
TStateBuffer::~TStateBuffer() {}


TIntrusivePtr<TBufferTransfer> TStateBuffer::GetTransfer(TPtrArg<TGraph> c, const TString &name)
{
    auto it = AllTransfers.find(name);
    if (it == AllTransfers.end()) {
        TIntrusivePtr<TBufferTransfer> &res = AllTransfers[name];
        res = new TBufferTransfer(Sync, AllocSyncId(), AllocSyncId());
        return res;
    } else {
        return it->second;
    }
}


yint TStateBuffer::AllocSyncId()
{
    Y_VERIFY(SyncIdCount < MAX_SYNC_COUNT);
    return SyncIdCount++;
}


TStateBuffer::TStateBuffer(TPtrArg<TMultiDeviceBuffers> multiBuffers, yint stateBufId, bool isPipelineParallel)
    : IsPipelineParallel(isPipelineParallel)
{
    State = multiBuffers->Fab().Create2DArray<TStateFloat>(Sprintf("fp32-state-%g", stateBufId * 1.));
    Sync = new TBufferTransferSync();
}


void TStateBuffer::AllocateCuda(yint deviceId, yint xSize, yint ySize)
{
    State->AllocateCuda(deviceId, xSize, ySize, null_ptr_arg);
    if (IsPipelineParallel) {
        Sync->AllocateCuda(deviceId);
    }
}


void TStateBuffer::RecvData(TPtrArg<TGraph> c, const TString &name, yint deviceId, yint prevDeviceId)
{
    auto &buf = State->GetData(deviceId);
    GetTransfer(c, name)->Receive(c, &buf, deviceId, prevDeviceId);
}


void TStateBuffer::CopyData(TPtrArg<TGraph> c, const TString &name, TKernelParameter<int> &len, yint deviceId, yint nextDeviceId)
{
    auto &srcBuf = State->GetData(deviceId);
    auto &dstBuf = State->GetData(nextDeviceId);
    GetTransfer(c, name)->Send(c, &dstBuf, srcBuf, len, deviceId, nextDeviceId);
}
}
