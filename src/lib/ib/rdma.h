#pragma once

namespace NCuda
{
class TStream;
struct TMemoryBlob;
class TCudaMemoryPool;
}

namespace NNet
{
//////////////////////////////////////////////////////////////////////////
struct TRdmaCompletion
{
    struct TDeviceCtx
    {
        TVector<ui64> RecvId;
        ui64 MinRecvId = 0;
    };
    TVector<TDeviceCtx> DeviceArr;

public:
    void Init(yint deviceCount, yint myRank, yint rankCount)
    {
        DeviceArr.resize(deviceCount);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TDeviceCtx &ctx = DeviceArr[deviceId];
            ctx.RecvId.resize(rankCount, 0);
            ctx.RecvId[myRank] = MAX_UI64;
        }
    }
    bool OnComplete(yint deviceId, yint rank, ui64 cmdId)
    {
        TDeviceCtx &ctx = DeviceArr[deviceId];
        ctx.RecvId[rank] = cmdId;
        ui64 minId = MAX_UI64;
        for (ui64 id : ctx.RecvId) {
            minId = min(id, minId);
        }
        if (ctx.MinRecvId != minId) {
            ctx.MinRecvId = minId;
            return true;
        }
        return false;
    }
};


//////////////////////////////////////////////////////////////////////////
struct TRdmaCompletionCounters
{
    ui64 SendCompleteId;
    ui64 RecvCompleteId;
};


//////////////////////////////////////////////////////////////////////////
struct IRdmaTransport : public TThrRefBase
{
    struct TChain
    {
        TRdmaCompletion SendCompletion;
        TRdmaCompletion RecvCompletion;
        TVector<TRdmaCompletionCounters *> CompleteIdArr; // device mem
    };
    TVector<TChain> ChainArr;

    IRdmaTransport(yint chainCount) { ChainArr.resize(chainCount); }
    virtual void PrepareConnections(TVector<ui8> *p) = 0;
    virtual void EstablishConnections(yint myRank, const TVector<TString> &peerList, TVector<TVector<ui8>> &handshakeArr) = 0;
    virtual void InitDevice(yint deviceId) = 0;
    virtual void OnInitComplete(
        const TVector<TVector<NCuda::TMemoryBlob>> &allBufArr, TVector<TIntrusivePtr<NCuda::TCudaMemoryPool>> &poolArr) = 0;
    virtual void RdmaWrite(yint chainId, yint dstRank, int deviceId, ui64 cmdId, int localBuf, ui64 localOffset, int remoteBuf,
        ui64 remoteOffset, yint sz) = 0;
    virtual void ProcessIncoming() = 0;
};
}
