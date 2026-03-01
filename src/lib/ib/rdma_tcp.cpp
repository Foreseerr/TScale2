#include "rdma.h"
#include <cuda.h>
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_init.h>
#include <lib/guid/guid.h>
#include <lib/ib/ib_buffers.h>
#include <lib/ib/ib_low.h>
#include <lib/net/ip_address.h>
#include <lib/net/p2p.h>


namespace NNet
{
using namespace NCuda;
static TGuid TcpRdmaToken(0xb9f2a9fe, 0x86a822f3, 0x448dc078, 0x32bf1a94);

//////////////////////////////////////////////////////////////////////////
class TTcpRdmaTransport : public IRdmaTransport
{
    struct TDeviceCtx : public TThrRefBase
    {
        TVector<NCuda::TMemoryBlob> DeviceBufArr;
        NCuda::TStream Stream;
    };

    struct TRdmaHeader
    {
        int ChainId;
        int DeviceId;
        int RemoteBufId;
        ui64 RemoteOffset;
        ui64 CmdId;
    };

    struct THandshake
    {
        int Port;
    };

private:
    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<TP2PNetwork> P2P;
    TVector<TIntrusivePtr<TDeviceCtx>> DeviceArr;
    TCudaVector<char> PageLockedBuf;


public:
    TTcpRdmaTransport(TPtrArg<ITcpSendRecv> net, yint maxTransferSize, yint chainCount) : IRdmaTransport(chainCount), Net(net.Get())
    {
        P2P = new TP2PNetwork(Net, TcpRdmaToken);
        PageLockedBuf.AllocateHost(maxTransferSize);
    }

    void PrepareConnections(TVector<ui8> *p) override
    {
        THandshake hs;
        hs.Port = P2P->GetPort();
        SerializeMem(IO_WRITE, p, hs);
    }

    void EstablishConnections(yint myRank, const TVector<TString> &peerList, TVector<TVector<ui8>> &handshakeArr) override
    {
        TVector<TString> peerAddr = peerList;
        for (yint k = 0; k < YSize(peerList); ++k) {
            THandshake hs;
            SerializeMem(IO_READ, &handshakeArr[k], hs);
            NNet::ReplacePort(&peerAddr[k], hs.Port);
        }
        P2P->ConnectP2P(myRank, peerAddr, TcpRdmaToken);
        // DebugPrintf("rdma net ok\n");
    }

    void InitDevice(yint deviceId) override
    {
        if (deviceId >= YSize(DeviceArr)) {
            DeviceArr.resize(deviceId + 1);
        }
        DeviceArr[deviceId] = new TDeviceCtx();
    }

    void OnInitComplete(const TVector<TVector<NCuda::TMemoryBlob>> &allBufArr, TVector<TIntrusivePtr<NCuda::TCudaMemoryPool>> &poolArr) override
    {
        for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
            TDeviceCtx &ctx = *DeviceArr[deviceId];
            ctx.DeviceBufArr = allBufArr[deviceId];
        }
    }

    void UpdateCompletionCounters(yint deviceId, yint chainId)
    {
        TDeviceCtx &ctx = *DeviceArr[deviceId];
        TChain &chain = ChainArr[chainId];
        TRdmaCompletionCounters rcc;
        rcc.RecvCompleteId = chain.RecvCompletion.DeviceArr[deviceId].MinRecvId;
        rcc.SendCompleteId = chain.SendCompletion.DeviceArr[deviceId].MinRecvId;
        memcpy(PageLockedBuf.GetHostPtr(), &rcc, sizeof(rcc));
        cudaError_t rv = cudaMemcpyAsync(
            chain.CompleteIdArr[deviceId], PageLockedBuf.GetHostPtr(), Max<yint>(128, sizeof(rcc)), cudaMemcpyHostToDevice, ctx.Stream);
        Y_VERIFY(rv == cudaSuccess);
        ctx.Stream.Sync();
    }

    void RdmaWrite(yint chainId, yint dstRank, int deviceId, ui64 cmdId, int localBuf, ui64 localOffset, int remoteBuf, ui64 remoteOffset,
        yint sz) override
    {
        // DebugPrintf("sending rdma to %g, device %g, buf %g, offset %lld, size %g, cmd %g\n", dstRank * 1., deviceId * 1., bufId * 1.,
        // remoteOffset, sz * 1., cmdId * 1.);
        Y_VERIFY(sz <= PageLockedBuf.GetSize());
        TDeviceCtx &ctx = *DeviceArr[deviceId];
        TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
        pkt->Data.resize(sizeof(TRdmaHeader) + sz);
        TRdmaHeader &hdr = *(TRdmaHeader *)pkt->Data.data();
        hdr.ChainId = chainId;
        hdr.DeviceId = deviceId;
        hdr.RemoteBufId = remoteBuf;
        hdr.RemoteOffset = remoteOffset;
        hdr.CmdId = cmdId;
        const char *localBufStart = (char *)ctx.DeviceBufArr[localBuf].Ptr;
        cudaError_t rv = cudaMemcpyAsync(PageLockedBuf.GetHostPtr(), localBufStart + localOffset, sz, cudaMemcpyDeviceToHost, ctx.Stream);
        Y_VERIFY(rv == cudaSuccess);
        ctx.Stream.Sync();
        memcpy(pkt->Data.data() + sizeof(TRdmaHeader), PageLockedBuf.GetHostPtr(), sz);
        P2P->Send(dstRank, pkt);
        if (ChainArr[chainId].SendCompletion.OnComplete(deviceId, dstRank, cmdId)) {
            UpdateCompletionCounters(deviceId, chainId);
        }
    }

    void ProcessIncoming() override
    {
        TIntrusivePtr<TTcpPacketReceived> pkt;
        if (P2P->GetQueue()->Dequeue(&pkt)) {
            yint fromRank = P2P->GetRank(pkt->Conn);
            TRdmaHeader &hdr = *(TRdmaHeader *)pkt->Data.data();
            TDeviceCtx &ctx = *DeviceArr[hdr.DeviceId];
            NCuda::TMemoryBlob &dstBuf = ctx.DeviceBufArr[hdr.RemoteBufId];
            char *deviceBufStart = (char *)dstBuf.Ptr;
            void *dst = deviceBufStart + hdr.RemoteOffset;
            const void *src = pkt->Data.data() + sizeof(TRdmaHeader);
            yint sz = YSize(pkt->Data) - sizeof(TRdmaHeader);
            Y_VERIFY(sz <= PageLockedBuf.GetSize());
            Y_VERIFY(hdr.RemoteOffset + sz <= dstBuf.GetSize());
            memcpy(PageLockedBuf.GetHostPtr(), src, sz);
            cudaError_t rv = cudaMemcpyAsync(dst, PageLockedBuf.GetHostPtr(), sz, cudaMemcpyHostToDevice, ctx.Stream);
            Y_VERIFY(rv == cudaSuccess);
            ctx.Stream.Sync();
            if (ChainArr[hdr.ChainId].RecvCompletion.OnComplete(hdr.DeviceId, fromRank, hdr.CmdId)) {
                UpdateCompletionCounters(hdr.DeviceId, hdr.ChainId);
            }
            // DebugPrintf("recv rdma device %g, from %g, buf %g, size %g, cmd %g\n", hdr.DeviceId * 1., fromRank * 1., hdr.BufId * 1., sz
            // * 1., hdr.CmdId * 1.);
        }
    }
};


TIntrusivePtr<IRdmaTransport> CreateTcpRdmaTransfer(TPtrArg<ITcpSendRecv> net, yint maxTransferSize, yint chainCount)
{
    return new TTcpRdmaTransport(net, maxTransferSize, chainCount);
}
}
