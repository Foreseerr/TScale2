#include "rdma.h"
#include <cuda.h>
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_init.h>
#include <lib/guid/guid.h>
#include <lib/ib/ib_buffers.h>
#include <lib/ib/ib_low.h>
#include <lib/net/ip_address.h>
#include <lib/net/p2p.h>


constexpr yint HOST_MEM_REGION_SIZE = 4096;
constexpr int MAX_SRQ_WORK_REQUESTS = 1000;
constexpr int MAX_CQ_EVENTS = 1000;
constexpr int QP_SEND_QUEUE_SIZE = 128; // unclear what is max


namespace NNet
{
using namespace NCuda;

static TGuid InfinibandRdmaToken(0x2db72df8, 0x744b0b82, 0x82e84d34, 0xe8a0694d);

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline ui64 GetAddr64(T *p)
{
    return (char *)p - (char *)nullptr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static int GetPciDistance(int bus1, int bus2)
{
    // correct way is complicated, we assume that busId 0x91 is closer to 0x9f then to 0x8f
    // busId look like group:id 4 bit each
    if (bus1 == bus2) {
        return 0;
    }
    if ((bus1 & 15) == (bus2 & 15)) {
        return abs(bus1 - bus2);
    }
    return abs((bus1 >> 4) - (bus2 >> 4)) * 100;
}

static void GetClosestPorts(TVector<TIntrusivePtr<TIBPort>> *p, yint cudaDeviceCount)
{
    cuInit(0);

    TVector<TIntrusivePtr<TIBPort>> ibDevices = GetIBDevices();
    Y_VERIFY(!ibDevices.empty() && "no ib devices found");

    // duplicate NICs if there are more GPUs then NICs
    for (yint ibDeviceCount = YSize(ibDevices); YSize(ibDevices) < cudaDeviceCount;) {
        for (yint k = 0; k < ibDeviceCount; ++k) {
            ibDevices.push_back(ibDevices[k]);
        }
    }

    // match
    p->resize(cudaDeviceCount);
    for (yint devId = 0; devId < cudaDeviceCount; ++devId) {
        CUdevice cuDev;
        int pciBufId = 0;
        Y_VERIFY(cuDeviceGet(&cuDev, devId) == CUDA_SUCCESS);
        Y_VERIFY(cuDeviceGetAttribute(&pciBufId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cuDev) == CUDA_SUCCESS);
        // rv = cuDeviceGetAttribute(&pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cuDev); // pci slot

        yint bestId = 0;
        yint bestDiff = 1000000;
        for (yint k = 0; k < YSize(ibDevices); ++k) {
            yint diff = GetPciDistance(ibDevices[k]->GetBusId(), pciBufId);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestId = k;
            }
        }
        (*p)[devId] = ibDevices[bestId];
        ibDevices.erase(ibDevices.begin() + bestId);
    }
}


//////////////////////////////////////////////////////////////////////////
static TIntrusivePtr<TMemoryRegion> MakeCudaMemRegion(TPtrArg<TIBContext> ibCtx, TPtrArg<TCudaMemoryPool> pool)
{
    void *ptr = pool->GetDevicePtr(0);
    yint len = pool->GetMemSize();
    bool useDmaBuf = false;
    if (useDmaBuf) {
        // int dmaBufSupport = 0; // if dma buf is supported peermem is not needed, can use ibv_reg_dmabuf_mr()
        // Y_VERIFY(cuDeviceGetAttribute(&dmaBufSupport, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cuDev) == CUDA_SUCCESS);
        // Y_VERIFY(dmaBufSupport);
        // note, for kernel version < 6.2 and open drivers cuda device will report dmabuf support, but ibv_reg_dmabuf_mr fails with ENOTSUP
        // int dmaBuf = 0;
        // CUresult rv = CUDA_SUCCESS;
        // CUdeviceptr buf = (CUdeviceptr)((char *)mem.Ptr - (char *)0);
        // rv = cuMemGetHandleForAddressRange(&dmaBuf, buf, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        // Y_VERIFY(rv == CUDA_SUCCESS);
        // return new TMemoryRegion(ibCtx, buf, dmaBuf, len);
    }
    // expect running peermem
    return new TMemoryRegion(ibCtx, ptr, len);
}


//////////////////////////////////////////////////////////////////////////
class TInfinibandRdmaTransport : public IRdmaTransport
{
    struct TAsyncPoolRKey
    {
        ui64 Addr = 0;
        ui32 Key = 0;

        TAsyncPoolRKey() {}
        TAsyncPoolRKey(TPtrArg<TMemoryRegion> mr) : Addr((char *)mr->GetData() - (char *)0), Key(mr->GetRKey()) {}
        //char *GetData() const { return (char *)Addr; }
    };

    struct TMemBuffer
    {
        ui64 Addr = 0;

        TMemBuffer() {}
        TMemBuffer(void *p) : Addr((char *)p - (char *)0) {}
        char *GetData() const { return (char *)Addr; }
    };

    struct TLink
    {
        TIntrusivePtr<TRCQueuePair> QP;
        TAsyncPoolRKey PeerAsyncPool;
        TVector<TMemBuffer> PeerBufArr;
        TVector<ui64> CmdIdAdd;
        TVector<ui64> PrevCmdId;
    };

    struct TDeviceCtx : public TThrRefBase
    {
        TVector<NCuda::TMemoryBlob> DeviceBufArr;
        TIntrusivePtr<TIBPort> Port;
        TIntrusivePtr<TMemoryRegion> AsyncPoolMR;
        TVector<TMemBuffer> BufArr;
        TIntrusivePtr<TComplectionQueue> CQ;
        TVector<char> HostMemBuf; // used to pass RCC to device only
        TIntrusivePtr<TMemoryRegion> HostMemRegion;
        TIntrusivePtr<TSharedReceiveQueue> SRQ;
        TVector<TLink> Links;
        TIntrusivePtr<TRCQueuePair> QPself;

        void PostRecv()
        {
            // not used to receive data since we do rdma only
            SRQ->PostReceive(HostMemRegion, 0, HostMemRegion->GetData(), HOST_MEM_REGION_SIZE);
        }
        yint GetRankByQPN(int qpn) const
        {
            yint rankCount = YSize(Links);
            for (yint rank = 0; rank < rankCount; ++rank) {
                if (Links[rank].QP.Get() && qpn == Links[rank].QP->GetQPN()) {
                    return rank;
                }
            }
            Y_VERIFY(0);
            return 0;
        }
    };

    struct THandshake
    {
        int Port;
    };

    struct TIbDeviceHandshake
    {
        int LID = 0;
        int QPN = 0;
        int PSN = 0;
        TAsyncPoolRKey AsyncPool;
        TVector<TMemBuffer> BufArr;
        SAVELOAD(LID, QPN, PSN, AsyncPool, BufArr);
    };


private:
    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<TP2PNetwork> P2P;
    TVector<TIntrusivePtr<TDeviceCtx>> DeviceArr;

public:
    TInfinibandRdmaTransport(TPtrArg<ITcpSendRecv> net, yint chainCount) : IRdmaTransport(chainCount), Net(net.Get())
    {
        P2P = new TP2PNetwork(Net, InfinibandRdmaToken);
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
        P2P->ConnectP2P(myRank, peerAddr, InfinibandRdmaToken);
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
        yint chainCount = YSize(ChainArr);
        yint deviceCount = YSize(DeviceArr);
        yint peerCount = P2P->GetWorkerCount();

        // create IB ports
        TVector<TIntrusivePtr<TIBPort>> ibPorts;
        GetClosestPorts(&ibPorts, deviceCount);

        // register buffers
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TDeviceCtx &ctx = *DeviceArr[deviceId];
            ctx.DeviceBufArr = allBufArr[deviceId];
            ctx.Port = ibPorts[deviceId];
            TIntrusivePtr<TIBContext> ibCtx = ctx.Port->GetCtx().Get();
            ctx.AsyncPoolMR = MakeCudaMemRegion(ibCtx, poolArr[deviceId]);
            for (yint k = 0; k < YSize(ctx.DeviceBufArr); ++k) {
                ctx.BufArr.push_back(TMemBuffer(ctx.DeviceBufArr[k].Ptr));
            }
            ctx.CQ = new TComplectionQueue(ibCtx, MAX_CQ_EVENTS);
            ctx.HostMemBuf.resize(HOST_MEM_REGION_SIZE);
            ctx.HostMemRegion = new TMemoryRegion(ibCtx, ctx.HostMemBuf.data(), HOST_MEM_REGION_SIZE);
            ctx.SRQ = new TSharedReceiveQueue(ibCtx, MAX_SRQ_WORK_REQUESTS);
            ctx.Links.resize(peerCount);
            for (yint k = 0; k < MAX_SRQ_WORK_REQUESTS; ++k) {
                ctx.PostRecv();
            }
        }

        // create QPs
        for (yint peerId = 0; peerId < peerCount; ++peerId) {
            if (peerId != P2P->GetMyRank()) {
                TVector<TIbDeviceHandshake> hsArr;
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    TDeviceCtx &ctx = *DeviceArr[deviceId];
                    TIntrusivePtr<TIBContext> ibCtx = ctx.Port->GetCtx().Get();
                    TLink &link = ctx.Links[peerId];
                    link.QP = new TRCQueuePair(ibCtx, ctx.CQ, ctx.SRQ, QP_SEND_QUEUE_SIZE);

                    TIbDeviceHandshake hs;
                    hs.LID = ctx.Port->GetLID();
                    hs.QPN = link.QP->GetQPN();
                    hs.PSN = link.QP->GetPSN();
                    hs.AsyncPool = TAsyncPoolRKey(ctx.AsyncPoolMR);
                    hs.BufArr = ctx.BufArr;
                    hsArr.push_back(hs);
                }
                P2P->SendData(peerId, hsArr);
            }
        }

        // init connections
        yint recvCount = 0;
        while (recvCount < peerCount - 1) {
            TIntrusivePtr<TTcpPacketReceived> pkt;
            if (P2P->GetQueue()->Dequeue(&pkt)) {
                ++recvCount;
                yint peerId = P2P->GetRank(pkt->Conn);
                TVector<TIbDeviceHandshake> hsArr;
                SerializeMem(IO_READ, &pkt->Data, hsArr);
                Y_VERIFY(YSize(hsArr) == deviceCount);
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    TDeviceCtx &ctx = *DeviceArr[deviceId];
                    const TIbDeviceHandshake &hs = hsArr[deviceId];
                    TLink &link = ctx.Links[peerId];
                    ibv_ah_attr peerAddr;
                    MakeAH(&peerAddr, ctx.Port, hs.LID, 0);
                    link.QP->Init(peerAddr, hs.QPN, hs.PSN);
                    link.PeerAsyncPool = hs.AsyncPool;
                    link.PeerBufArr = hs.BufArr;
                    ClearPodArray(&link.CmdIdAdd, chainCount);
                    ClearPodArray(&link.PrevCmdId, chainCount);
                }
            }
        }

        // self connection
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TDeviceCtx &ctx = *DeviceArr[deviceId];
            TIntrusivePtr<TIBContext> ibCtx = ctx.Port->GetCtx().Get();
            ctx.QPself = new TRCQueuePair(ibCtx, ctx.CQ, ctx.SRQ, QP_SEND_QUEUE_SIZE);
            ibv_ah_attr selfAddr;
            MakeAH(&selfAddr, ctx.Port, ctx.Port->GetLID(), 0);
            ctx.QPself->Init(selfAddr, ctx.QPself->GetQPN(), ctx.QPself->GetPSN());
        }

        DebugPrintf("ib connections ok\n");
    }

    void RdmaWrite(yint chainId, yint dstRank, int deviceId, ui64 cmdId, int localBuf, ui64 localOffset, int remoteBuf, ui64 remoteOffset,
        yint sz) override
    {
        // DebugPrintf("sending rdma to %g, device %g, buf %g, offset %lld, size %g, cmd %g\n", dstRank * 1., deviceId * 1., bufId * 1.,
        TDeviceCtx &ctx = *DeviceArr[deviceId];
        TLink &link = ctx.Links[dstRank];
        const TMemBuffer &rBuf = link.PeerBufArr[remoteBuf];
        const TMemBuffer &lBuf = ctx.BufArr[localBuf];
        const TAsyncPoolRKey &remoteMR = link.PeerAsyncPool;
        TIntrusivePtr<TMemoryRegion> localMR = ctx.AsyncPoolMR;
        ui32 immData = (cmdId & 0xffff) + (chainId << 16);
        const char *localData = lBuf.GetData() + localOffset;
        ui64 wr_id = cmdId + (chainId << 48);
        link.QP->PostRDMAWriteImm(rBuf.Addr + remoteOffset, remoteMR.Key, immData, localMR, wr_id, localData, sz);
    }

    void UpdateCompletionCounters(yint deviceId, yint chainId)
    {
        TDeviceCtx &ctx = *DeviceArr[deviceId];
        TChain &chain = ChainArr[chainId];
        TRdmaCompletionCounters &rcc = *(TRdmaCompletionCounters *)ctx.HostMemBuf.data();
        Y_VERIFY(sizeof(rcc) <= ctx.HostMemBuf.size());
        rcc.RecvCompleteId = chain.RecvCompletion.DeviceArr[deviceId].MinRecvId;
        rcc.SendCompleteId = chain.SendCompletion.DeviceArr[deviceId].MinRecvId;
        //
        TIntrusivePtr<TMemoryRegion> mr = ctx.AsyncPoolMR;
        ui64 dstAddr = GetAddr64(chain.CompleteIdArr[deviceId]);
        ctx.QPself->PostRDMAWrite(dstAddr, ctx.AsyncPoolMR->GetRKey(), ctx.HostMemRegion, 0, &rcc, sizeof(rcc));
    }

    void ProcessIncoming() override
    {
        yint deviceCount = YSize(DeviceArr);
        ibv_wc wcArr[100];
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            TDeviceCtx &ctx = *DeviceArr[deviceId];
            int rv = ctx.CQ->Poll(wcArr, ARRAY_SIZE(wcArr));
            for (int k = 0; k < rv; ++k) {
                const ibv_wc &wc = wcArr[k];
                if (wc.status != IBV_WC_SUCCESS) {
                    DebugPrintf("rdma fail, status %d", (int)wc.status);
                    abort();
                }
                if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                    yint from = ctx.GetRankByQPN(wc.qp_num);
                    TLink &link = ctx.Links[from];
                    yint chainId = wc.imm_data >> 16;
                    yint cmdIdTal = wc.imm_data & 0xffff;
                    if (cmdIdTal == 0) {
                        link.CmdIdAdd[chainId] += 1ull << 16; // hopefully works
                    }
                    ui64 cmdId = link.CmdIdAdd[chainId] + cmdIdTal;
                    Y_VERIFY((cmdId == link.PrevCmdId[chainId] + 1 || link.PrevCmdId[chainId] == 0) && "we expect cmdId to increase by 1");
                    link.PrevCmdId[chainId] = cmdId;
                    if (ChainArr[chainId].RecvCompletion.OnComplete(deviceId, from, cmdId)) {
                        UpdateCompletionCounters(deviceId, chainId);
                    }
                    ctx.PostRecv();

                } else if (wc.opcode == IBV_WC_RDMA_WRITE) {
                    if (wc.qp_num == ctx.QPself->GetQPN()) {
                        // successful rcc update
                    } else {
                        yint to = ctx.GetRankByQPN(wc.qp_num);
                        ui64 cmdId = wc.wr_id & ((1ull << 48) - 1);
                        yint chainId = wc.wr_id >> 48;
                        if (ChainArr[chainId].SendCompletion.OnComplete(deviceId, to, cmdId)) {
                            UpdateCompletionCounters(deviceId, chainId);
                        }
                    }

                } else {
                    Y_ASSERT(0);
                }
            }
        }
    }
};


TIntrusivePtr<IRdmaTransport> CreateInfinibandRdmaTransfer(TPtrArg<ITcpSendRecv> net, yint chainCount)
{
    return new TInfinibandRdmaTransport(net, chainCount);
}

}
