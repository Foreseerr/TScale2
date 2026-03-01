#pragma once
#include <lib/net/tcp_net.h>
#include <lib/net/tcp_cmds.h>

namespace NNet
{

///////////////////////////////////////////////////////////////////////////////////////////////////
const yint DEFAULT_WORKER_PORT = 10000;

typedef int TNetRank;


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void SendData(TPtrArg<ITcpSendRecv> net, TPtrArg<ITcpConnection> conn, T &x)
{
    TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
    SerializeMem(IO_WRITE, &pkt->Data, x);
    net->Send(conn, pkt);
}


template <class TRes>
static void WaitData(TPtrArg<TTcpRecvQueue> q, TPtrArg<ITcpConnection> conn, TRes *pRes)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    while (!q->Dequeue(&pkt)) {
        SchedYield(); // lag?
    }
    Y_VERIFY(pkt->Conn == conn);
    SerializeMem(IO_READ, &pkt->Data, *pRes);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMasterNetBase
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<TTcpRecvQueue> Queue;
    THashMap<TIntrusivePtr<ITcpConnection>, TNetRank> WorkerSet;
    TVector<TIntrusivePtr<ITcpConnection>> Workers;

public:
    TMasterNetBase(TPtrArg<ITcpSendRecv> net) : Net(net) { Queue = new TTcpRecvQueue; }
    void ConnectWorkers(const TVector<TString> &workerList, const TGuid &token);
    yint GetWorkerCount() const { return YSize(Workers); }
};


template <class TCmd>
struct TMasterNetTempl : public TMasterNetBase
{
    TCommandFabric<TCmd> &CmdFabric;

public:
    TMasterNetTempl(TCommandFabric<TCmd> &cmdFabric, TPtrArg<ITcpSendRecv> net) : CmdFabric(cmdFabric), TMasterNetBase(net) {}

    template <class TRet>
    void CollectCommandResults(TVector<TRet> *pResArr)
    {
        yint workerCount = YSize(WorkerSet);
        pResArr->resize(workerCount);
        yint confirmCount = 0;
        while (confirmCount < workerCount) {
            TIntrusivePtr<TTcpPacketReceived> pkt;
            if (Queue->Dequeue(&pkt)) {
                auto it = WorkerSet.find(pkt->Conn);
                Y_ASSERT(it != WorkerSet.end());
                SerializeMem(IO_READ, &pkt->Data, (*pResArr)[it->second]);
                ++confirmCount;
            }
        }
    }

    template <class TArg, class TRet>
    void BroadcastCommand(TArg *cmdArg, TVector<TRet> *pResArr)
    {
        TIntrusivePtr<TTcpPacket> pkt = SerializeCommand(CmdFabric, cmdArg);
        for (auto it = WorkerSet.begin(); it != WorkerSet.end(); ++it) {
            Net->Send(it->first, pkt);
        }
        CollectCommandResults(pResArr);
    }

    template <class TArg>
    void SendCommand(TPtrArg<ITcpConnection> conn, TArg *cmdArg)
    {
        Net->Send(conn, SerializeCommand(CmdFabric, cmdArg));
    }

    template <class TRes>
    void DistributedExec(const TVector<TIntrusivePtr<TCmd>> &cmdArr, TVector<TRes> *pRes)
    {
        // execute all commands, even commands distribution
        yint sz = YSize(cmdArr);
        yint dstRank = 0;
        for (yint b = 0; b < sz; ++b) {
            SendCommand(Workers[dstRank], cmdArr[b].Get());
            if (++dstRank == YSize(Workers)) {
                dstRank = 0;
            }
        }
        // collect out of order results
        pRes->resize(sz);
        for (yint confirmCount = 0; confirmCount < sz;) {
            TIntrusivePtr<TTcpPacketReceived> pkt;
            if (Queue->Dequeue(&pkt)) {
                SerializeMem(IO_READ, &pkt->Data, (*pRes)[confirmCount]);
                ++confirmCount;
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMasterConnection
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<TTcpRecvQueue> Queue;
    TIntrusivePtr<ITcpConnection> Conn;
    TNetRank MyRank = 0;
public:
    void ConnectMaster(TPtrArg<ITcpSendRecv> net, yint port, const TGuid &token);

    TIntrusivePtr<TTcpRecvQueue> GetQueue() const
    {
        return Queue;
    }

    template <class T>
    void Send(T &data)
    {
        SendData(Net, Conn, data);
    }

    template <class T>
    void SendCopy(const T &dataArg)
    {
        T data = dataArg;
        SendData(Net, Conn, data);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TP2PNetwork : public TThrRefBase
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<ITcpAccept> Accept;
    TIntrusivePtr<TTcpRecvQueue> Queue;
    TVector<TIntrusivePtr<ITcpConnection>> Peers;
    TNetRank MyRank = 0;

public:
    TP2PNetwork(TPtrArg<ITcpSendRecv> net, const TGuid &token)
    {
        Net = net.Get();
        Queue = new TTcpRecvQueue;
        Accept = Net->StartAccept(0, token, nullptr);
    }

    yint GetPort() const { return Accept->GetPort(); }

    TNetRank GetMyRank() const { return MyRank; }

    yint GetWorkerCount() const { return YSize(Peers); }

    TIntrusivePtr<TTcpRecvQueue> GetQueue() const { return Queue; }

    yint GetRank(TPtrArg<ITcpConnection> conn) const
    {
        for (yint k = 0, sz = YSize(Peers); k < sz; ++k) {
            if (Peers[k] == conn) {
                return k;
            }
        }
        return -1;
    }

    void Send(TNetRank rank, TPtrArg<TTcpPacket> pkt);

    template <class T>
    void SendData(TNetRank rank, T &x)
    {
        TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
        SerializeMem(IO_WRITE, &pkt->Data, x);
        Send(rank, pkt);
    }

    void ConnectP2P(TNetRank myRank, const TVector<TString> &peerList, const TGuid &token);
};
}
