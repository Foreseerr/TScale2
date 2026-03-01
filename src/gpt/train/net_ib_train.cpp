#include "net_ib_train.h"
#include "net_common.h"
#include <gpt/train_ctx/train_ctx.h>
#include <lib/net/p2p.h>
#include <lib/ib/ib_reducer.h>


using namespace NNet;

namespace NNetIbTrain
{

static TGuid NetTrainToken(0x0c5728e0, 0x1a773572, 0x07001e45, 0x00a074b6);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TNetTrainContext;
struct TCommandPacket : public TCommandBase
{
    typedef TNetTrainContext TExecCtx;

    virtual void Exec(TNetTrainContext *pCtx) {}
};

typedef TMasterNetTempl<TCommandPacket> TMasterNet;

static TCommandFabric<TCommandPacket> cmdFabric;


///////////////////////////////////////////////////////////////////////////////////////////////////
//
class TMMNetDeltaReduceGen;
struct TNetTrainContext
{
    yint Rank = 0;
    yint RankCount = 0;
    TIntrusivePtr<ITcpSendRecv> Net;
    TMasterConnection Master;
    TIntrusivePtr<INetReducer> NetReducer;
    TIntrusivePtr<IModel> Model;
    TVector<ui8> ModelSnapshot;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
REGISTER_PACKET(cmdFabric, TCalcModelError<TCommandPacket>, 1);
REGISTER_PACKET(cmdFabric, TBackprop<TCommandPacket>, 2);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCreateModel : public TCommandPacket
{
    yint DeviceCount = 0;
    TModelParams Params;
    yint DeviceMaxNodeCount = 0;
    TModelSplit MSplit;
    SAVELOAD_OVERRIDE(DeviceCount, Params, DeviceMaxNodeCount, MSplit);

public:
    TCreateModel() {}
    TCreateModel(yint deviceCount, const TModelParams &params, yint deviceMaxNodeCount, const TModelSplit &msplit)
        : DeviceCount(deviceCount), Params(params), DeviceMaxNodeCount(deviceMaxNodeCount), MSplit(msplit)
    {
    }
    void Exec(TNetTrainContext *p) override
    {
        p->Model = CreateClusterTransformer(Params, DeviceCount, DeviceMaxNodeCount, p->Rank, p->RankCount, MSplit, p->NetReducer);
        p->NetReducer->OnInitComplete();
        p->Master.SendCopy(CMD_OK);
    }
};
REGISTER_PACKET(cmdFabric, TCreateModel, 3);


///////////////////////////////////////////////////////////////////////////////////////////////////
class THandshake : public TCommandPacket
{
public:
    void Exec(TNetTrainContext *p) override
    {
        constexpr yint CHAIN_COUNT = 2;
        TVector<ui8> hs;
#ifdef PLATFORM_HAS_IBVERBS
        p->NetReducer = CreateInfinibandReducer(p->Net, CHAIN_COUNT);
#else
        p->NetReducer = CreateTcpReducer(p->Net, CHAIN_COUNT);
#endif
        p->NetReducer->PrepareConnections(&hs);
        p->Master.Send(hs);
    }
};
REGISTER_PACKET(cmdFabric, THandshake, 4);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TEstablishConnections : public TCommandPacket
{
    yint Rank = 0;
    TVector<TString> PeerList;
    TVector<TVector<ui8>> HandshakeArr;
    SAVELOAD_OVERRIDE(Rank, PeerList, HandshakeArr);

public:
    TEstablishConnections() {}
    TEstablishConnections(yint rank, const TVector<TString> &peerList, const TVector<TVector<ui8>> &handshakeArr)
        : Rank(rank), PeerList(peerList), HandshakeArr(handshakeArr)
    {
    }
    void Exec(TNetTrainContext *p) override
    {
        TVector<ui8> hs;
        p->NetReducer->EstablishConnections(Rank, PeerList, HandshakeArr);
        p->Master.SendCopy(CMD_OK);
        p->Rank = Rank;
        p->RankCount = YSize(PeerList);
    }
};
REGISTER_PACKET(cmdFabric, TEstablishConnections, 5);


///////////////////////////////////////////////////////////////////////////////////////////////////
REGISTER_PACKET(cmdFabric, TWaitDelayedUpdates<TCommandPacket>, 8);
REGISTER_PACKET(cmdFabric, TMakeParamsSnapshot<TCommandPacket>, 9);
REGISTER_PACKET(cmdFabric, TGetParamsSnapshotFragment<TCommandPacket>, 10);


///////////////////////////////////////////////////////////////////////////////////////////////////
void RunWorker(yint port)
{
    TNetTrainContext ctx;
    ctx.Net = CreateTcpSendRecv();
    ctx.Master.ConnectMaster(ctx.Net, port, NetTrainToken);
    //ctx.P2PNet = new TP2PNetwork(ctx.Net, NetTrainToken);
    DebugPrintf("executing incoming commands\n");
    for (;;) {
        TIntrusivePtr<TCommandPacket> cmd = RecvCommand(cmdFabric, ctx.Master.GetQueue());
        if (cmd.Get()) {
            //DebugPrintf("Worker got command %s\n", typeid(*cmd.Get()).name());
            cmd->Exec(&ctx);
        }
    }
}


void RunMaster(yint startIteration, yint deviceCount, const TModelSplit &msplit, const TVector<TString> &workerAddrArr,
    const TTrainContext &trainCtx, TIntrusivePtr<TModelParamsHolder> pParams)
{
    const THostBatchConfig &bc = trainCtx.GetBatchConfig();
    yint workerCount = YSize(workerAddrArr);
    TVector<ECommandResult> cmdResults;

    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    TMasterNet masterNet(cmdFabric, net);
    masterNet.ConnectWorkers(workerAddrArr, NetTrainToken);

    DebugPrintf("rdma connect\n");
    TVector<TVector<ui8>> handshakeArr;
    masterNet.BroadcastCommand(new THandshake(), &handshakeArr);
    for (yint rank = 0, total = masterNet.GetWorkerCount(); rank < total; ++rank) {
        masterNet.SendCommand(masterNet.Workers[rank], new TEstablishConnections(rank, workerAddrArr, handshakeArr));
    }
    masterNet.CollectCommandResults(&cmdResults);

    // requires established connections
    DebugPrintf("create model\n");
    masterNet.BroadcastCommand(new TCreateModel(deviceCount, pParams->Params, bc.GetInstanceMaxNodeCount(), msplit), &cmdResults);
    pParams = 0;

    NetTrainModel(masterNet, trainCtx, startIteration);
}
}
