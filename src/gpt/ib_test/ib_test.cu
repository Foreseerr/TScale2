#include <util/pch.h>
#define KERNEL_UNIT "ib_test/"
#include <lib/ib/ib_reducer.h>
#include <lib/ib/ib_buffers.h>
#include <lib/ib/ib_low.h>
#include <cuda.h>
#include <lib/config/config.h>
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_init.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/net/ip_address.h>
#include <lib/net/p2p.h>
#include <lib/hp_timer/hp_timer.h>


using namespace NNet;
using namespace NCuda;

static TGuid IBTestToken(0x466a8601, 0x8e605d32, 0xbc0f6e6e, 0x4cbe5edf);

///////////////////////////////////////////////////////////////////////////////////////////////////
//
namespace NIBTest 
{
struct TTestBuffers : public TThrRefBase
{
    TIntrusivePtr<TMultiDevice2DArray<float>> Buf;
    TCuda2DArray<float> HostBuf;
};

struct TIbTestContext
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TMasterConnection Master;
    TIntrusivePtr<INetReducer> NetReducer;
    TIntrusivePtr<TCudaMemoryAllocator> CudaAllocator;
    TIntrusivePtr<TCudaMemoryPool> Pool;
    TIntrusivePtr<TMultiDeviceBufferFabric> Fabric;
    TVector<TIntrusivePtr<TTestBuffers>> BufArr;
};

struct TCommandPacket : public TCommandBase
{
    virtual void Exec(TIbTestContext *pCtx) {}
};

static TCommandFabric<TCommandPacket> cmdFabric;

///////////////////////////////////////////////////////////////////////////////////////////////////
class THandshake : public TCommandPacket
{
    bool UseInfiniband = false;
    yint ChainCount = 0;
    yint XSize = 0;
    yint YSize = 0;
    SAVELOAD_OVERRIDE(UseInfiniband, ChainCount, XSize, YSize);

public:
    THandshake() {}
    THandshake(bool ib, yint chainCount, yint xSize, yint ySize) : UseInfiniband(ib), ChainCount(chainCount), XSize(xSize), YSize(ySize) {}
    void Exec(TIbTestContext *p) override
    {
        if (UseInfiniband) {
            p->NetReducer = CreateInfinibandReducer(p->Net, ChainCount);
        } else {
            p->NetReducer = CreateTcpReducer(p->Net, ChainCount);
        }
        TVector<ui8> hs;
        p->NetReducer->PrepareConnections(&hs);
        p->Master.Send(hs);
        p->BufArr.resize(ChainCount);
        for (yint k = 0; k < ChainCount; ++k) {
            TTestBuffers *tb = new TTestBuffers();
            tb->Buf = p->Fabric->Create2DArray<float>(Sprintf("buf-%d", (int)k));
            tb->Buf->AllocateCuda(0, XSize, YSize, p->Pool);
            tb->HostBuf.AllocateHost(XSize, YSize);
            p->BufArr[k] = tb;
        }
    }
};
REGISTER_PACKET(cmdFabric, THandshake, 1);


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
    void Exec(TIbTestContext *p) override
    {
        TVector<ui8> hs;
        p->NetReducer->EstablishConnections(Rank, PeerList, HandshakeArr);
        p->NetReducer->InitDevice(0, p->Pool);
        for (yint k = 0; k < YSize(p->BufArr); ++k) {
            p->NetReducer->RegisterBuffer(p->BufArr[k]->Buf);
        }
        p->CudaAllocator->AllocateMemory();
        p->NetReducer->OnInitComplete();
        p->Master.SendCopy((int)0);
    }
};
REGISTER_PACKET(cmdFabric, TEstablishConnections, 2);


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline void CheckMatch(const TArray2D<T> &a, const TArray2D<T> &b)
{
    yint xSize = a.GetXSize();
    yint ySize = a.GetYSize();
    Y_ASSERT(xSize == b.GetXSize() && ySize == b.GetYSize());
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            if (a[y][x] != b[y][x]) {
                DebugPrintf("mismatch at %g %g, %g != %g\n", x * 1., y * 1., a[y][x], b[y][x]);
                return;
            }
        }
    }
}

class TTestReducer : public TCommandPacket
{
    bool TestCorrect = true;
    yint ChainCount = 0;
    yint Mult = 0;
    yint Total = 0;
    SAVELOAD_OVERRIDE(TestCorrect, ChainCount, Mult, Total);

public:
    TTestReducer() {}
    TTestReducer(bool testCorrect, yint chainCount, yint mult, yint total)
        : TestCorrect(testCorrect), ChainCount(chainCount), Mult(mult), Total(total)
    {
    }
    void Exec(TIbTestContext *p) override
    {
        TIbTestContext &ctx = *p;
        yint xSize = ctx.BufArr[0]->HostBuf.GetXSize();
        yint ySize = ctx.BufArr[0]->HostBuf.GetYSize();

        TIntrusivePtr<TGraph> c = new TGraph;
        {
            c->SetMemPool(ctx.Pool);
            yint deviceId = 0;
            TCudaSpan ySpan(0, ySize);
            for (yint chainId = 0; chainId < ChainCount; ++chainId) {
                TTestBuffers &tb = *ctx.BufArr[chainId];
                TCuda2DArray<float> &buf = tb.Buf->GetData(0);
                if (TestCorrect) {
                    c->KernelCopy(&buf, tb.HostBuf);
                    ctx.NetReducer->AllReduce(c, chainId, deviceId, ySpan, tb.Buf);
                    c->KernelCopy(&tb.HostBuf, buf);
                } else {
                    ctx.NetReducer->AllReduce(c, chainId, deviceId, ySpan, tb.Buf);
                }
            }
        }
        TStream stream;
        TArray2D<float> arg;
        TArray2D<float> refSum;
        {
            arg.SetSizes(xSize, ySize);
            refSum.SetSizes(xSize, ySize);
            for (yint y = 0; y < ySize; ++y) {
                for (yint x = 0; x < xSize; ++x) {
                    float val = (y * xSize + x);
                    arg[y][x] = val * Mult;
                    refSum[y][x] = val * Total;
                }
            }
        }
        // run
        for (yint iter = 1;; ++iter) {
            for (yint chainId = 0; chainId < ChainCount; ++chainId) {
                TTestBuffers &tb = *ctx.BufArr[chainId];
                PutHost(&tb.HostBuf, arg);
            }

            NHPTimer::STime tStart;
            NHPTimer::GetTime(&tStart);
            c->Run(stream);
            stream.Sync();
            double tPassed = NHPTimer::GetTimePassed(&tStart);

            if (TestCorrect) {
                for (yint chainId = 0; chainId < ChainCount; ++chainId) {
                    TTestBuffers &tb = *ctx.BufArr[chainId];
                    TArray2D<float> chk;
                    GetAllData(tb.HostBuf, &chk);
                    CheckMatch(chk, refSum);
                }
            }
            DebugPrintf("iter %g, %g gb/sec\n", iter * 1., sizeof(float) * xSize * ySize * ChainCount / tPassed / 1e9);
        }
        p->Master.SendCopy((int)0);
    }
};
REGISTER_PACKET(cmdFabric, TTestReducer, 3);


///////////////////////////////////////////////////////////////////////////////////////////////////
void RunWorker(yint port)
{
    TIbTestContext ctx;
    ctx.Net = CreateTcpSendRecv();
    ctx.Master.ConnectMaster(ctx.Net, port, IBTestToken);
    ctx.Fabric = new TMultiDeviceBufferFabric();
    ctx.CudaAllocator = new TCudaMemoryAllocator();
    ctx.Pool = ctx.CudaAllocator->CreatePool();
    ctx.Pool->UseForAsyncIO(true);

    DebugPrintf("executing incoming commands\n");
    for (;;) {
        TIntrusivePtr<TCommandPacket> cmd = RecvCommand(cmdFabric, ctx.Master.GetQueue());
        if (cmd.Get()) {
            // DebugPrintf("Worker got command %s\n", typeid(*cmd.Get()).name());
            cmd->Exec(&ctx);
        }
    }
}


void RunMaster(const TVector<TString> &workerAddrArr, bool useInfiniband, bool testCorrect, yint chainCount)
{
    // constexpr yint xSize = 128;
    // constexpr yint ySize = 128;
    constexpr yint xSize = 4096;
    constexpr yint ySize = 1024;
    yint workerCount = YSize(workerAddrArr);

    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    TMasterNetTempl<TCommandPacket> masterNet(cmdFabric, net);
    masterNet.ConnectWorkers(workerAddrArr, IBTestToken);

    DebugPrintf("rdma connect\n");
    TVector<TVector<ui8>> handshakeArr;
    masterNet.BroadcastCommand(new THandshake(useInfiniband, chainCount, xSize, ySize), &handshakeArr);
    for (yint rank = 0, total = masterNet.GetWorkerCount(); rank < total; ++rank) {
        masterNet.SendCommand(masterNet.Workers[rank], new TEstablishConnections(rank, workerAddrArr, handshakeArr));
    }
    TVector<int> cmdResults;
    masterNet.CollectCommandResults(&cmdResults);

    for (yint rank = 0; rank < workerCount; ++rank) {
        yint total = (workerCount * (workerCount + 1)) / 2;
        masterNet.SendCommand(masterNet.Workers[rank], new TTestReducer(testCorrect, chainCount, rank + 1, total));
    }
    masterNet.CollectCommandResults(&cmdResults);
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    TString workerPort = "10000";
    TString workerFileName = GetHomeDir() + "workers.txt";
#ifdef PLATFORM_HAS_IBVERBS
    bool useInfiniband = true;
#else
    bool useInfiniband = false;
#endif
    yint chainCount = 1;
    bool testCorrect = true;

    TOpt cmdline("w:m:r:c:t:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "w") {
            workerPort = param.Args[0];
        } else if (param.Name == "m") {
            workerFileName = param.Args[0];
        } else if (param.Name == "r") {
            useInfiniband = (atoi(param.Args[0].c_str()) != 0);
        } else if (param.Name == "c") {
            chainCount = atoi(param.Args[0].c_str());
        } else if (param.Name == "t") {
            testCorrect = (atoi(param.Args[0].c_str()) != 0);
        }
    }
    if (workerPort.empty() || workerPort == "master") {
        TVector<TString> workerArr;
        ReadNonEmptyLines(&workerArr, workerFileName);
        NIBTest::RunMaster(workerArr, useInfiniband, testCorrect, chainCount);
    } else {
        NIBTest::RunWorker(atoi(workerPort.c_str()));
    }
    return 0;
}
