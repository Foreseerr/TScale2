#pragma once
#include <lib/cuda/multi_device_buf.h>

namespace NCuda
{
class TGraph;
}

namespace NNet
{
struct INetReducer : public TThrRefBase
{
    virtual void PrepareConnections(TVector<char> *p) = 0;
    virtual void EstablishConnections(yint myRank, const TVector<TString> &peerList, TVector<TVector<char>> &handshakeArr) = 0;
    virtual void InitDevice(yint deviceId, TPtrArg<NCuda::TCudaMemoryPool> asyncIOPool) = 0;
    virtual void OnInitComplete() = 0;
    virtual void RegisterBuffer(TPtrArg<NCuda::TMultiDevice2DArray<float>> buf) = 0;
    virtual void AllReduce(
        TPtrArg<NCuda::TGraph> c, yint chainId, yint deviceId, NCuda::TCudaSpan ySpan, TPtrArg<NCuda::TMultiDevice2DArray<float>> p) = 0;
    virtual yint GetChainCount() = 0;
};

struct ITcpSendRecv;
TIntrusivePtr<INetReducer> CreateTcpReducer(TPtrArg<ITcpSendRecv> net, yint chainCount);
TIntrusivePtr<INetReducer> CreateInfinibandReducer(TPtrArg<ITcpSendRecv> net, yint chainCount);
}
