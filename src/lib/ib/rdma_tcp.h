#pragma once
#include "rdma.h"


namespace NNet
{
struct ITcpSendRecv;
TIntrusivePtr<IRdmaTransport> CreateTcpRdmaTransfer(TPtrArg<ITcpSendRecv> net, yint maxTransferSize, yint chainCount);
}
