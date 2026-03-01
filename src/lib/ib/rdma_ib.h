#pragma once
#include "rdma.h"


namespace NNet 
{
struct ITcpSendRecv;
TIntrusivePtr<IRdmaTransport> CreateInfinibandRdmaTransfer(TPtrArg<ITcpSendRecv> net, yint chainCount);
}
