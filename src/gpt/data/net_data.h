#pragma once
#include "data.h"
#include <lib/net/tcp_net.h>

TIntrusivePtr<IDataSource> ConnectDataServer(TPtrArg<NNet::ITcpSendRecv> net, const TString &addr);
TIntrusivePtr<IDataSource> ConnectHttpDataServer(const TString &addr);
void RunDataServer(TPtrArg<NNet::ITcpSendRecv> net, TPtrArg<IDataSource> data);
