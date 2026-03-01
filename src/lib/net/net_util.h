#pragma once

namespace NNet
{
SOCKET CreateDatagramSocket();
SOCKET CreateStreamSocket();
void MakeNonBlocking(SOCKET s);
void SetNoTcpDelay(SOCKET s);
void AllowDualStack(SOCKET s);
}
