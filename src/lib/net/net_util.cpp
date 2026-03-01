#include "net_util.h"

#ifndef _win_
#include <fcntl.h>
#endif


namespace NNet
{

SOCKET CreateDatagramSocket()
{
    SOCKET sock = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        DebugPrintf("socket() failed\n");
        abort();
    }
    NNet::AllowDualStack(sock);
    return sock;
}


SOCKET CreateStreamSocket()
{
    SOCKET sock = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        DebugPrintf("socket() failed\n");
        abort();
    }
    NNet::AllowDualStack(sock);
    return sock;
}


void MakeNonBlocking(SOCKET s)
{
#if defined(_win_)
    unsigned long dummy = 1;
    ioctlsocket(s, FIONBIO, &dummy);
#else
    fcntl(s, F_SETFL, O_NONBLOCK);
    // added to prevent socket duplication into child processes
    fcntl(s, F_SETFD, FD_CLOEXEC);
#endif
}

void SetNoTcpDelay(SOCKET s)
{
    int flag = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
}

void AllowDualStack(SOCKET s)
{
    int v6Only = 0; // 0 for dual-stack, 1 for IPv6-only
    int rv = setsockopt(s, IPPROTO_IPV6, IPV6_V6ONLY, (char *)&v6Only, sizeof(v6Only));
    Y_ASSERT(rv == 0);
}
}
