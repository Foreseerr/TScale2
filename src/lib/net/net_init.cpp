#include "net_init.h"


#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")

static std::atomic<yint> NetIsInitialized;

void NetInit()
{
    while (NetIsInitialized != 1) {
        yint val = 0;
        if (NetIsInitialized.compare_exchange_strong(val, 2)) {
            WSADATA wsaData;
            Y_VERIFY(WSAStartup(MAKEWORD(2, 2), &wsaData) == 0);
            NetIsInitialized = 1;
        }
    }
}
#else
void NetInit()
{
}
#endif
