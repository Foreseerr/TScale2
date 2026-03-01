#include "ip_address.h"


namespace NNet
{

TIPAddress::TIPAddress(const sockaddr_in6 &addr)
{
    if (addr.sin6_family == AF_INET) {
        const sockaddr_in &addr4 = *(const sockaddr_in *)&addr;
        Network = 0;
        Interface = 0xffff0000ll + (((ui64)(ui32)addr4.sin_addr.s_addr) << 32);
        Scope = 0;
        Port = ntohs(addr4.sin_port);
    } else if (addr.sin6_family == AF_INET6) {
        Network = *(ui64 *)(addr.sin6_addr.s6_addr + 0);
        Interface = *(ui64 *)(addr.sin6_addr.s6_addr + 8);
        Scope = addr.sin6_scope_id;
        Port = ntohs(addr.sin6_port);
    } else {
        Y_ASSERT(0);
        *this = TIPAddress();
    }
}


sockaddr_in6 TIPAddress::GetSockAddr()
{
    sockaddr_in6 res;
    Zero(res);
    res.sin6_family = AF_INET6;
    *(ui64 *)(res.sin6_addr.s6_addr + 0) = Network;
    *(ui64 *)(res.sin6_addr.s6_addr + 8) = Interface;
    res.sin6_scope_id = Scope;
    res.sin6_port = htons((u_short)Port);
    return res;
}


TString TIPAddress::GetAddressString() const
{
    TString res;
    if (IsIPv4()) {
        int ip = GetIPv4();
        res = Sprintf("%d.%d.%d.%d:%d", (ip >> 0) & 0xff, (ip >> 8) & 0xff, (ip >> 16) & 0xff, (ip >> 24) & 0xff, Port);
    } else {
        ui16 ipv6[8];
        *(ui64 *)(ipv6) = Network;
        *(ui64 *)(ipv6 + 4) = Interface;
        TString suffix;
        if (Scope != 0) {
            suffix = Sprintf("%%%d", Scope);
        }
        res = Sprintf("[%x:%x:%x:%x:%x:%x:%x:%x%s]:%d", ntohs(ipv6[0]), ntohs(ipv6[1]), ntohs(ipv6[2]), ntohs(ipv6[3]), ntohs(ipv6[4]),
            ntohs(ipv6[5]), ntohs(ipv6[6]), ntohs(ipv6[7]), suffix.c_str(), Port);
    }
    return res;
}


////////////////////////////////////////////////////////////////////////////////////
timeval MakeTimeval(float timeoutSec)
{
    int timeoutSecint = timeoutSec;
    timeval tvTimeout = { timeoutSecint, static_cast<long>((timeoutSec - timeoutSecint) * 1000000) };
    return tvTimeout;
}


static bool IsValidIPv6(const char *sz)
{
    enum {
        S1,
        SEMICOLON,
        SCOPE
    };
    int state = S1, scCount = 0, digitCount = 0, hasDoubleSemicolon = false;
    while (*sz) {
        if (state == S1) {
            switch (*sz) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            case 'A':
            case 'B':
            case 'C':
            case 'D':
            case 'E':
            case 'F':
            case 'a':
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'f':
                ++digitCount;
                if (digitCount > 4)
                    return false;
                break;
            case ':':
                state = SEMICOLON;
                ++scCount;
                break;
            case '%':
                state = SCOPE;
                break;
            default:
                return false;
            }
            ++sz;
        } else if (state == SEMICOLON) {
            if (*sz == ':') {
                if (hasDoubleSemicolon)
                    return false;
                hasDoubleSemicolon = true;
                ++scCount;
                digitCount = 0;
                state = S1;
                ++sz;
            } else {
                digitCount = 0;
                state = S1;
            }
        } else if (state == SCOPE) {
            // arbitrary string is allowed as scope id
            ++sz;
        }
    }
    if (!hasDoubleSemicolon && scCount != 7)
        return false;
    return scCount <= 7;
}


bool ParseInetName(sockaddr_in6 *pRes, const char *name, int nDefaultPort)
{
    Zero(*pRes);
    int nPort = nDefaultPort;

    TString host;
    if (name[0] == '[') {
        ++name;
        const char *nameFin = name;
        for (; *nameFin; ++nameFin) {
            if (nameFin[0] == ']') {
                break;
            }
        }
        host.assign(name, nameFin);
        if (!IsValidIPv6(host.c_str())) {
            Y_ASSERT(0);
            return false;
        }
        name = *nameFin ? nameFin + 1 : nameFin;
        if (name[0] == ':') {
            char *endPtr = NULL;
            nPort = strtol(name + 1, &endPtr, 10);
            if (!endPtr || *endPtr != '\0')
                return false;
        }
    } else {
        host = name;
        if (!IsValidIPv6(name)) {
            size_t nIdx = host.find(':');
            if (nIdx != (size_t)TString::npos) {
                const char *pszPort = host.c_str() + nIdx + 1;
                char *endPtr = NULL;
                nPort = strtol(pszPort, &endPtr, 10);
                if (!endPtr || *endPtr != '\0')
                    return false;
                host.resize(nIdx);
            }
        }
    }

    addrinfo aiHints;
    Zero(aiHints);
    aiHints.ai_family = AF_UNSPEC;
    aiHints.ai_socktype = SOCK_STREAM;
    aiHints.ai_protocol = IPPROTO_TCP;

    static TAtomic cs;
    TGuard<TAtomic> lock(cs);

    addrinfo *aiList = 0;
    for (int attempt = 0; attempt < 1000; ++attempt) {
        int rv = getaddrinfo(host.c_str(), "31331", &aiHints, &aiList);
        if (rv == 0)
            break;
        if (aiList) {
            freeaddrinfo(aiList);
        }
        if (rv != EAI_AGAIN) {
            return false;
        }
        SleepSeconds(0.1f);
    }
    for (addrinfo *ptr = aiList; ptr; ptr = ptr->ai_next) {
        sockaddr *addr = ptr->ai_addr;
        if (addr == 0) {
            continue;
        }
        if (addr->sa_family != AF_INET && addr->sa_family != AF_INET6) {
            continue;
        }

        TIPAddress resAddr(*(sockaddr_in6 *)addr);
        *pRes = resAddr.GetSockAddr();
        pRes->sin6_port = htons(nPort);

        freeaddrinfo(aiList);
        return true;
    }
    freeaddrinfo(aiList);
    return false;
}


void ReplacePort(TString *pAddr, int newPort)
{
    // strip port
    if (!pAddr->empty()) {
        if ((*pAddr)[0] == '[') {
            // we expect ip v6 address in the form [0::0]:port
            size_t nIdx = pAddr->find(']');
            if (nIdx != TString::npos) {
                pAddr->resize(nIdx + 1);
            }
        } else {
            // regular address
            size_t nIdx = pAddr->find(':');
            if (nIdx != TString::npos) {
                pAddr->resize(nIdx);
            }
        }
    }
    *pAddr += Sprintf(":%d", newPort);
}


TString GetHostName()
{
    char buf[10000];
    if (gethostname(buf, ARRAY_SIZE(buf) - 1)) {
        Y_ASSERT(0);
        return "???";
    }
    return buf;
}


}
