#pragma once

struct sockaddr_in6;

namespace NNet
{
inline sockaddr_in6 MakeAcceptSockAddr(ui16 port)
{
    sockaddr_in6 name;
    Zero(name);
    name.sin6_family = AF_INET6;
    name.sin6_addr = in6addr_any;
    name.sin6_port = htons(port);
    return name;
}


struct TIPAddress
{
    ui64 Network, Interface;
    int Scope, Port;

    TIPAddress() : Network(0), Interface(0), Scope(0), Port(0) {}
    TIPAddress(const sockaddr_in6 &addr);
    sockaddr_in6 GetSockAddr();
    TString GetAddressString() const;
    bool IsIPv4() const { return (Network == 0 && (Interface & 0xffffffffll) == 0xffff0000ll); }
    ui32 GetIPv4() const { return Interface >> 32; }
};

inline bool operator==(const TIPAddress &a, const TIPAddress &b)
{
    return a.Network == b.Network && a.Interface == b.Interface && a.Scope == b.Scope && a.Port == b.Port;
}


timeval MakeTimeval(float timeoutSec);
bool ParseInetName(sockaddr_in6 *pRes, const char *name, int nDefaultPort);
void StripPort(TString *pszPureHost);
void ReplacePort(TString *pszPureHost, int newPort);
TString GetHostName();
}
