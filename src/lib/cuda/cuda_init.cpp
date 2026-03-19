#include "cuda_init.h"
#include <cuda_runtime.h>


yint GetCudaDeviceCount()
{
    int res = 0;
    Y_VERIFY(cudaSuccess == cudaGetDeviceCount(&res));
    return res;
}


static __thread yint CurrentDeviceId;

void CudaSetDevice(yint deviceId)
{
    if (SIMULATE_MULTI_GPU) {
        deviceId = 0;
    }
    CurrentDeviceId = deviceId;
    Y_VERIFY(cudaSetDevice(deviceId) == cudaSuccess);
}


yint CudaGetDevice()
{
    return CurrentDeviceId;
}

yint GetCudaSMCount()
{
    int deviceId;
    Y_VERIFY(cudaSuccess == cudaGetDevice(&deviceId));
    cudaDeviceProp deviceProp;
    Y_VERIFY(cudaSuccess == cudaGetDeviceProperties(&deviceProp, deviceId));
    return deviceProp.multiProcessorCount;
}

bool CudaCanEnablePeerAccess()
{
    if (SIMULATE_MULTI_GPU) {
        return true;
    }
    yint deviceCount = GetCudaDeviceCount();
    if (deviceCount < 2) {
        return true;
    }
    bool ok = true;
    for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
        for (yint peerId = 0; peerId < deviceCount; ++peerId) {
            if (peerId != deviceId) {
                int canAccess = 0;
                Y_VERIFY(cudaDeviceCanAccessPeer(&canAccess, deviceId, peerId) == cudaSuccess);
                ok &= (canAccess == 1);
            }
        }
    }
    return ok;
}

static std::atomic<yint> PeerAccessOk;
void CudaEnablePeerAccess()
{
    enum {
        ACCESS_NO = 0,
        ACCESS_WIP = 1,
        ACCESS_OK = 2,
    };
    if (SIMULATE_MULTI_GPU) {
        return;
    }
    yint prevDeviceOk = PeerAccessOk.exchange(ACCESS_WIP);
    if (prevDeviceOk == ACCESS_NO) {
        // enable
        int keepDeviceId = 0;
        cudaGetDevice(&keepDeviceId);
        yint deviceCount = GetCudaDeviceCount();
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            cudaSetDevice(deviceId);
            for (yint peerId = 0; peerId < deviceCount; ++peerId) {
                if (peerId != deviceId) {
                    Y_VERIFY(cudaDeviceEnablePeerAccess(peerId, 0) == cudaSuccess);
                }
            }
        }
        cudaSetDevice(keepDeviceId);
        PeerAccessOk = ACCESS_OK;
    } if (prevDeviceOk == ACCESS_OK) {
        // already enabled
        PeerAccessOk = ACCESS_OK;
    } else {
        // wait other thread
        while (PeerAccessOk == ACCESS_WIP) {
            SchedYield();
        }
    }
}
