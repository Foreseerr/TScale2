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


static std::atomic<yint> PeerAccessOk;
void CudaEnablePeerAccess()
{
    if (SIMULATE_MULTI_GPU) {
        return;
    }
    yint prevDeviceOk = PeerAccessOk.exchange(1);
    if (prevDeviceOk == 0) {
        int keepDeviceId = 0;
        cudaGetDevice(&keepDeviceId);
        yint deviceCount = GetCudaDeviceCount();
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            cudaSetDevice(deviceId);
            for (yint k = 0; k < deviceCount; ++k) {
                if (k != deviceId) {
                    Y_VERIFY(cudaDeviceEnablePeerAccess(k, 0) == cudaSuccess);
                }
            }
        }
        cudaSetDevice(keepDeviceId);
    }
}
