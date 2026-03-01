#include <util/pch.h>
#include "sm90.cuh"


namespace NCuda
{
bool IgnoreSm90Kernels = false;
static yint IsSm90  = -1;

bool UseSm90Kernels()
{
    if (IgnoreSm90Kernels) {
        return false;
    }
    if (IsSm90 < 0) {
        cudaDeviceProp deviceProp;
        Y_VERIFY(cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0)); // query first device, assume all are equal
        IsSm90 = (deviceProp.major == 9) && (deviceProp.minor == 0);
    }
    return IsSm90;
}
}
