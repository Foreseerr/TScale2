#pragma once

#ifdef _MSC_VER
constexpr bool SIMULATE_MULTI_GPU = true;
#else
constexpr bool SIMULATE_MULTI_GPU = false;
#endif

constexpr int MAX_NUM_DEVICES = 8;

constexpr int BG_SM_COUNT = 16;

yint GetCudaDeviceCount();
void CudaSetDevice(yint deviceId);
yint CudaGetDevice();
yint GetCudaSMCount();
bool CudaCanEnablePeerAccess();
void CudaEnablePeerAccess();
