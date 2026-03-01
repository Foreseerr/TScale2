#pragma once

constexpr bool SIMULATE_MULTI_GPU = false;
// constexpr bool SIMULATE_MULTI_GPU = true;

constexpr int MAX_NUM_DEVICES = 8;

constexpr int BG_SM_COUNT = 16;

yint GetCudaDeviceCount();
void CudaSetDevice(yint deviceId);
yint CudaGetDevice();
yint GetCudaSMCount();
void CudaEnablePeerAccess();
