#pragma once
#include "acpu_host.h"
#include "base_cuda.cuh"
#include "matrix_scale.h"
#include "delta_cuda.h"
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/multi_device_buf.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// kernels
const int COPY_DELTA_WARPS = 32;
__global__ void CopyDelta(TCuda2DPtr<float> srcArr, int xSize, int ySize, TCuda1DPtr<int> iterCounter, TCuda2DPtr<float> tileScaleBuf,
    TCuda2DPtr<i8> dstArr, TCuda2DPtr<float> dstTileScale, int *launchFlag);
__global__ void CopyPackedDelta(TCuda2DPtr<i8> srcArr, TCuda2DPtr<float> srcTileScale, int xSize, int ySize, TCuda1DPtr<int> iterCounter,
    TCuda2DPtr<i8> dstArr, TCuda2DPtr<float> dstTileScale, int *launchFlag);
__global__ void LaunchOpKernel(TCuda1DPtr<int> iterCounter, int *launchOpPtr);
__global__ void AssignIterCounterKernel(int *hostIterCounter, TCuda1DPtr<int> iterCounter);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAcpuDeviceInfo : public TThrRefBase
{
    struct TDeviceCtx : public TThrRefBase
    {
        TIntrusivePtr<TCudaModelMatrixScale> CudaMatrixScale;
        TCudaVector<int> IterCounter;
    };

    TVector<TIntrusivePtr<TDeviceCtx>> DeviceArr;
    TIntrusivePtr<THostModelMatrixScale> HostMatrixScale;
    TIntrusivePtr<TMultiDeviceBufferFabric> MultiBufferFabric;
    TVector<TIntrusivePtr<TMultiDeviceBuffers>> MultiBufferArr;

public:
    TAcpuDeviceInfo(yint deviceCount, yint maxMatrixCount)
    {
        HostMatrixScale = new THostModelMatrixScale(maxMatrixCount);
        DeviceArr.resize(deviceCount);
        MultiBufferArr.resize(deviceCount);
        MultiBufferFabric = new TMultiDeviceBufferFabric();
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            DeviceArr[deviceId] = new TDeviceCtx;
            MultiBufferArr[deviceId] = new TMultiDeviceBuffers(MultiBufferFabric, TDeviceGroup(deviceId, deviceId + 1));
        }
    }
    void InitDevice(yint deviceId, TStream &stream)
    {
        TDeviceCtx &ctx = *DeviceArr[deviceId];
        ctx.CudaMatrixScale = new TCudaModelMatrixScale(HostMatrixScale);
        ctx.IterCounter.AllocateCuda(1);
        ctx.IterCounter.ClearDeviceMem(stream);
        MultiBufferArr[deviceId]->InitSync(deviceId);
    }
    void AssignIterCounter(yint deviceId, TPtrArg<TGraph> c, TCudaPOD<int> hostIterCounter)
    {
        CudaCall(c, AssignIterCounterKernel).Read(hostIterCounter).Write(&DeviceArr[deviceId]->IterCounter);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
class TAcpuCudaModelMatrix : public ICudaModelMatrixBase<TMatrixFloat>
{
    using ICudaModelMatrixBase<TMatrixFloat>::DeviceId;
   
    EModelMatrixMemory Mem = MM_MEM_DEVICE;
    TIntrusivePtr<TAcpuHostModelMatrix<TMatrixFloat>> Matrix;
    TIntrusivePtr<TAcpuDeviceInfo> DeviceInfo;
    TCuda2DArray<TMatrixFloat> FastDevice;
    TCuda2DArray<float> TileScaleBuf;

    TAcpuDeviceInfo::TDeviceCtx &GetCtx() const { return *DeviceInfo->DeviceArr[DeviceId]; }

public:
    TAcpuCudaModelMatrix(
        yint deviceId, TPtrArg<TAcpuDeviceInfo> deviceInfo, TAcpuHostModelMatrix<TMatrixFloat> *pMatrix, EModelMatrixMemory mmm)
        : ICudaModelMatrixBase<TMatrixFloat>(deviceId), Matrix(pMatrix), DeviceInfo(deviceInfo), Mem(mmm)
    {
        if (Mem == MM_MEM_DEVICE) {
            yint xSize = pMatrix->GetXSize();
            yint ySize = pMatrix->GetYSize();
            Y_ASSERT((xSize % MM_TILE) == 0);
            yint roundYSize = RoundUp(ySize, MM_TILE);
            FastDevice.AllocateCuda(xSize, roundYSize);
            TileScaleBuf.AllocateCuda(ySize, xSize / MODEL_INT8_DELTA_TILE);
        }
    }

    yint GetLocalXSize() const override { return Matrix->GetXSize(); }
    yint GetLocalYSize() const override { return Matrix->GetYSize(); }

    void CopyToDevice(TPtrArg<TGraph> c) override
    {
        if (Mem == MM_MEM_HOST) {
            return;
        }
        TCuda2DArray<TMatrixFloat> &fastHost = Matrix->GetFastHost();
        // copy over PCIE bypassing CPU completely
        // one active pcie transfer to device at a time
        c->KernelCopy(&FastDevice, fastHost, fastHost.GetYSize())//
            .Chain(CHAIN_OP_PCI_TO_DEVICE, c);
    }

    void AddDelta(TPtrArg<TGraph> c, TCuda2DArray<float> &delta, EBackpropMode bm) override
    {
        (void)bm;
        // copy first rows, delta might have more rows due to size rounding
        TCudaPackedDeltaMatrix &hostDelta = GetHostDelta();
        int xSize = Matrix->GetXSize();
        int ySize = Matrix->GetYSize();
        TCudaPOD<int> flag = Matrix->GetLaunchFlag(DeviceId);
        CudaCall(c, CopyDelta)
            .Block(WARP_SIZE, COPY_DELTA_WARPS)
            .Read(delta, xSize, ySize, GetCtx().IterCounter)
            .Write(&TileScaleBuf, &hostDelta.Delta, &hostDelta.TileScale, &flag)
            .Chain(CHAIN_OP_PCI_TO_HOST, c);
    }

    void AddDelta(TPtrArg<TGraph> c, TCudaPackedDeltaMatrix &delta, EBackpropMode bm) override
    {
        (void)bm;
        // copy first rows, delta might have more rows due to size rounding
        TCudaPackedDeltaMatrix &hostDelta = GetHostDelta();
        int xSize = Matrix->GetXSize();
        int ySize = Matrix->GetYSize();
        TCudaPOD<int> flag = Matrix->GetLaunchFlag(DeviceId);
        CudaCall(c, CopyPackedDelta)
            .Block(WARP_SIZE, COPY_DELTA_WARPS)
            .Read(delta.Delta, delta.TileScale, xSize, ySize, GetCtx().IterCounter)
            .Write(&hostDelta.Delta, &hostDelta.TileScale, &flag)
            .Chain(CHAIN_OP_PCI_TO_HOST, c);
    }

    // host delta manipulation
    TCudaPackedDeltaMatrix &GetHostDelta() override { return Matrix->GetHostDelta(DeviceId); }

    void AddHostDelta(TPtrArg<TGraph> c, EBackpropMode bm) override
    {
        (void)bm;
        TCudaPackedDeltaMatrix &hostDelta = GetHostDelta();
        TCudaPOD<int> flag = Matrix->GetLaunchFlag(DeviceId);
        // add dependency, should wait all host delta writes
        CudaCall(c, LaunchOpKernel).DepRead(hostDelta.Delta, hostDelta.TileScale).Read(GetCtx().IterCounter).Write(&flag);
    }

    void AllowDelayedUpdates(TPtrArg<TGraph> c) override
    {
        TCuda2DArray<TMatrixFloat> &data = GetFast();
        TCudaPOD<float> scale = GetScale();
        TCudaPOD<int> flag = Matrix->GetAllowDelayedFlag(DeviceId);
        // add dependency, should wait all matrix reads
        CudaCall(c, LaunchOpKernel).DepWrite(data, scale).Read(GetCtx().IterCounter).Write(&flag);
    }

    // data
    TCuda2DArray<TMatrixFloat> &GetFast() override
    {
        if (Mem == MM_MEM_HOST) {
            return Matrix->GetFastHost();
        } else {
            return FastDevice;
        }
    }
    TCudaPOD<float> GetScale() const override { return GetCtx().CudaMatrixScale->GetElement(Matrix->GetMatrixScaleIndex()); }
    void CopyModelParamsToHost() override {}
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
class TAcpuModelMatrix : public TAcpuHostModelMatrix<TMatrixFloat>
{
    typedef TAcpuHostModelMatrix<TMatrixFloat> TParent;
    TIntrusivePtr<TAcpuDeviceInfo> DeviceInfo;

    TIntrusivePtr<NCuda::ICudaModelMatrixBase<TMatrixFloat>> CreateCudaMatrix(yint deviceId) override
    {
        return new TAcpuCudaModelMatrix<TMatrixFloat>(deviceId, DeviceInfo, this, TParent::GetMemoryType());
    }

public:
    TAcpuModelMatrix(TPtrArg<TAcpuDeviceInfo> deviceInfo) : DeviceInfo(deviceInfo) {}
};
}
