#pragma once
#include "base_cuda.cuh"
#include "base_host.h"
#include "delta_cuda.h"
#include "matrix_scale.h"
#include <lib/ib/ib_reducer.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_init.h>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/multi_device_buf.h>


namespace NCuda
{

constexpr yint AGPU_DEP_CHAIN_COUNT = 2; // default chain count, will take from IB reducer if present

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAddGradientKernelParams
{
    float DispDecay;
    float Beta1;
    float Weight0;
    float Weight1;
    float L2Reg;
    float StepMult;
};

__global__ void AddMatrixKernel(int xSize, int ySize, TCuda2DPtr<float> src, TCuda2DPtr<float> dst);
__global__ void CopyMatrixKernel(int xSize, int ySize, TCuda2DPtr<float> src, TCuda2DPtr<float> dst);
__global__ void MatrixRowSum2Kernel(int xSize, int ySize, TCuda2DPtr<float> matr, TCuda1DPtr<float> sum2arr);


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int ADD_TO_TARGET>
__global__ void UnpackDeltaKernel(
    int xSize, int ySize, TCuda2DPtr<i8> delta, TCuda2DPtr<float> tileScale, TCuda2DPtr<float> dstArr)
{
    for (int y = threadIdx.y + blockIdx.x * blockDim.y; y < ySize; y += blockDim.y * gridDim.x) {
        for (int x = 0, tile = 0; x < xSize; x += MM_TILE, ++tile) {
            float scale = tileScale[tile][y];
            float4 val = Scale(LoadWarpVec(delta[y] + x), scale);
            if (ADD_TO_TARGET) {
                val = val + LoadWarpVec(dstArr[y] + x);
            }
            StoreWarpVec(dstArr[y] + x, val);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void UpdateGlobalRowDisp(int xSize, int ySize, int rowDispSize, int rowDispStep,
    TCuda1DPtr<float> deltaSum2arr, TAddGradientKernelParams *pParams, TCuda1DPtr<float> sumWeightArr, TCuda1DPtr<float> rowDispArr,
    TCuda1DPtr<float> rowScaleArr);

__global__ void AddGradientKernel(int xSize, int ySize, TCuda2DPtr<float> delta, TAddGradientKernelParams *pParams,
    TCuda1DPtr<float> rowScale, TCuda1DPtr<float> sparsity, TCuda2DPtr<float> avgGrad, TCuda2DPtr<float> weights,
    TCuda1DPtr<float> weightsRowSum2arr);


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
__global__ void ConvertMatrixKernel(int xSize, int ySize, int yBufSize, int totalXSize, int totalYSize, TCuda2DPtr<float> matr, TCuda1DPtr<float> sum2arr,
    TCuda2DPtr<TMatrixFloat> dst, TCuda1DPtr<float> resDiscrScale)
{
    __shared__ float multShared;
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    {
        float sum2 = 0;
        for (int k = h + warpId * WARP_SIZE; k < totalYSize; k += blockDim.y * WARP_SIZE) {
            sum2 += sum2arr[k];
        }
        sum2 = BlockSum(sum2);
        if (h == 0 && warpId == 0) {
            float sko = sqrt(sum2 / (totalXSize * totalYSize));
            float discrScale = sko * MODEL_DISCR_SCALE;
            multShared = (sko == 0) ? 0 : (1 / discrScale);
            if (blockIdx.x == 0) {
                resDiscrScale[0] = sko;
            }
        }
    }
    __syncthreads();
    float mult = multShared;
    for (int y = warpId + blockIdx.x * blockDim.y; y < yBufSize; y += blockDim.y * gridDim.x) {
        if (y < ySize) {
            for (int x = 0; x < xSize; x += MM_TILE) {
                float4 vec = LoadWarpVec(matr[y] + x);
                StoreWarpVec(dst[y] + x, Scale(vec, mult));
            }

        } else {
            float4 zero = ZeroWarpVec();
            for (int x = 0; x < xSize; x += MM_TILE) {
                StoreWarpVec(dst[y] + x, zero);
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAgpuMatrixWindow
{
    ETensorParallelSplit TPS = TPS_COPY;
    yint XSize = 0;
    yint YSize = 0;
    yint LocalXSize = 0;
    yint LocalYSize = 0;
    yint XOffset = 0;
    yint YOffset = 0;
    TCudaVector<int> RowRename; // host_row = RowRename[device_row]

    TAgpuMatrixWindow(yint xSize, yint ySize, yint rowDispSize, yint expertCount, yint dgRank, yint dgSize, ETensorParallelSplit tps)
    {
        TPS = tps;
        XSize = xSize;
        YSize = ySize;
        TVector<int> rowRename;
        if (tps == TPS_COPY) {
            LocalXSize = xSize;
            LocalYSize = ySize;
            for (yint y = 0; y < ySize; ++y) {
                rowRename.push_back(y);
            }
        } else if (tps == TPS_ROW) {
            yint wy = RoundUp(ySize / dgSize, MM_TILE);
            Y_VERIFY(wy * dgSize == ySize);
            LocalXSize = xSize;
            LocalYSize = wy;
            YOffset = dgRank * LocalYSize;
            for (yint y = 0; y < wy; ++y) {
                rowRename.push_back(y + YOffset);
            }
        } else if (tps == TPS_COLUMN) {
            yint wx = RoundUp(xSize / dgSize, MM_TILE);
            Y_VERIFY(wx * dgSize == xSize);
            LocalXSize = wx;
            LocalYSize = ySize;
            XOffset = dgRank * wx;
            for (yint y = 0; y < ySize; ++y) {
                rowRename.push_back(y);
            }
        } else if (tps == TPS_ROW_MOE) {
            yint wy = RoundUp(ySize / dgSize / expertCount, MM_TILE);
            Y_VERIFY(wy * dgSize * expertCount == ySize);
            LocalXSize = xSize;
            LocalYSize = wy * expertCount;
            YOffset = dgRank * LocalYSize; // need rename, but we don't use (hopefully) per row disp for moe matrices
            for (yint e = 0; e < expertCount; ++e) {
                for (yint y = 0; y < wy; ++y) {
                    rowRename.push_back(e * wy * dgSize + dgRank * wy + y);
                }
            }
        } else {
            Y_VERIFY(0);
        }
        RowRename.Init(rowRename);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TAgpuDeviceInfo : public TThrRefBase
{
public:
    struct TDeviceCtx : public TThrRefBase
    {
        TCudaVector<TAddGradientKernelParams> CudaAGParams;
        TIntrusivePtr<TCudaMemoryPool> AsyncIOPool;
        TVector<TIntrusivePtr<TKernelOp>> ChainPrevOpArr;

        TDeviceCtx(yint chainCount) { ChainPrevOpArr.resize(chainCount); }
        void ResetDepChains()
        {
            for (yint k = 0; k < YSize(ChainPrevOpArr); ++k) {
                ChainPrevOpArr[k] = 0;
            }
        }
    };

private:
    struct TChain
    {
        TVector<TIntrusivePtr<TMultiDeviceReducer>> DGReducerArr;
        TVector<TIntrusivePtr<TMultiDeviceReducer>> InstanceReducerArr;
    };

private:
    yint TP = 0;
    yint PP = 0;
    yint InstanceDeviceCount = 0;
    yint InstanceCount = 0;
    TVector<TIntrusivePtr<TDeviceCtx>> DeviceArr;
    TIntrusivePtr<TMultiDeviceBufferFabric> MultiBufferFabric;
    TVector<TIntrusivePtr<TMultiDeviceBuffers>> MultiBufferArr;
    TVector<TChain> ChainArr;
    TIntrusivePtr<NNet::INetReducer> IbReducer;
    yint PrevChainId = 0;

public:
    TAgpuDeviceInfo(yint deviceCount, yint tp, yint pp, TPtrArg<NNet::INetReducer> ibReducer)
        : TP(tp), PP(pp), IbReducer(ibReducer)
    {
        InstanceDeviceCount = TP * PP;
        InstanceCount = deviceCount / InstanceDeviceCount;
        Y_VERIFY(deviceCount == InstanceCount * InstanceDeviceCount);
        yint chainCount = AGPU_DEP_CHAIN_COUNT;
        if (ibReducer.Get()) {
            chainCount = ibReducer->GetChainCount();
        }
        DeviceArr.resize(deviceCount);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            DeviceArr[deviceId] = new TDeviceCtx(chainCount);
        }
        yint deviceGroupCount = deviceCount / TP;
        MultiBufferFabric = new TMultiDeviceBufferFabric();
        // multi buffer
        MultiBufferArr.resize(deviceGroupCount);
        for (yint k = 0; k < deviceGroupCount; ++k) {
            TDeviceGroup dg(k * TP, (k + 1) * TP);
            MultiBufferArr[k] = new TMultiDeviceBuffers(MultiBufferFabric, TDeviceGroup(k * TP, (k + 1) * TP));
        }
        // chains
        ChainArr.resize(chainCount);
        for (yint chainId = 0; chainId < chainCount; ++chainId) {
            TChain &chain = ChainArr[chainId];
            chain.DGReducerArr.resize(deviceGroupCount);
            for (yint k = 0; k < deviceGroupCount; ++k) {
                TDeviceGroup dg(k * TP, (k + 1) * TP);
                chain.DGReducerArr[k] = new TMultiDeviceReducer(dg, RP_LOW_PRIORITY);
            }
            chain.InstanceReducerArr.resize(InstanceDeviceCount);
            for (yint k = 0; k < InstanceDeviceCount; ++k) {
                TVector<int> deviceArr;
                for (yint id = k; id < deviceCount; id += InstanceDeviceCount) {
                    deviceArr.push_back(id);
                }
                chain.InstanceReducerArr[k] = new TMultiDeviceReducer(TDeviceGroup(deviceArr), RP_LOW_PRIORITY);
            }
        }
    }

    void InitDevice(yint deviceId, TStream &stream, TPtrArg<TCudaMemoryAllocator> cudaMemAllocator)
    {
        TDeviceCtx &ctx = *DeviceArr[deviceId];
        ctx.CudaAGParams.AllocateCuda(1);
        ctx.AsyncIOPool = cudaMemAllocator->CreateNonsharedPool();
        ctx.AsyncIOPool->UseForAsyncIO(true);
        MultiBufferArr[deviceId / TP]->InitSync(deviceId);
        for (TChain &chain : ChainArr) {
            chain.DGReducerArr[deviceId / TP]->InitSync(deviceId);
            chain.InstanceReducerArr[deviceId % InstanceDeviceCount]->InitSync(deviceId);
        }
        if (IbReducer.Get()) {
            IbReducer->InitDevice(deviceId, ctx.AsyncIOPool);
        }
    }

    TDeviceCtx &GetDeviceCtx(yint deviceId) { return *DeviceArr[deviceId]; }
    TPtrArg<TMultiDeviceBufferFabric> GetFabric() { return MultiBufferFabric; }
    TPtrArg<TMultiDeviceBuffers> GetDeviceGroupMultiBuffers(yint deviceGroupId) { return MultiBufferArr[deviceGroupId]; }
    TPtrArg<TMultiDeviceReducer> GetDeviceGroupReducer(yint chainId, yint deviceId)
    {//
        return ChainArr[chainId].DGReducerArr[deviceId / TP];
    }
    TPtrArg<TMultiDeviceReducer> GetInstanceReducer(yint chainId, yint deviceId)
    {
        return ChainArr[chainId].InstanceReducerArr[deviceId % InstanceDeviceCount];
    }
    TPtrArg<NNet::INetReducer> GetIbReducer() { return IbReducer; }
    yint GetDeviceCount() const { return YSize(DeviceArr); }
    yint GetDeviceGroupCount() const { return YSize(MultiBufferArr); }
    yint GetInstanceCount() const { return InstanceCount; }
    yint GetDeviceGroupRank(yint deviceId) const { return deviceId % TP; }
    yint GetInstanceId(yint deviceId) const { return deviceId / InstanceDeviceCount; }
    yint GetTP() const { return TP; }

    yint AssignChainId()
    {
        if (++PrevChainId = YSize(ChainArr)) {
            PrevChainId = 0;
        }
        return PrevChainId;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAgpuAllCudaMatrices : public TThrRefBase
{
    TVector<TIntrusivePtr<ICopyModelParamsToHost>> MatrixArr;

    void CopyModelParamsToHost()
    {
        for (auto x : MatrixArr) {
            x->CopyModelParamsToHost();
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline void CopyDeviceToHost(T *pHostArr, const T &devArr)
{
    TMemoryBlob hostMem = pHostArr->GetHostMem();
    TMemoryBlob devMem = devArr.GetDeviceMem();
    yint sz = Min<yint>(hostMem.GetSize(), devMem.GetSize());
    cudaError_t err = cudaMemcpy(hostMem.Ptr, devMem.Ptr, sz, cudaMemcpyDeviceToHost);
    Y_VERIFY(err == cudaSuccess);
}

void CopyDeviceToHost(TCuda2DArray<float> *pHostArr, const TCuda2DArray<float> &devArr, const TAgpuMatrixWindow &win);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAgpuCudaVector
{
    TCudaVector<float> Cuda;
    TCudaVector<float> Host;

    void Allocate(yint sz)
    {
        Cuda.AllocateCuda(sz);
        Host.AllocateHost(sz);
    }
    void Set(TPtrArg<TGraph> c, const TVector<float> &arr)
    {
        PutHost(&Host, arr);
        c->KernelCopy(&Cuda, Host);
    }
    void Set(TPtrArg<TGraph> c, float val)
    {
        TVector<float> arr;
        arr.push_back(val);
        Set(c, arr);
    }
    TVector<float> GetVec()
    {
        CopyDeviceToHost(&Host, Cuda);
        TVector<float> arr;
        GetAllData(Host, &arr);
        return arr;
    }
    float GetValue()
    {
        TVector<float> arr = GetVec();
        return arr[0];
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
class TAgpuCudaModelMatrix : public ICudaModelMatrixBase<TMatrixFloat>
{
    using ICudaModelMatrixBase<TMatrixFloat>::DeviceId;

    yint ChainId = 0;
    TAgpuMatrixWindow Window;
    TIntrusivePtr<IModelMatrixBase<TMatrixFloat>> Matrix;
    TIntrusivePtr<TAgpuDeviceInfo> DeviceInfo;
    TCuda2DArray<float> Weights;
    TCuda2DArray<float> AvgGrad;
    TCuda2DArray<TMatrixFloat> FastDevice;
    TIntrusivePtr<TMultiDevice2DArray<float>> Grad;
    TIntrusivePtr<TMultiDeviceVector<float>> GradRowSum2;
    TAgpuCudaVector RowDisp;
    TAgpuCudaVector SumWeight;
    TAgpuCudaVector Sparsity;
    TCudaVector<float> RowScale;
    TIntrusivePtr<TMultiDeviceVector<float>> WeightsRowSum2;
    TCudaVector<float> MatrixScale;
    TCudaPackedDeltaMatrix PackedDelta;

private:
    NCuda::TCudaVectorFragment<float> GetYWindow(NCuda::TCudaVector<float> &vec)
    {
        return vec.MakeFragment(Window.YOffset, Window.LocalYSize);
    }

    NCuda::TCuda2DArrayFragment<float> GetWindow(NCuda::TCuda2DArray<float> &matr)
    {
        return matr.MakeFragment(Window.XOffset, Window.LocalXSize, 0, Window.YSize);
    }

    NCuda::TCudaSpan SplitYacrossInstances()
    {
        yint instanceCount = DeviceInfo->GetInstanceCount();
        yint id = DeviceInfo->GetInstanceId(DeviceId);
        yint w = DivCeil(Window.LocalYSize, instanceCount);
        return TCudaSpan(w * id, Min<yint>(Window.LocalYSize, w * (id + 1)));
    }

    TKernelOp &CallConvertKernel(TPtrArg<TGraph> c)
    {
        int totalXSize = Window.XSize;
        int totalYSize = Window.YSize;
        int winXSize = Window.LocalXSize;
        int winYSize = Window.LocalYSize;
        int yBufSize = FastDevice.GetYSize();
        return CudaCall(c, ConvertMatrixKernel<TMatrixFloat>)
            .Block(WARP_SIZE, MAX_WARPS)
            .FullGrid(c)
            .Read(winXSize, winYSize, yBufSize, totalXSize, totalYSize, Weights, WeightsRowSum2->GetData(DeviceId))
            .Write(&FastDevice, &MatrixScale);
    }

    void Convert(TPtrArg<TGraph> c)
    {
        // weight sum2 compute
        int winXSize = Window.LocalXSize;
        int winYSize = Window.LocalYSize;
        auto myWSum2 = GetYWindow(WeightsRowSum2->GetData(DeviceId));
        TIntrusivePtr<TMultiDeviceReducer> dgReducer = DeviceInfo->GetDeviceGroupReducer(0, DeviceId).Get();

        CudaCall(c, MatrixRowSum2Kernel).Block(WARP_SIZE, MAX_WARPS).FullGrid(c).Read(winXSize, winYSize, Weights).Write(&myWSum2);
        if (Window.TPS == TPS_ROW || Window.TPS == TPS_ROW_MOE) {
            dgReducer->AllGather(c, WeightsRowSum2, DeviceId);
        } else if (Window.TPS == TPS_COLUMN) {
            dgReducer->AllReduce(c, WeightsRowSum2, DeviceId);
        }
        // convert
        CallConvertKernel(c);
    }

public:
    TAgpuCudaModelMatrix(void *groupPtr, yint deviceId, TPtrArg<TAgpuDeviceInfo> deviceInfo,
        IModelMatrixBase<TMatrixFloat> *pMatrix, yint moeExpertCount, EModelMatrixMemory mmm, ETensorParallelSplit tps)
        : ICudaModelMatrixBase<TMatrixFloat>(deviceId), Matrix(pMatrix), DeviceInfo(deviceInfo),
          Window(pMatrix->GetXSize(), pMatrix->GetYSize(), pMatrix->GetRowDispSize(), moeExpertCount, deviceId % deviceInfo->GetTP(),
              deviceInfo->GetTP(), tps)
    {
        int tp = DeviceInfo->GetTP();
        yint ySize = Window.YSize;
        yint winXSize = Window.LocalXSize;
        yint winYSize = Window.LocalYSize;
        Y_ASSERT((winXSize % MM_TILE) == 0);
        yint roundYSize = RoundUp(ySize, MM_TILE);
        yint roundWinYSize = RoundUp(winYSize, MM_TILE);
        yint rowDispSize = Matrix->GetRowDispSize();
        auto &devCtx = DeviceInfo->GetDeviceCtx(DeviceId);
        TIntrusivePtr<TMultiDeviceBufferFabric> fabric = DeviceInfo->GetFabric().Get();
        ChainId = DeviceInfo->AssignChainId();
        Weights.AllocateCuda(winXSize, roundWinYSize);
        AvgGrad.AllocateCuda(winXSize, roundWinYSize);
        FastDevice.AllocateCuda(winXSize, roundWinYSize);
        Grad = fabric->Create2DArray<float>(Sprintf("%p-Grad", groupPtr));
        Grad->AllocateCuda(DeviceId, winXSize, winYSize, devCtx.AsyncIOPool); // used to accumulate gradient across accumulate steps
        if (DeviceInfo->GetIbReducer().Get()) {
            DeviceInfo->GetIbReducer()->RegisterBuffer(Grad);
        }
        GradRowSum2 = fabric->CreateVector<float>(Sprintf("%p-GradRowSum2", groupPtr));
        GradRowSum2->AllocateCuda(DeviceId, roundYSize, null_ptr_arg);
        RowDisp.Allocate(rowDispSize);
        SumWeight.Allocate(1);
        Sparsity.Allocate(1);
        RowScale.Allocate(ySize);
        WeightsRowSum2 = fabric->CreateVector<float>(Sprintf("%p-WeightsRowSum2", groupPtr));
        WeightsRowSum2->AllocateCuda(DeviceId, roundYSize, null_ptr_arg);
        MatrixScale.AllocateCuda(1);
        if (mmm == MM_MEM_HOST) {
            PackedDelta.AllocateCuda(winXSize, winYSize);
        }
    }

    yint GetLocalXSize() const override { return Window.LocalXSize; }
    yint GetLocalYSize() const override { return Window.LocalYSize; }

    void CopyToDevice(TPtrArg<TGraph> c) override
    {
        THostModelMatrix &matr = Matrix->GetHostMatrix();
        c->KernelCopy2D(&Weights, GetWindow(matr.GetCudaMatr()), Window.RowRename);
        c->KernelCopy2D(&AvgGrad, GetWindow(matr.GetCudaGrad()), Window.RowRename);
        RowDisp.Set(c, matr.GetRowDisp());
        SumWeight.Set(c, matr.GetSumWeight());
        Sparsity.Set(c, matr.GetSparsity());
        Convert(c);
    }

    void ApplyGradientImpl(TPtrArg<TGraph> c)
    {
        int totalXSize = Window.XSize;
        int totalYSize = Window.YSize;
        int winXSize = Window.LocalXSize;
        int winYSize = Window.LocalYSize;
        int yBufSize = FastDevice.GetYSize();
        auto &devCtx = DeviceInfo->GetDeviceCtx(DeviceId);
        TCudaPOD<TAddGradientKernelParams> agParams = devCtx.CudaAGParams.GetElement(0);
        TIntrusivePtr<TMultiDeviceReducer> dgReducer = DeviceInfo->GetDeviceGroupReducer(ChainId, DeviceId).Get();
        TIntrusivePtr<TMultiDeviceReducer> instanceReducer = DeviceInfo->GetInstanceReducer(ChainId, DeviceId).Get();

        instanceReducer->AllReduce(c, Grad, DeviceId);
        if (DeviceInfo->GetIbReducer().Get()) {
            TCudaSpan ySpan = SplitYacrossInstances();
            DeviceInfo->GetIbReducer()->AllReduce(c, ChainId, DeviceId, ySpan, Grad);
            instanceReducer->AllGatherYSplit(c, Grad, ySpan, DeviceId);
        }
        TCuda2DArray<float> &totalGrad = Grad->GetData(DeviceId);
        auto myGradRowSum2 = GetYWindow(GradRowSum2->GetData(DeviceId));
        CudaCall(c, MatrixRowSum2Kernel).Block(WARP_SIZE, MAX_WARPS).FullGrid(c).Read(winXSize, winYSize, totalGrad).Write(&myGradRowSum2);
        if (Window.TPS == TPS_ROW || Window.TPS == TPS_ROW_MOE) {
            dgReducer->AllGather(c, GradRowSum2, DeviceId);
        } else if (Window.TPS == TPS_COLUMN) {
            dgReducer->AllReduce(c, GradRowSum2, DeviceId);
        }

        int rowDispSize = Matrix->GetRowDispSize();
        int rowDispStep = Window.YSize / rowDispSize;
        CudaCall(c, UpdateGlobalRowDisp)
            .Block(WARP_SIZE, MAX_WARPS)
            .Read(totalXSize, totalYSize, rowDispSize, rowDispStep)
            .Read(GradRowSum2->GetData(DeviceId), agParams)
            .Write(&SumWeight.Cuda, &RowDisp.Cuda, &RowScale);

        auto myRowScale = GetYWindow(RowScale);
        auto myWSum2 = GetYWindow(WeightsRowSum2->GetData(DeviceId));
        CudaCall(c, AddGradientKernel)
            .Block(WARP_SIZE, MAX_WARPS)
            .FullGrid(c)
            .Read(winXSize, winYSize, totalGrad, agParams, myRowScale, Sparsity.Cuda)
            .Write(&AvgGrad, &Weights, &myWSum2);
        if (Window.TPS == TPS_ROW || Window.TPS == TPS_ROW_MOE) {
            dgReducer->AllGather(c, WeightsRowSum2, DeviceId);
        } else if (Window.TPS == TPS_COLUMN) {
            dgReducer->AllReduce(c, WeightsRowSum2, DeviceId);
        }

        // convert
        devCtx.ChainPrevOpArr[ChainId] = &CallConvertKernel(c);
    }

    void AddDelta(TPtrArg<TGraph> c, TCuda2DArray<float> &delta, EBackpropMode bm) override
    {
        int winXSize = Window.LocalXSize;
        int winYSize = Window.LocalYSize;
        TCuda2DArray<float> &grad = Grad->GetData(DeviceId);
        Y_ASSERT(winXSize <= delta.GetXSize() && winYSize <= delta.GetYSize());
        auto &devCtx = DeviceInfo->GetDeviceCtx(DeviceId);
        c->SetMemPool(devCtx.AsyncIOPool);
        if (bm & BM_GRAD_ADD) {
            CudaCall(c, AddMatrixKernel)
                .Block(WARP_SIZE, MAX_WARPS)
                .FullGrid(c)
                .Dep(PtrArg(devCtx.ChainPrevOpArr[ChainId]))
                .Read(winXSize, winYSize, delta)
                .Write(&grad);
        } else {
            CudaCall(c, CopyMatrixKernel)
                .Block(WARP_SIZE, MAX_WARPS)
                .FullGrid(c)
                .Dep(PtrArg(devCtx.ChainPrevOpArr[ChainId]))
                .Read(winXSize, winYSize, delta)
                .Write(&grad);
        }
        if (bm & BM_GRAD_APPLY) {
            ApplyGradientImpl(c);
        }
    }

    void AddDelta(TPtrArg<TGraph> c, TCudaPackedDeltaMatrix &delta, EBackpropMode bm) override
    {
        int winXSize = Window.LocalXSize;
        int winYSize = Window.LocalYSize;
        TCuda2DArray<float> &grad = Grad->GetData(DeviceId);
        Y_ASSERT(winXSize <= delta.Delta.GetXSize() && winYSize <= delta.Delta.GetYSize());
        auto &devCtx = DeviceInfo->GetDeviceCtx(DeviceId);
        c->SetMemPool(devCtx.AsyncIOPool);
        if (bm & BM_GRAD_ADD) {
            CudaCall(c, UnpackDeltaKernel<1>)
                .Block(WARP_SIZE, MAX_WARPS)
                .FullGrid(c)
                .Dep(PtrArg(devCtx.ChainPrevOpArr[ChainId]))
                .Read(winXSize, winYSize, delta.Delta, delta.TileScale)
                .Write(&grad);
        } else {
            CudaCall(c, UnpackDeltaKernel<0>)
                .Block(WARP_SIZE, MAX_WARPS)
                .FullGrid(c)
                .Dep(PtrArg(devCtx.ChainPrevOpArr[ChainId]))
                .Read(winXSize, winYSize, delta.Delta, delta.TileScale)
                .Write(&grad);
        }
        if (bm & BM_GRAD_APPLY) {
            ApplyGradientImpl(c);
        }
    }

    // host delta manipulation
    TCudaPackedDeltaMatrix &GetHostDelta() override
    {
        Y_ASSERT(!PackedDelta.IsEmpty());
        return PackedDelta;
    }

    void AddHostDelta(TPtrArg<TGraph> c, EBackpropMode bm) override
    {
        Y_ASSERT(!PackedDelta.IsEmpty());
        AddDelta(c, PackedDelta, bm);
    }

    void AllowDelayedUpdates(TPtrArg<TGraph> c) override
    {
        // no delayed updates are supported yet
    }

    // data
    TCuda2DArray<TMatrixFloat> &GetFast() override { return FastDevice; }
    TCudaPOD<float> GetScale() const override { return MatrixScale.GetElement(0); }

    // copy to host
    void CopyModelParamsToHost() override
    {
        if (DeviceInfo->GetInstanceId(DeviceId) == 0) {
            bool isFirstRank = DeviceInfo->GetDeviceGroupRank(DeviceId) == 0;
            if (Window.TPS != TPS_COPY || isFirstRank) {
                THostModelMatrix &matr = Matrix->GetHostMatrix();
                CopyDeviceToHost(&matr.GetCudaMatr(), Weights, Window);
                CopyDeviceToHost(&matr.GetCudaGrad(), AvgGrad, Window);
                if (isFirstRank) {
                    TVector<float> rowDisp = RowDisp.GetVec();
                    rowDisp.resize(matr.GetRowDispSize());
                    matr.SetRowDisp(rowDisp, SumWeight.GetValue());
                }
                matr.OnDataUpdate();
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
class TAgpuModelMatrix : public IModelMatrixBase<TMatrixFloat>
{
    TIntrusivePtr<TAgpuDeviceInfo> DeviceInfo;
    TIntrusivePtr<TAgpuAllCudaMatrices> AllMatrixArr;
    yint MoeExpertCount = 0;
    EModelMatrixMemory MMM = MM_MEM_DEVICE;
    ETensorParallelSplit TPS = TPS_COPY;

    TIntrusivePtr<NCuda::ICudaModelMatrixBase<TMatrixFloat>> CreateCudaMatrix(yint deviceId) override
    {
        TIntrusivePtr<TAgpuCudaModelMatrix<TMatrixFloat>> res;
        res = new TAgpuCudaModelMatrix<TMatrixFloat>(this, deviceId, DeviceInfo, this, MoeExpertCount, MMM, TPS);
        AllMatrixArr->MatrixArr.push_back(res);
        return res;
    }
    IModelMatrixHostCompute *GetHostCompute() override { return nullptr; }

public:
    TAgpuModelMatrix(TPtrArg<TAgpuDeviceInfo> deviceInfo,
        TIntrusivePtr<TAgpuAllCudaMatrices> allMatrixArr, yint moeExpertCount, EModelMatrixMemory mmm, ETensorParallelSplit tps)
        : DeviceInfo(deviceInfo), AllMatrixArr(allMatrixArr), MoeExpertCount(moeExpertCount), MMM(mmm), TPS(tps)
    {
    }
};
}
