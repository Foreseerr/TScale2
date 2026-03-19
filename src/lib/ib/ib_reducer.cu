#include <util/pch.h>
#define KERNEL_UNIT "ib_reducer/"
#include "ib_reducer.h"
#include "rdma_tcp.h"
#include "rdma_ib.h"
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/fast_div.cuh>
#include <lib/cuda/vec_util.cuh>
#include <lib/net/ip_address.h>
#include <lib/net/p2p.h>
#include <lib/net/tcp_net.h>
#include <util/thread.h>


namespace NCuda
{

enum {
    RRCMD_SEND_LOCAL_DATA,
    RRCMD_SEND_SUMS,
};

constexpr int RDMA_BUF_SIZE = 1 << 24;
constexpr int RDMA_CMD_BUF_SIZE = 32;
constexpr int MAX_HOST_COUNT = 1024;
constexpr int SYSTEM_BUFFER_COUNT = 2;


//////////////////////////////////////////////////////////////////////////
struct TRdmaReduceCmd
{
    union {
        struct {
            ui64 CmdId;
            void *LocalBuf;
            void *LocalData;
            int RemoteSysBufId;
            int CmdType;
            int ElemCount;
            int RankSize;
        };
        char Buf[128]; // command occupies one cache line
    };
};


__device__ void CopyCmd(TRdmaReduceCmd *dst, TRdmaReduceCmd *src)
{
    int h = threadIdx.x;
    ((int *)dst)[h] = ((int *)src)[h];
    __syncwarp();
    __threadfence_system();
}


__device__ void PostRdma(TCuda1DPtr<TRdmaReduceCmd> cmdBuf, TCuda1DPtr<ui64> sendCmdId, void *localBuf, void *localData, int remoteSysBufId,
    int elemCount, int rankSize, int cmdType)
{
    __shared__ TRdmaReduceCmd cmd;
    int slot = 0;
    if (threadIdx.x == 0) {
        ui64 cmdId = sendCmdId[0];
        sendCmdId[0] = cmdId + 1;
        cmd.CmdId = cmdId;
        cmd.LocalBuf = localBuf;
        cmd.LocalData = localData;
        cmd.RemoteSysBufId = remoteSysBufId;
        cmd.CmdType = cmdType;
        cmd.ElemCount = elemCount;
        cmd.RankSize = rankSize;
        slot = cmdId & (RDMA_CMD_BUF_SIZE - 1);
    }
    slot = __shfl_sync(ALL_WARPS_MASK, slot, 0);
    CopyCmd(&cmdBuf[slot], &cmd);
}


__device__ ui64 LoadVolatile(ui64 &val)
{
    volatile ui64 *p = &val;
    return *p;
}


__global__ void RdmaReduce(int chainId, int xSize, TCudaSpan ySpan, TIntDivision bufStride, int myRank, TIntDivision rankCount, TCuda2DPtr<float> buf,
    TCuda1DPtr<TRdmaReduceCmd> cmdBuf, TCuda1DPtr<ui64> sendCmdId, TCuda1DPtr<NNet::TRdmaCompletionCounters> rcc,
    TCuda1DPtr<char> remoteDataBuf0, TCuda1DPtr<char> remoteDataBuf1)
{
    CUDA_ASSERT((xSize & (WARP_VEC_DIM - 1)) == 0);
    CUDA_ASSERT(bufStride <= RDMA_BUF_SIZE);
    int blockSize = max(1, bufStride.Div(RDMA_BUF_SIZE));
    int tmpBufId = 0;
    for (int blockOffset = ySpan.Beg; blockOffset < ySpan.Fin; blockOffset += blockSize) {
        int blockSizeY = min(blockSize, ySpan.Fin - blockOffset);
        int szBytes = blockSizeY * bufStride;
        int elemCount = szBytes / sizeof(float);
        int rankSize = rankCount.DivCeil(elemCount);
        rankSize = (rankSize + 31) & ~31; // round up to make warp loads 128 byte aligned
        void *localBuf = &buf[0][0];
        float *localData = &buf[blockOffset][0];
        float *remoteData = (float *)((tmpBufId == 0) ? &remoteDataBuf0[0] : &remoteDataBuf1[0]);

        if (threadIdx.y == 0) {
            // send local data
            PostRdma(cmdBuf, sendCmdId, localBuf, localData, chainId * 2 + tmpBufId, elemCount, rankSize, RRCMD_SEND_LOCAL_DATA);
            // wait receiving all data, waiting all sends allows to overwrite localData
            if (threadIdx.x == 0) {
                ui64 opId = sendCmdId[0] - 1;
                while (LoadVolatile(rcc[0].RecvCompleteId) < opId || LoadVolatile(rcc[0].SendCompleteId) < opId) {
                }
            }
        }
        __syncthreads();

        // inplace compute sum for myRank
        int myRankSize = min(rankSize, elemCount - myRank * rankSize);
        {
            // use float4 for most part
            int bulkSize = myRankSize & ~(WARP_VEC_DIM - 1);
            for (int offset = threadIdx.y * WARP_VEC_DIM; offset < bulkSize; offset += blockDim.y * WARP_VEC_DIM) {
                float4 sum = LoadWarpVec(localData + myRank * rankSize + offset);
                for (int rank = 0; rank < rankCount; ++rank) {
                    if (rank != myRank) {
                        sum = sum + LoadWarpVec(remoteData + rank * rankSize + offset);
                    }
                }
                StoreWarpVec(localData + myRank * rankSize + offset, sum);
            }
            // add tail
            for (int offset = bulkSize + threadIdx.x + threadIdx.y * WARP_SIZE; offset < myRankSize; offset += blockDim.y * WARP_SIZE) {
                float sum = localData[myRank * rankSize + offset];
                for (int rank = 0; rank < rankCount; ++rank) {
                    if (rank != myRank) {
                        sum += remoteData[rank * rankSize + offset];
                    }
                }
                localData[myRank * rankSize + offset] = sum;
            }
        }
        __threadfence_system();
        __syncthreads();

        if (threadIdx.y == 0) {
            // send sums
            PostRdma(cmdBuf, sendCmdId, localBuf, localData, -1, elemCount, rankSize, RRCMD_SEND_SUMS);
            // can skip waiting for summation completion on all ranks since we are using alternating remoteData buffers
        }
        tmpBufId ^= 1;
    }
    // wait all transfers
    if (threadIdx.y == 0) {
        if (threadIdx.x == 0) {
            ui64 opId = sendCmdId[0] - 1;
            while (LoadVolatile(rcc[0].RecvCompleteId) < opId || LoadVolatile(rcc[0].SendCompleteId) < opId) {
            }
        }
    }
    __syncthreads();
}
KERNEL_BLOCK_SIZE(RdmaReduce, WARP_SIZE, MAX_WARPS);
}
using namespace NCuda;


namespace NNet
{
//////////////////////////////////////////////////////////////////////////
class TNetReducer : public INetReducer
{
    struct TChain : public TThrRefBase
    {
        ui64 WaitCmdId = RDMA_CMD_BUF_SIZE;
        TCudaVector<char> RemoteDataBuf0;
        TCudaVector<char> RemoteDataBuf1;
        TCudaVector<TRdmaReduceCmd> RdmaCmdArr;
        TCudaVector<ui64> SendCmdId;
        TCudaVector<TRdmaCompletionCounters> RdmaCompletionCounters;

        TChain(TPtrArg<TCudaMemoryPool> asyncIOPool)
        {
            RemoteDataBuf0.AllocateCuda(RDMA_BUF_SIZE + MAX_HOST_COUNT * 128, asyncIOPool);
            RemoteDataBuf1.AllocateCuda(RDMA_BUF_SIZE + MAX_HOST_COUNT * 128, asyncIOPool);
            RdmaCmdArr.AllocateHost(RDMA_CMD_BUF_SIZE);
            RdmaCmdArr.ClearHostMem();
            TVector<ui64> sendCmdId(1);
            sendCmdId[0] = RDMA_CMD_BUF_SIZE;
            SendCmdId.Init(sendCmdId);
            RdmaCompletionCounters.AllocateCuda(1, asyncIOPool);
        }

        void ResetCompletionCounters()
        {
            RdmaCompletionCounters.ClearDeviceMem();
        }
    };

    struct TDeviceCtx : public TThrRefBase
    {
        TVector<TIntrusivePtr<TChain>> ChainArr;
        TIntrusivePtr<TCudaMemoryPool> AsyncIOPool;
        THashMap<void *, yint, TPtrHash> AllBuffers;

        TDeviceCtx(TPtrArg<TCudaMemoryPool> asyncIOPool, yint chainCount) : AsyncIOPool(asyncIOPool.Get())
        {
            ChainArr.resize(chainCount);
            for (yint k = 0; k < chainCount; ++k) {
                ChainArr[k] = new TChain(asyncIOPool);
            }
        }

        void RegisterBuffers(const TVector<TMemoryBlob> &bufArr)
        {
            for (yint k = 0; k < YSize(bufArr); ++k) {
                AllBuffers[bufArr[k].Ptr] = k;
            }
        }

        void ResetCompletionCounters()
        {
            for (yint k = 0; k < YSize(ChainArr); ++k) {
                ChainArr[k]->ResetCompletionCounters();
            }
        }
    };


private:
    int MyRank = 0;
    int RankCount = 0;
    TIntrusivePtr<IRdmaTransport> Transport;
    TVector<TIntrusivePtr<TDeviceCtx>> DeviceArr;
    THashMap<TIntrusivePtr<TMultiDevice2DArray<float>>, yint> AllBuffers;
    TThread Worker;
    volatile bool Exit = false;

    ~TNetReducer()
    {
        Exit = true;
        Worker.Join();
    }

    void PrepareConnections(TVector<char> *p) override { Transport->PrepareConnections(p); }

    void EstablishConnections(yint myRank, const TVector<TString> &peerList, TVector<TVector<char>> &handshakeArr) override
    {
        MyRank = myRank;
        RankCount = YSize(peerList);
        Y_VERIFY(RankCount <= MAX_HOST_COUNT);
        Transport->EstablishConnections(myRank, peerList, handshakeArr);
    }

    void InitDevice(yint deviceId, TPtrArg<TCudaMemoryPool> asyncIOPool) override
    {
        if (deviceId >= YSize(DeviceArr)) {
            DeviceArr.resize(deviceId + 1);
        }
        TIntrusivePtr<TDeviceCtx> pCtx = new TDeviceCtx(asyncIOPool, GetChainCount());
        DeviceArr[deviceId] = pCtx;
        Transport->InitDevice(deviceId);
    }

    void OnInitComplete() override
    {
        yint chainCount = GetChainCount();
        yint deviceCount = YSize(DeviceArr);
        Transport->ChainArr.resize(chainCount);
        for (yint chainId = 0; chainId < chainCount; ++chainId) {
            IRdmaTransport::TChain &chain = Transport->ChainArr[chainId];
            chain.CompleteIdArr.resize(deviceCount);
            for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                auto &devCtx = *DeviceArr[deviceId];
                devCtx.ResetCompletionCounters();
                chain.CompleteIdArr[deviceId] = devCtx.ChainArr[chainId]->RdmaCompletionCounters.GetDevicePtr().Data;
            }
            chain.SendCompletion.Init(deviceCount, MyRank, RankCount);
            chain.RecvCompletion.Init(deviceCount, MyRank, RankCount);
        }
        // register buffers
        {
            TVector<TVector<NCuda::TMemoryBlob>> allBufArr;
            TVector<TIntrusivePtr<NCuda::TCudaMemoryPool>> poolArr;
            allBufArr.resize(deviceCount);
            poolArr.resize(deviceCount);
            yint bufCount = YSize(AllBuffers) + SYSTEM_BUFFER_COUNT * chainCount;
            for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                allBufArr[deviceId].resize(bufCount);
                for (yint chainId = 0; chainId < chainCount; ++chainId) {
                    auto &chain = *DeviceArr[deviceId]->ChainArr[chainId];
                    allBufArr[deviceId][chainId * 2 + 0] = chain.RemoteDataBuf0.GetDeviceMem();
                    allBufArr[deviceId][chainId * 2 + 1] = chain.RemoteDataBuf1.GetDeviceMem();
                }
                poolArr[deviceId] = DeviceArr[deviceId]->AsyncIOPool;
            }
            for (auto it = AllBuffers.begin(); it != AllBuffers.end(); ++it) {
                yint id = it->second;
                Y_ASSERT(id >= SYSTEM_BUFFER_COUNT * chainCount && id < bufCount);
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    if (it->first->IsAllocated(deviceId)) {
                        auto &buf = it->first->GetData(deviceId);
                        Y_ASSERT(buf.GetMemPool().Get() == DeviceArr[deviceId]->AsyncIOPool.Get());
                        allBufArr[deviceId][id] = buf.GetDeviceMem();
                    }
                }
            }
            for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                DeviceArr[deviceId]->RegisterBuffers(allBufArr[deviceId]);
            }
            Transport->OnInitComplete(allBufArr, poolArr);
        }
        Worker.Create(this);
    }

    void RegisterBuffer(TPtrArg<TMultiDevice2DArray<float>> buf) override
    {
        auto it = AllBuffers.find(buf.Get());
        if (it == AllBuffers.end()) {
            yint chainCount = GetChainCount();
            yint id = YSize(AllBuffers) + SYSTEM_BUFFER_COUNT * chainCount;
            AllBuffers[buf.Get()] = id;
        }
    }

    void AllReduce(
        TPtrArg<NCuda::TGraph> c, yint chainId, yint deviceId, NCuda::TCudaSpan ySpan, TPtrArg<TMultiDevice2DArray<float>> p) override
    {
        Y_VERIFY(RankCount > 1);
        Y_ASSERT(AllBuffers.find(p.Get()) != AllBuffers.end());
        TCuda2DArray<float> &buf = p->GetData(deviceId);
        TDeviceCtx &ctx = *DeviceArr[deviceId];
        TChain &chain = *ctx.ChainArr[chainId];
        CudaCall(c, RdmaReduce)
            .Read((int)chainId, (int)buf.GetXSize(), ySpan, TIntDivision(buf.GetDeviceMem().GetStrideInBytes()))
            .Read((int)MyRank, TIntDivision(RankCount))
            .Write(&buf, &chain.RdmaCmdArr)
            .Write(&chain.SendCmdId, &chain.RdmaCompletionCounters)
            .Write(&chain.RemoteDataBuf0, &chain.RemoteDataBuf1);
    }

    yint GetChainCount() override { return YSize(Transport->ChainArr); }

    inline yint GetBoundedRankSize(yint elemCount, yint rank, yint rankSize)
    {
        yint res = Min<yint>(rankSize, elemCount - rank * rankSize);
        Y_VERIFY(res > 0);
        return res;
    }

public:
    void WorkerThread()
    {
        while (!Exit) {
            for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
                TDeviceCtx &ctx = *DeviceArr[deviceId];
                for (yint chainId = 0; chainId < YSize(ctx.ChainArr); ++chainId) {
                    TChain &chain = *ctx.ChainArr[chainId];
                    yint slot = (chain.WaitCmdId) & (RDMA_CMD_BUF_SIZE - 1);
                    TRdmaReduceCmd &cmd = chain.RdmaCmdArr.GetHostPtr()[slot];
                    if (cmd.CmdId != chain.WaitCmdId) {
                        continue;
                    }
                    ++chain.WaitCmdId;

                // perform cmd
                    auto it = ctx.AllBuffers.find(cmd.LocalBuf);
                    Y_VERIFY(it != ctx.AllBuffers.end());
                    yint localBufId = it->second;
                    yint elemSize = sizeof(float);
                    yint localOffset = ((char *)cmd.LocalData - (char *)cmd.LocalBuf) / elemSize;
                    for (int rank = 0; rank < RankCount; ++rank) {
                        if (rank != MyRank) {
                            int fromBufId = 0;
                            ui64 fromOffset = 0;
                            int toBufId = 0;
                            ui64 toOffset = 0;
                            yint sendSize = 0;
                            if (cmd.CmdType == RRCMD_SEND_LOCAL_DATA) {
                            // each rank collecting its fragment from all hosts
                                fromBufId = localBufId;
                                fromOffset = localOffset + rank * cmd.RankSize;
                                toBufId = cmd.RemoteSysBufId;
                                toOffset = MyRank * cmd.RankSize;
                                sendSize = GetBoundedRankSize(cmd.ElemCount, rank, cmd.RankSize);

                            } else if (cmd.CmdType == RRCMD_SEND_SUMS) {
                            // each rank sends sum of the fragment to all hosts
                                fromBufId = localBufId;
                                fromOffset = localOffset + MyRank * cmd.RankSize;
                                toBufId = localBufId;
                                toOffset = localOffset + MyRank * cmd.RankSize;
                                sendSize = GetBoundedRankSize(cmd.ElemCount, MyRank, cmd.RankSize);
                            } else {
                                Y_VERIFY(0);
                            }
                            Transport->RdmaWrite(chainId, rank, deviceId, cmd.CmdId, fromBufId, fromOffset * elemSize, toBufId,
                                toOffset * elemSize, sendSize * elemSize);
                        }
                    }
                }
            }
            Transport->ProcessIncoming();
        }
    }

    TNetReducer(TPtrArg<IRdmaTransport> tr) : Transport(tr) {}
};


TIntrusivePtr<INetReducer> CreateTcpReducer(TPtrArg<ITcpSendRecv> net, yint chainCount)
{
    TIntrusivePtr<IRdmaTransport> transport = CreateTcpRdmaTransfer(net, RDMA_BUF_SIZE, chainCount);
    return new TNetReducer(transport);
}

TIntrusivePtr<INetReducer> CreateInfinibandReducer(TPtrArg<ITcpSendRecv> net, yint chainCount)
{
    TIntrusivePtr<IRdmaTransport> transport = CreateInfinibandRdmaTransfer(net, chainCount);
    return new TNetReducer(transport);
}
}
