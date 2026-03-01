#pragma once
#include "sm90.cuh"
#include "cuda_matmul_iter.cuh"


namespace NCuda
{
namespace NSm90Fp8MatMul
{
constexpr int WG_COUNT = 2;
constexpr int M_GROUP_COUNT = 2;
constexpr int QSIZE = 3;

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreData>
struct TMatMulData
{
    union {
        struct
        {
            i8 aFrag[QSIZE][M_GROUP_COUNT][64 * 128];
            i8 bFrag[QSIZE][WG_COUNT][128 * 128];
        };
    };
    TStoreData StoreData;
    ui64 BarLoad[QSIZE], BarCompute[QSIZE];
    char Padding[128];
};


struct TResultComputer
{
    TMatMulWarpGroup128Result<float> Buf;
    TRegTile<float> TmpSum[8];

    __device__ void ComputeRow(i8 *a, i8 *b)
    {
        warpgroup_arrive();
        wgmma128e4e4<0>(TmpSum, a, b);
        for (int k = 1; k < 4; ++k) {
            wgmma128e4e4<1>(TmpSum, a + 32 * k, b + 32 * k);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    __device__ void AddRow(TRegTile<float> *sum)
    {
        for (int k = 0; k < ARRAY_SIZE(TmpSum); ++k) {
            sum[k].Add(TmpSum[k]);
        }
    }

    template <class TStoreData>
    __device__ void Compute(TMatMulData<TStoreData> *pData, int wgId, int q)
    {
        i8 *a0 = pData->aFrag[q][0];
        i8 *a1 = pData->aFrag[q][1];
        i8 *b = pData->bFrag[q][wgId];
        ComputeRow(a0, b);
        AddRow(Buf.Sum0);
        ComputeRow(a1, b);
        AddRow(Buf.Sum1);
        // precision loss
        // warpgroup_arrive(); 
        // for (int k = 0; k < 4; ++k) {
        //     wgmma128e4e4<1>(Buf.Sum0, a0 + 32 * k, b + 32 * k);
        //     wgmma128e4e4<1>(Buf.Sum1, a1 + 32 * k, b + 32 * k);
        // }
        // warpgroup_commit_batch();
        // warpgroup_wait<0>();
    }
};


struct TDataLoader
{
    int Row;
    int Column;
    int Ka, Kb;
    const CUtensorMap &TensorMapA;
    const CUtensorMap &TensorMapB;

    __device__ TDataLoader(const CUtensorMap &tensorMapA, const CUtensorMap &tensorMapB) : TensorMapA(tensorMapA), TensorMapB(tensorMapB) {}
    __device__ void Start(int row, int column, int ka, int kb)
    {
        Row = row;
        Column = column;
        Ka = ka;
        Kb = kb;
    }
    template <class TStoreData>
    __device__ void LoadData(TMatMulData<TStoreData> *pData, ui64 *barArr, int q)
    {
        ui64 *bar = &barArr[q];
        mbarrier_expect_bytes(bar, M_GROUP_COUNT * (64 * 128) + WG_COUNT * (128 * 128));
        for (int y = 0; y < M_GROUP_COUNT; ++y) {
            tma_copy(pData->aFrag[q][y], TensorMapA, bar, Ka, Row + y * 64);
        }
        for (int y = 0; y < WG_COUNT; ++y) {
            tma_copy(pData->bFrag[q][y], TensorMapB, bar, Kb, Column + y * 128);
        }
        Ka += 128;
        Kb += 128;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TPassMatMulKernelArguments
{
    template <class T>
    using TMatrixArg = CUtensorMap;

    template <class T>
    void PassMatrixA(TKernelOp &op, T &matr)
    {
        op.Read(GetTensorMap<64, 128>(matr)).DepRead(matr);
    }
    template <class T>
    void PassMatrixB(TKernelOp &op, T &matr)
    {
        op.Read(GetTensorMap<128, 128>(matr)).DepRead(matr);
    }
};


struct TNextGenLoadData
{
    int Lag = QSIZE;
    TQueueIndex<QSIZE> QLoad, QCompute;

    template <int TRANSPOSE_A, int TRANSPOSE_B, class TShmem>
    inline __device__ void ComputeMatMul(
        TShmem &data, const CUtensorMap &aMatr, int ax, int ay, const CUtensorMap &bMatr, int bx, int by, int kSize)
    {
        TDataLoader load(aMatr, bMatr);
        load.Start(ay, by, ax, bx); // make normal order order
        for (int kPtr = 0; kPtr < kSize; kPtr += 128) {
            if (Lag > 0) {
                --Lag;
            } else {
                QCompute.Wait(data.BarCompute);
            }
            load.LoadData(&data, data.BarLoad, QLoad.Allocate());
        }
    }
};


struct TNextGenComputer
{
    TResultComputer Sum;
    TQueueIndex<QSIZE> QLoad, QCompute;

    template <int TRANSPOSE_A, int TRANSPOSE_B, class TShmem>
    inline __device__ void ComputeMatMul(
        TShmem &data, const CUtensorMap &aMatr, int ax, int ay, const CUtensorMap &bMatr, int bx, int by, int kSize)
    {
        int wgId = threadIdx.y / 4;
#pragma unroll(2)
        for (int kPtr = 0; kPtr < kSize; kPtr += 128) {
            int dataQ = QLoad.Wait(data.BarLoad);
            Sum.Compute(&data, wgId, dataQ);
            QCompute.Arrive(data.BarCompute);
        }
    }
};


template <class TStoreFunc, class TRes, class TMultiArgs>
__global__ __launch_bounds__(WARP_SIZE * (4 * WG_COUNT + 4), 1) //
    void Sm90Fp8MatMulKernel(
        int mSize, int nSize, TCuda2DPtr<TRes> resBuf, __grid_constant__ const TMultiArgs multiArgs, typename TStoreFunc::TParams params)
{
    extern __shared__ char shmem[];
    typedef TMatMulData<typename TStoreFunc::TShmem<TStoreFunc>> TShmem;
    TShmem &data = *(TShmem *)AlignDevicePtr(shmem, 128);

    if (threadIdx.x < QSIZE && threadIdx.y == 0) {
        int k = threadIdx.x;
        mbarrier_init(&data.BarLoad[k], 1);
        mbarrier_init(&data.BarCompute[k], 128 * WG_COUNT);
    }
    __syncthreads();

    TMatMulTileIterator<MM_TILE * WG_COUNT, MM_TILE, TStoreFunc> mmIter(mSize, nSize);
    if (threadIdx.y >= 4 * WG_COUNT) {
        // load warp
        warp_reg_free<24>();
        if (threadIdx.y > 4 * WG_COUNT || threadIdx.x != 0) {
            return;
        }

        TNextGenLoadData loadData;
        TCallComputeMatMul<TNextGenLoadData, TShmem> cc(loadData, data);
        while (mmIter.Next()) {
            multiArgs.EnumArgs(cc, mmIter.GetX(), mmIter.GetY());
        }

    } else {
        // compute warps
        warp_reg_alloc<240>();

        TNextGenComputer computer;
        TCallComputeMatMul<TNextGenComputer, TShmem> cc(computer, data);
        while (mmIter.Next()) {
            computer.Sum.Buf.Clear();
            multiArgs.EnumArgs(cc, mmIter.GetX(), mmIter.GetY());

            int wgId = (threadIdx.y / 4);
            int startResRow = blockIdx.z * mSize;
            mmIter.Store(wgId, WG_COUNT, computer.Sum.Buf, params, data.StoreData, resBuf, startResRow);
        }
    }
}
KERNEL_BLOCK_SIZE(Sm90Fp8MatMulKernel, WARP_SIZE, 4 * WG_COUNT + 4);
}
}
