#pragma once
#include "sm90.cuh"
#include "cuda_matmul_iter.cuh"

namespace NCuda
{
namespace NSm90Fp16MatMul
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
            T4x4SMemHalfTile aFrag[QSIZE][M_GROUP_COUNT];
            T4x4SMemHalfTile bFrag[QSIZE][2 * WG_COUNT];
        };
    };
    TStoreData StoreData;
    ui64 BarLoad[QSIZE], BarCompute[QSIZE];
    char Padding[128];
};


struct TResultComputer
{
    TMatMulWarpGroupResult<float> Buf;

    template <int TRANSPOSE_A, int TRANSPOSE_B, class TStoreData>
    inline __device__ void Compute(TMatMulData<TStoreData> *pData, int wgId, int q)
    {
        half *a0 = (half *)&pData->aFrag[q][0];
        half *a1 = (half *)&pData->aFrag[q][1];
        half *b0 = (half *)&pData->bFrag[q][0 + wgId * 2];
        half *b1 = (half *)&pData->bFrag[q][1 + wgId * 2];
        int aNext = TRANSPOSE_A ? 16 * 64 : 16;
        int bNext = TRANSPOSE_B ? 16 * 64 : 16;
        warpgroup_arrive();
        for (int k = 0; k < 4; ++k) {
            wgmma64<1, TRANSPOSE_A, TRANSPOSE_B>(Buf.Sum00, a0 + aNext * k, b0 + bNext * k);
            wgmma64<1, TRANSPOSE_A, TRANSPOSE_B>(Buf.Sum01, a0 + aNext * k, b1 + bNext * k);
            wgmma64<1, TRANSPOSE_A, TRANSPOSE_B>(Buf.Sum10, a1 + aNext * k, b0 + bNext * k);
            wgmma64<1, TRANSPOSE_A, TRANSPOSE_B>(Buf.Sum11, a1 + aNext * k, b1 + bNext * k);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }
};


template <int TRANSPOSE>
struct TSliceLoad
{
    int Row, Column;

    __device__ void Start(int row, int column)
    {
        Row = row;
        Column = column;
    }
    __device__ void LoadData(T4x4SMemHalfTile *p, const CUtensorMap &tensorMap, ui64 *bar, int rowCount, int q)
    {
        for (int r = 0; r < rowCount; ++r) {
            if (TRANSPOSE) {
                tma_copy(&p[r], tensorMap, bar, Row + r * 64, Column);
            } else {
                tma_copy(&p[r], tensorMap, bar, Column, Row + r * 64);
            }
        }
        Column += 64;
    }
};


template <int TRANSPOSE_A, int TRANSPOSE_B>
struct TDataLoader
{
    TSliceLoad<TRANSPOSE_A> LoadA;
    TSliceLoad<TRANSPOSE_B> LoadB;
    const CUtensorMap &TensorMapA;
    const CUtensorMap &TensorMapB;

    __device__ TDataLoader(const CUtensorMap &tensorMapA, const CUtensorMap &tensorMapB)
        : TensorMapA(tensorMapA), TensorMapB(tensorMapB)
    {
    }
    __device__ void Start(int row, int column, int ka, int kb)
    {
        LoadA.Start(row, ka);
        LoadB.Start(column, kb);
    }
    template <class TStoreData>
    __device__ void LoadData(TMatMulData<TStoreData> *pData, ui64 *barArr, int q)
    {
        ui64 *bar = &barArr[q];
        mbarrier_expect_bytes(bar, (M_GROUP_COUNT + 2 * WG_COUNT) * sizeof(T4x4SMemHalfTile));
        LoadA.LoadData(pData->aFrag[q], TensorMapA, bar, M_GROUP_COUNT, q);
        LoadB.LoadData(pData->bFrag[q], TensorMapB, bar, 2 * WG_COUNT, q);
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
        op.Read(GetTensorMap<64, 64>(matr)).DepRead(matr);
    }
    template <class T>
    void PassMatrixB(TKernelOp &op, T &matr)
    {
        op.Read(GetTensorMap<64, 64>(matr)).DepRead(matr);
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
        TDataLoader<TRANSPOSE_A, TRANSPOSE_B> load(aMatr, bMatr);
        load.Start(ay, by, ax, bx); // make normal order order
        for (int kPtr = 0; kPtr < kSize; kPtr += 64) {
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
        for (int kPtr = 0; kPtr < kSize; kPtr += 64) {
            int dataQ = QLoad.Wait(data.BarLoad);
            Sum.Compute<TRANSPOSE_A, TRANSPOSE_B>(&data, wgId, dataQ);
            QCompute.Arrive(data.BarCompute);
        }
    }
};


template <class TStoreFunc, class TRes, class TMultiArgs>
__global__ __launch_bounds__(WARP_SIZE * (4 * WG_COUNT + 4), 1) //
    void Sm90Fp16MatMulKernel(
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
KERNEL_BLOCK_SIZE(Sm90Fp16MatMulKernel, WARP_SIZE, 4 * WG_COUNT + 4);
}
}
