#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "cuda_matmul.cuh"
#include "sm90_fp16.cuh"


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int Rot>
struct TMatMulRotation
{
};

template <class T>
struct TMatMulRotation<T, 0>
{
    static __device__ int GetXStep(int stride) { return sizeof(T); }
    static __device__ int GetYStep(int stride) { return stride; }
    static __device__ TRegTile<half> Frag(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTile(ht, x, y); }
};

template <class T>
struct TMatMulRotation<T, 1>
{
    static __device__ int GetXStep(int stride) { return stride; }
    static __device__ int GetYStep(int stride) { return sizeof(T); }
    static __device__ TRegTile<half> Frag(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTileTransposed(ht, x, y); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreData>
struct TFp16MatMulData
{
    union {
        struct {
            T4x4SMemHalfTile aFrag[2];
            T4x4SMemHalfTile bFrag[2];
        };
        TStoreData StoreData;
    };
};


template <class TStoreFunc, class TSumFloat>
struct TFp16MatMulCtx
{
    typedef TFp16MatMulData<typename TStoreFunc::TShmem<TStoreFunc>> TShmem;

    TMatMulWarpResult<TSumFloat> Res;
    int warpId;
    int aWarpBlk;
    int bWarpBlk;

    __device__ TFp16MatMulCtx()
    {
        warpId = threadIdx.y;
        aWarpBlk = (warpId & 1);
        bWarpBlk = (warpId >> 1);
    }

    __device__ void Clear() { Res.Clear(); }

    template <int TRANSPOSE_A, int TRANSPOSE_B, class TFloatA, class TFloatB>
    inline __device__ void ComputeMatMul(TShmem &data, TCuda2DPtr<TFloatA> aMatr, int ax, int ay, TCuda2DPtr<TFloatB> bMatr, int bx, int by, int kSize)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);

        typedef TMatMulRotation<TFloatA, TRANSPOSE_A> ARotate;
        typedef TMatMulRotation<TFloatB, TRANSPOSE_B> BRotate;

        int aStride = aMatr.GetStrideInBytes();
        int aXStep = ARotate::GetXStep(aStride);
        int aYStep = ARotate::GetYStep(aStride);

        int bStride = bMatr.GetStrideInBytes();
        int bXStep = BRotate::GetXStep(bStride);
        int bYStep = BRotate::GetYStep(bStride);

        ui8 *aPtr0 = aMatr.GetRawData() + ax * aXStep + ay * aYStep;
        ui8 *bPtr0 = bMatr.GetRawData() + bx * bXStep + by * bYStep;
        ui8 *aPtr1 = aPtr0 + TILE_GROUP_SIZE * aYStep;
        ui8 *bPtr1 = bPtr0 + TILE_GROUP_SIZE * bYStep;

        for (int k = 0; k < kSize; k += 64) {
            __syncthreads();
            Copy4x4Tile(&data.aFrag[0], warpId, TCuda2DPtr<TFloatA>(aPtr0, aStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            Copy4x4Tile(&data.aFrag[1], warpId, TCuda2DPtr<TFloatA>(aPtr1, aStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            Copy4x4Tile(&data.bFrag[0], warpId, TCuda2DPtr<TFloatB>(bPtr0, bStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            Copy4x4Tile(&data.bFrag[1], warpId, TCuda2DPtr<TFloatB>(bPtr1, bStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            __syncthreads();

            for (int k = 0; k < 4; ++k) {
                TRegTile<half> b[2];
                b[0] = BRotate::Frag(data.bFrag[bWarpBlk], k, 0);
                b[1] = BRotate::Frag(data.bFrag[bWarpBlk], k, 1);
                for (int ty = 0; ty < 4; ++ty) {
                    TRegTile<half> a;
                    a = ARotate::Frag(data.aFrag[aWarpBlk], k, ty);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&Res.Sum[ty][tx], a, b[tx]);
                    }
                }
                b[0] = BRotate::Frag(data.bFrag[bWarpBlk], k, 2);
                b[1] = BRotate::Frag(data.bFrag[bWarpBlk], k, 3);
                for (int ty = 0; ty < 4; ++ty) {
                    TRegTile<half> a;
                    a = ARotate::Frag(data.aFrag[aWarpBlk], k, ty);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&Res.Sum[ty][tx + 2], a, b[tx]);
                    }
                }
            }
            aPtr0 += aXStep * TILE_GROUP_SIZE;
            aPtr1 += aXStep * TILE_GROUP_SIZE;
            bPtr0 += bXStep * TILE_GROUP_SIZE;
            bPtr1 += bXStep * TILE_GROUP_SIZE;
        }
    }

    template <class TRes>
    inline __device__ void SaveResult(
        const TTileCoord &tc, TShmem &data, TCuda2DPtr<TRes> resBuf, int mmTileX, int mmTileY, typename TStoreFunc::TParams params)
    {
        typedef TStoreMatMulWarpResult<TStoreFunc> TStoreResult;
        int offsetX = bWarpBlk * TILE_GROUP_SIZE;
        int offsetY = aWarpBlk * TILE_GROUP_SIZE;
        TStoreResult::Store(params, data.StoreData, tc, Res, 1.0f, offsetX, offsetY, mmTileX, mmTileY, resBuf);
    }
};


template <class TStoreFunc, class TRes, class TMSize, class TNSize, class TMultiArgs>
TKernelOp &Fp16MatMulMulti(TPtrArg<TGraph> c, TRes *pResMatr, TMSize &&mSize, TNSize &&nSize, const TMultiArgs &multiArgs)
{
    if (UseSm90Kernels() && multiArgs.TypeMatch((half*)0)) {
        typedef NSm90Fp16MatMul::TPassMatMulKernelArguments TAdapter;
        typedef typename TMakeKernelParam<TMultiArgs>::Result<TAdapter> TKernelParams;
        using NSm90Fp16MatMul::Sm90Fp16MatMulKernel; // otherwise KERNEL_BLOCK_SIZE() is not picked up
        TAdapter ad;
        TKernelOp &res = CudaCall(c, Sm90Fp16MatMulKernel<TStoreFunc, typename TRes::TElem, TKernelParams>);
        res.FullGrid(c);
        res.Shmem(sizeof(NSm90Fp16MatMul::TMatMulData<typename TStoreFunc::TShmem<TStoreFunc>>));
        res.Read(mSize, nSize).Write(pResMatr).Struct();
        PassKernelParams(ad, res, multiArgs);
        return res;
    } else {
        typedef TPassMatMulKernelArguments TAdapter;
        typedef typename TMakeKernelParam<TMultiArgs>::Result<TAdapter> TKernelParams;
        typedef float TSumFloat;
        typedef TFp16MatMulCtx<TStoreFunc, TSumFloat> TMatMulCtx;
        TAdapter ad;
        TKernelOp &res = CudaCall(c, GenericMatMulKernel<TStoreFunc, typename TRes::TElem, TKernelParams, TMatMulCtx>);
        res.Grid(MMTiles(nSize), MMTiles(mSize));
        res.Read(mSize, nSize);
        res.Write(pResMatr).Struct();
        PassKernelParams(ad, res, multiArgs);
        return res;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int TRANSPOSE_A, int TRANSPOSE_B, class TStoreFunc, class T1, class T2, class TRes, class TMSize, class TNSize, class TKSize>
TKernelOp &Fp16MatMul(TPtrArg<TGraph> c, const T1 &aMatr, const T2 &bMatr, TRes *pResMatr, TMSize &&mSize, TNSize &&nSize, TKSize &&kSize)
{
    auto mm = MakeMatMulArgs<TRANSPOSE_A, TRANSPOSE_B>(nullptr, aMatr, bMatr, kSize);
    return Fp16MatMulMulti<TStoreFunc>(c, pResMatr, mSize, nSize, MultiArg() + mm);
}
}
