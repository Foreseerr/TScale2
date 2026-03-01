#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"


namespace NCuda
{
struct TSortNode
{
    int NodeId;
    float Score;
};

constexpr int FLOAT_SORT_WARP_COUNT = 32;


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void SortFloatsKernel(TCuda1DPtr<float> valArr, int nodeCount, TCuda1DPtr<TSortNode> nodes);
KERNEL_BLOCK_SIZE(SortFloatsKernel, WARP_SIZE, FLOAT_SORT_WARP_COUNT);

template <class TXSize>
void SortFloats(TPtrArg<TGraph> c, TCudaVector<float> &valArr, TXSize &&len, TCudaVector<TSortNode> *pDst)
{
    CudaCall(c, SortFloatsKernel).Read(valArr, len).Write(pDst);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void HistSortFloatsKernel(TCuda1DPtr<float> valArr, int nodeCount, TCuda1DPtr<TSortNode> nodes);
KERNEL_BLOCK_SIZE(HistSortFloatsKernel, WARP_SIZE, FLOAT_SORT_WARP_COUNT);

template <class TXSize>
void SortPositiveFloatsApproxFast(TPtrArg<TGraph> c, TCudaVector<float> &valArr, TXSize &&len, TCudaVector<TSortNode> *pDst)
{
    CudaCall(c, HistSortFloatsKernel).Read(valArr, len).Write(pDst);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void HistSortFloatsStableKernel(TCuda1DPtr<float> valArr, int nodeCount, TCuda1DPtr<TSortNode> nodes);
KERNEL_BLOCK_SIZE(HistSortFloatsStableKernel, WARP_SIZE, FLOAT_SORT_WARP_COUNT);

template <class TXSize>
void SortPositiveFloatsApproxStable(TPtrArg<TGraph> c, TCudaVector<float> &valArr, TXSize &&len, TCudaVector<TSortNode> *pDst)
{
    CudaCall(c, HistSortFloatsStableKernel).Read(valArr, len).Write(pDst);
}
}
