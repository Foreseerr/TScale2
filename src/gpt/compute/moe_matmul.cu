#include <util/pch.h>
#define KERNEL_UNIT "moe_matmul/"
#include "moe_matmul.cuh"


/////////////////////////////////////////////////////////////////////////////////////
namespace NCuda
{
__global__ void MoeTransposeI8Kernel(TCuda2DPtr<i8> src, TCudaSpan expertSpan, TCuda1DPtr<int> tileExpert, TCuda2DPtr<i8> dst)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    int expertId = tileExpert[blockIdx.y];
    if (expertId < expertSpan.Beg || expertId >= expertSpan.Fin) {
        return;
    }
    int xBlock = blockIdx.x * MM_TILE;
    int yBlock = blockIdx.y * MM_TILE;
    TransposeKernelImpl(xBlock, yBlock, src, dst);
}
}
