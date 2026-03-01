#pragma once
#include "delta.h"
#include <lib/cuda/cuda_arrays.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
struct TCudaPackedDeltaMatrix
{
    TCuda2DArray<i8> Delta; // [y][x]
    TCuda2DArray<float> TileScale; // [tile][y]

    void AllocateCuda(int xSize, int ySize)
    {
        Y_ASSERT((xSize % MODEL_INT8_DELTA_TILE) == 0);
        Delta.AllocateCuda(xSize, ySize);
        TileScale.AllocateCuda(ySize, xSize / MODEL_INT8_DELTA_TILE);
    }

    void AllocateCuda(int xSize, int ySize, TPtrArg<TCudaMemoryPool> pool)
    {
        Y_ASSERT((xSize % MODEL_INT8_DELTA_TILE) == 0);
        Delta.AllocateCuda(xSize, ySize, pool);
        TileScale.AllocateCuda(ySize, xSize / MODEL_INT8_DELTA_TILE, pool);
    }

    void AllocateHost(yint xSize, yint ySize)
    {
        Delta.AllocateHost(xSize, ySize);
        Delta.ClearHostMem();
        TileScale.AllocateHost(ySize, xSize / MODEL_INT8_DELTA_TILE);
        TileScale.ClearHostMem();
    }

    THostPackedDeltaPtr GetDelta() { return THostPackedDeltaPtr(Delta.GetHostPtr(), TileScale.GetHostPtr()); }
    bool IsEmpty() const { return Delta.IsEmpty(); }
};
}
