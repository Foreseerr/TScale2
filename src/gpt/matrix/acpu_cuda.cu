#include <util/pch.h>
#define KERNEL_UNIT "acpu_cuda/"
#include "acpu_cuda.cuh"
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
//
__device__ void CopyTileScale(int deltaXSize, int deltaYSize, TCuda2DPtr<float> srcTileScale, TCuda2DPtr<float> dstTileScale)
{
    if (srcTileScale.GetStrideInBytes() == dstTileScale.GetStrideInBytes()) {
        int tileCount = deltaXSize / MODEL_INT8_DELTA_TILE;
        int lenBytes = dstTileScale.GetStrideInBytes() * tileCount;
        ui8 *srcPtr = (ui8 *)srcTileScale[0];
        ui8 *dstPtr = (ui8 *)dstTileScale[0];
        int thrOffset = threadIdx.x * 16 + threadIdx.y * blockDim.x * 16;
        for (int offset = thrOffset; offset < lenBytes; offset += blockDim.y * blockDim.x * 16) {
        // overwrite up to 15 bytes, but no buffer overrun will happen since buffer sizes are rounded up
            *(int4 *)(dstPtr + offset) = *(int4 *)(srcPtr + offset);
        }
    } else {
        int tileCount = deltaXSize / MODEL_INT8_DELTA_TILE;
        for (int tile = threadIdx.y; tile < tileCount; tile += blockDim.y) {
            for (int y = threadIdx.x; y < deltaYSize; y += blockDim.x) {
                dstTileScale[tile][y] = srcTileScale[tile][y];
            }
        }
    }
}


__global__ void CopyDelta(TCuda2DPtr<float> srcArr, int xSize, int ySize, TCuda1DPtr<int> iterCounter, TCuda2DPtr<float> tileScaleBuf,
    TCuda2DPtr<i8> dstArr, TCuda2DPtr<float> dstTileScale, int *launchFlag)
{
    CUDA_STATIC_ASSERT(MODEL_INT8_DELTA_TILE == 128);
    CUDA_ASSERT((xSize % 128) == 0);

    int h = threadIdx.x;
    int warpId = threadIdx.y;

    constexpr int LINE = 512;

    __shared__ i8 resPacked[COPY_DELTA_WARPS][LINE];
    __shared__ float resScale[COPY_DELTA_WARPS][LINE / 128];

    for (int offsetY = 0; offsetY < ySize; offsetY += COPY_DELTA_WARPS) {
        int y = offsetY + warpId;
        if (y >= ySize) {
            break;
        }
        i8 *dstRowPtr = dstArr[y];
        for (int offsetX = 0; offsetX < xSize; offsetX += LINE) {
            // each warp packs LINE elements in 128 width blocks
            // read and pack data
            for (int blk = 0; blk < LINE / 128; ++blk) {
                int x = offsetX + blk * 128 + h * 4;
                float maxVal = 0;
                float4 val4;
                if (x < xSize) {
                    val4 = *(float4 *)(srcArr[y] + x);
                    maxVal = max(maxVal, max(max(fabsf(val4.x), fabsf(val4.y)), max(fabsf(val4.z), fabsf(val4.w))));
                }
                maxVal = WarpMax(maxVal);

                float scale = (maxVal > 0) ? maxVal / 127 : 0;
                float mult = (maxVal > 0) ? 1 / scale : 0;
                resScale[warpId][blk] = scale;

                // convert
                if (x < xSize) {
                    union {
                        int res4;
                        i8 res[4];
                    };
                    res[0] = CvtToI8(val4.x * mult);
                    res[1] = CvtToI8(val4.y * mult);
                    res[2] = CvtToI8(val4.z * mult);
                    res[3] = CvtToI8(val4.w * mult);
                    *(int *)(&resPacked[warpId][blk * 128 + h * 4]) = res4;
                }
            }
            __syncwarp();
            // write packed data
            int thrOffset = h * 16;
            int4 packedData = *(int4 *)&resPacked[warpId][thrOffset];
            int writeX = offsetX + thrOffset;
            if (writeX < xSize) {
                *(int4 *)&dstRowPtr[writeX] = packedData;
            }

            // write scale
            int warpWidth = min(LINE, xSize - offsetX);
            if (h < warpWidth / 128) {
                tileScaleBuf[(offsetX / 128) + h][y] = resScale[warpId][h];
            }
            __syncwarp();
        }
    }
    __syncthreads();

    // copy scale
    CopyTileScale(xSize, ySize, tileScaleBuf, dstTileScale);
    // apply
    __threadfence_system(); // flush cache
    __syncthreads();
    if (h == 0 && warpId == 0) {
        *launchFlag = iterCounter[0];
    }
    __threadfence_system(); // neccessary, does not happen on kernel finish
}


__global__ void CopyPackedDelta(TCuda2DPtr<i8> srcArr, TCuda2DPtr<float> srcTileScale, int xSize, int ySize, TCuda1DPtr<int> iterCounter,
    TCuda2DPtr<i8> dstArr, TCuda2DPtr<float> dstTileScale, int *launchFlag)
{
    CUDA_STATIC_ASSERT(MODEL_INT8_DELTA_TILE == 128);
    CUDA_STATIC_ASSERT(MODEL_INT8_DELTA_TILE == MM_TILE);
    CUDA_ASSERT((xSize % 128) == 0);

    int h = threadIdx.x;
    int warpId = threadIdx.y;

    // copy data
    for (int offsetY = 0; offsetY < ySize; offsetY += COPY_DELTA_WARPS) {
        int y = offsetY + warpId;
        if (y >= ySize) {
            break;
        }
        i8 *dstRowPtr = dstArr[y];
        for (int base = 0; base < xSize; base += 16 * WARP_SIZE) {
            int x = base + h * 16;
            int4 data;
            if (x < xSize) {
                data = *(int4 *)&srcArr[y][x];
            }
            __syncwarp();
            if (x < xSize) {
                *(int4 *)&dstRowPtr[x] = data;
            }
        }
    }
    // copy scale
    CopyTileScale(xSize, ySize, srcTileScale, dstTileScale);
    // apply
    __threadfence_system(); // flush cache
    __syncthreads();
    if (h == 0 && warpId == 0) {
        *launchFlag = iterCounter[0];
    }
    __threadfence_system(); // neccessary, does not happen on kernel finish
}


__global__ void LaunchOpKernel(TCuda1DPtr<int> iterCounter, int *launchOpPtr)
{
    if (threadIdx.x == 0) {
        *launchOpPtr = iterCounter[0];
    }
    __threadfence_system(); // flush cache asap
}


__global__ void AssignIterCounterKernel(int *hostIterCounter, TCuda1DPtr<int> iterCounter)
{
    if (threadIdx.x == 0) {
        iterCounter[0] = *hostIterCounter;
    }
}

}
