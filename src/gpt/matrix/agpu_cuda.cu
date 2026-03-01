#include <util/pch.h>
#define KERNEL_UNIT "agpu_cuda/"
#include "agpu_cuda.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void AddMatrixKernel(int xSize, int ySize, TCuda2DPtr<float> src, TCuda2DPtr<float> dst)
{
    for (int y = threadIdx.y + blockIdx.x * blockDim.y; y < ySize; y += blockDim.y * gridDim.x) {
        for (int x = 0; x < xSize; x += MM_TILE) {
            float4 srcVal = LoadWarpVec(src[y] + x);
            float4 val = srcVal + LoadWarpVec(dst[y] + x);
            StoreWarpVec(dst[y] + x, val);
        }
    }
}


__global__ void CopyMatrixKernel(int xSize, int ySize, TCuda2DPtr<float> src, TCuda2DPtr<float> dst)
{
    for (int y = threadIdx.y + blockIdx.x * blockDim.y; y < ySize; y += blockDim.y * gridDim.x) {
        for (int x = 0; x < xSize; x += MM_TILE) {
            float4 srcVal = LoadWarpVec(src[y] + x);
            StoreWarpVec(dst[y] + x, srcVal);
        }
    }
}


__global__ void MatrixRowSum2Kernel(int xSize, int ySize, TCuda2DPtr<float> matr, TCuda1DPtr<float> sum2arr)
{
    for (int y = threadIdx.y + blockIdx.x * blockDim.y; y < ySize; y += blockDim.y * gridDim.x) {
        float sum2 = 0;
        int x = 0;
        for (; x + 4 * WARP_VEC_DIM <= xSize; x += 4 * WARP_VEC_DIM) {
            float4 val0 = LoadWarpVec(matr[y] + x + 0 * WARP_VEC_DIM);
            float4 val1 = LoadWarpVec(matr[y] + x + 1 * WARP_VEC_DIM);
            float4 val2 = LoadWarpVec(matr[y] + x + 2 * WARP_VEC_DIM);
            float4 val3 = LoadWarpVec(matr[y] + x + 3 * WARP_VEC_DIM);
            sum2 += HorizontalSum(val0 * val0 + val1 * val1 + val2 * val2 + val3 * val3);
        }
        for (; x < xSize; x += WARP_VEC_DIM) {
            float4 val = LoadWarpVec(matr[y] + x);
            sum2 += HorizontalSum(val * val);
        }
        sum2arr[y] = WarpSum(sum2);
    }
}


inline __device__ float CalcRowScale(float rowDisp)
{
    return (rowDisp > 0) ? 1 / sqrt(rowDisp) : 0;
}

__global__ void UpdateGlobalRowDisp(int xSize, int ySize, int rowDispSize, int rowDispStep,
    TCuda1DPtr<float> deltaSum2arr, TAddGradientKernelParams *pParams, TCuda1DPtr<float> sumWeightArr, TCuda1DPtr<float> rowDispArr,
    TCuda1DPtr<float> rowScaleArr)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    TAddGradientKernelParams params = *pParams;
    float sumWeight = sumWeightArr[0];
    sumWeight = sumWeight * params.DispDecay + 1;
    float rowDispNorm = 1 / sumWeight;

    if (rowDispStep == 1) {
        for (int y = h + warpId * WARP_SIZE; y < ySize; y += blockDim.y * WARP_SIZE) {
            float rowDisp = (rowDispArr[y] * params.DispDecay) + (deltaSum2arr[y] / xSize);
            rowDispArr[y] = rowDisp;
            rowScaleArr[y] = CalcRowScale(rowDisp * rowDispNorm);
        }
    } else {
        for (int blockId = 0; blockId < rowDispSize; ++blockId) {
            int blkStart = blockId * rowDispStep;
            float sum2 = 0;
            for (int k = h + warpId * WARP_SIZE; k < rowDispStep; k += blockDim.y * WARP_SIZE) {
                int y = blkStart + k;
                sum2 += deltaSum2arr[y];
            }
            sum2 = BlockSum(sum2);

            __shared__ float rowScaleValue;
            if (h == 0 && warpId == 0) {
                float disp = sum2 / (xSize * rowDispStep);
                float rowDisp = (rowDispArr[blockId] * params.DispDecay) + disp;
                rowDispArr[blockId] = rowDisp;
                rowScaleValue = CalcRowScale(rowDisp * rowDispNorm);
            }
            __syncthreads();
            
            float rowScale = rowScaleValue;
            for (int k = h + warpId * WARP_SIZE; k < rowDispStep; k += blockDim.y * WARP_SIZE) {
                int y = blkStart + k;
                rowScaleArr[y] = rowScale;
            }
        }
    }
    if (h == 0 && warpId == 0) {
        sumWeightArr[0] = sumWeight;
    }
}


__global__ void AddGradientKernel(int xSize, int ySize, TCuda2DPtr<float> delta, TAddGradientKernelParams *pParams,
    TCuda1DPtr<float> rowScale, TCuda1DPtr<float> sparsity, TCuda2DPtr<float> avgGrad, TCuda2DPtr<float> weights,
    TCuda1DPtr<float> weightsRowSum2arr)
{
    TAddGradientKernelParams params = *pParams;
    float stepMult = params.StepMult * sqrt(sparsity[0]);
    float shrinkMult = GetShrinkMult(stepMult, params.L2Reg);
    for (int y = threadIdx.y + blockIdx.x * blockDim.y; y < ySize; y += blockDim.y * gridDim.x) {
        float deltaScale = rowScale[y];

        float *deltaRow = delta[y];
        float *avgGradRow = avgGrad[y];
        float *weightsRow = weights[y];
        float sum2 = 0;
        for (int x = 0; x < xSize; x += MM_TILE) {
            float4 delta = Scale(LoadWarpVec(deltaRow + x), deltaScale);
            float4 oldAG = LoadWarpVec(avgGradRow + x);
            float4 w = LoadWarpVec(weightsRow + x);
            float4 newAG = Scale(oldAG, params.Beta1) + Scale(delta, (1 - params.Beta1));
            float4 totalDelta = Scale(newAG, params.Weight1) + Scale(delta, params.Weight0);
            float4 val = Scale(w, shrinkMult) + Scale(totalDelta, stepMult);
            sum2 += HorizontalSum(val * val);
            StoreWarpVec(avgGradRow + x, newAG);
            StoreWarpVec(weightsRow + x, val);
        }
        weightsRowSum2arr[y] = WarpSum(sum2);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopyMatrixToHostKernel(
    int4 *dst, int4 *src, int dstStride, int srcStride, int rowWidthInBytes, int *rowRename, int rowCount)
{
    int rowWidth16 = rowWidthInBytes / 16;
    int rowTail = rowWidthInBytes - rowWidth16 * 16;
    int h = threadIdx.x;
    for (int y = threadIdx.y; y < rowCount; y += blockDim.y) {
        int yDst = rowRename[y];
        int4 *rowDst = AdvancePtr(dst, dstStride * yDst);
        int4 *rowSrc = AdvancePtr(src, srcStride * y);
        for (int x = h; x < rowWidth16; x += WARP_SIZE) {
            rowDst[x] = rowSrc[x];
        }
        if (h < rowTail) {
            int offset = rowWidth16 * 16 + h;
            *AdvancePtr((ui8 *)rowDst, offset) = *AdvancePtr((ui8 *)rowSrc, offset);
        }
    }
}

void CopyDeviceToHost(TCuda2DArray<float> *pHostArr, const TCuda2DArray<float> &devArr, const TAgpuMatrixWindow &win)
{
    TMemoryBlob hostMem = pHostArr->GetHostMem();
    TMemoryBlob devMem = devArr.GetDeviceMem();
    void *hostPtr = ((char *)hostMem.Ptr) + win.XOffset * sizeof(float);
    int rowSizeInBytes = win.LocalXSize * sizeof(float);
    CopyMatrixToHostKernel<<<1, dim3(32, 32)>>>((int4 *)hostPtr, (int4 *)devMem.Ptr, hostMem.Stride, devMem.Stride,
        rowSizeInBytes, win.RowRename.GetDevicePtr().Data, win.LocalYSize);
}
}
