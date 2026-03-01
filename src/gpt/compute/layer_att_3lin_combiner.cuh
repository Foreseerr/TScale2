#pragma once


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float WarpShuffle(float val, int blk)
{
    return __shfl_xor_sync(0xffffffff, val, blk);
}

template <class TRes>
__global__ void KVProductKernel(
    int len, int kvRep,
    TCuda2DPtr<half> gateArr, TCuda2DPtr<half> vArr,
    TCuda2DPtr<TRes> ccVecArr
)
{
    int h = threadIdx.x;
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    if (t < len) {
        float4 gate = LoadWarpVec(gateArr[t] + offset);
        float4 value = LoadWarpVec(vArr[t] + offset);
        
        __shared__ float sgate[MM_TILE];
        __shared__ float svalue[MM_TILE];
        __shared__ float svec[MM_TILE];
        StoreWarpVec(sgate, gate);
        StoreWarpVec(svalue, value);
        __syncwarp();

        for (int blk = 0; blk < kvRep; ++blk) {
            for (int wxOffset = 0; wxOffset < MM_TILE; wxOffset += WARP_SIZE) {
                int x = wxOffset + h;
                svec[x] = sgate[x ^ blk] * svalue[x] * (V_VEC_SCALE * V_VEC_SCALE / V_VEC_SCALE);
            }
            __syncwarp();
            yint ccOffset = offset * kvRep + blk * MM_TILE;
            StoreWarpVec(ccVecArr[t] + ccOffset, LoadWarpVecSmem(svec));
            __syncwarp();
        }
    } else {
        for (int blk = 0; blk < kvRep; ++blk) {
            StoreZeroWarpVec(ccVecArr[t] + offset * kvRep + blk * MM_TILE);
        }
    }
}


template <class TGateGrad>
__global__ void KVProductBackpropKernel(int len, int kvRep, TCuda2DPtr<half> gateArr, TCuda2DPtr<half> valueArr, TCuda2DPtr<half> dccArr,
    TCuda2DPtr<TGateGrad> dGateArr, TCuda2DPtr<half> dvArr)
{
    int h = threadIdx.x;
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    if (t < len) {
        float4 gate = LoadWarpVec(gateArr[t] + offset);
        gate = Scale(gate, V_VEC_SCALE);
        float4 value = LoadWarpVec(valueArr[t] + offset);
        value = Scale(value, V_VEC_SCALE);

        __shared__ float sgate[MM_TILE];
        __shared__ float svalue[MM_TILE];
        StoreWarpVec(sgate, gate);
        StoreWarpVec(svalue, value);

        __shared__ float sdGate[MM_TILE];
        __shared__ float sdValue[MM_TILE];
        StoreWarpVec(sdGate, ZeroWarpVec());
        StoreWarpVec(sdValue, ZeroWarpVec());
        __syncwarp();

        __shared__ float sdcc[MM_TILE];
        for (int blk = 0; blk < kvRep; ++blk) {
            yint ccOffset = offset * kvRep + blk * MM_TILE;
            float4 dcc = LoadWarpVec(dccArr[t] + ccOffset);
            StoreWarpVec(sdcc, dcc);
            __syncwarp();

            for (int wxOffset = 0; wxOffset < MM_TILE; wxOffset += WARP_SIZE) {
                int x = wxOffset + h;
                sdGate[x ^ blk] += sdcc[x] * svalue[x];
                sdValue[x] += sdcc[x] * sgate[x ^ blk];
            }
            __syncwarp();
        }

        StoreWarpVec(dGateArr[t] + offset, LoadWarpVecSmem(sdGate));
        StoreWarpVec(dvArr[t] + offset, LoadWarpVecSmem(sdValue));
    } else {
        StoreZeroWarpVec(dGateArr[t] + offset);
        StoreZeroWarpVec(dvArr[t] + offset);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU reference
//
static void KVProduct(yint kvRep, const TArray2D<float> &gateState, const TArray2D<float> &valLookup,
    TArray2D<float> *pCCState)
{
    yint ttSum = gateState.GetXSize();
    yint len = gateState.GetYSize();
    yint tileCount = ttSum / MM_TILE;
    Y_ASSERT(valLookup.GetXSize() == ttSum);
    Y_ASSERT(valLookup.GetYSize() == len);
    pCCState->SetSizes(ttSum * kvRep, len);
    for (yint t = 0; t < len; ++t) {
        for (yint tileId = 0; tileId < tileCount; ++tileId) {
            yint srcOffset = tileId * MM_TILE;
            for (int blk = 0; blk < kvRep; ++blk) {
                yint ccOffset = (tileId * kvRep + blk) * MM_TILE;
                for (yint k = 0; k < MM_TILE; ++k) {
                    yint srcK = srcOffset + k;
                    yint ccK = ccOffset + k;
                    float gateShfl = gateState[t][srcK ^ blk];
                    float value = valLookup[t][srcK];
                    (*pCCState)[t][ccK] = gateShfl * value;
                }
            }
        }
    }
}


static void KVProductBackprop(yint kvRep, const TArray2D<float> &gateState, const TArray2D<float> &valLookup, const TArray2D<float> &dcc,
    TArray2D<float> *pDGateState, TArray2D<float> *pDValLookup)
{
    yint ttSum = gateState.GetXSize();
    yint len = gateState.GetYSize();
    yint tileCount = ttSum / MM_TILE;
    Y_ASSERT(gateState.GetXSize() == ttSum);
    Y_ASSERT(valLookup.GetXSize() == ttSum);
    Y_ASSERT(valLookup.GetYSize() == len);
    Y_ASSERT(dcc.GetXSize() == ttSum * kvRep);
    Y_ASSERT(dcc.GetYSize() == len);

    TArray2D<float> &dGate = *pDGateState;
    dGate.SetSizes(ttSum, len);
    dGate.FillZero();
    TArray2D<float> &dValLookup = *pDValLookup;
    dValLookup.SetSizes(ttSum, len);
    dValLookup.FillZero();
    for (yint t = 0; t < len; ++t) {
        for (yint tileId = 0; tileId < tileCount; ++tileId) {
            yint srcOffset = tileId * MM_TILE;
            for (int blk = 0; blk < kvRep; ++blk) {
                yint ccOffset = (tileId * kvRep + blk) * MM_TILE;
                for (yint k = 0; k < MM_TILE; ++k) {
                    yint srcK = srcOffset + k;
                    yint ccK = ccOffset + k;
                    dGate[t][srcK] += dcc[t][ccK ^ blk] * valLookup[t][srcK ^ blk];
                    dValLookup[t][srcK] += dcc[t][ccK] * gateState[t][srcK ^ blk];
                }
            }
        }
    }
}
}
