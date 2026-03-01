#pragma once

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TDst>
__global__ void LayerNormalizeStateVecs(int len, float vecScale,
    TCuda2DPtr<TSrc> state,
    TCuda2DPtr<TDst> normState, TCuda2DPtr<float> normStateScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;

    float discrScale = 0;
    float4 vec;
    if (t < len) {
        vec = LoadWarpVec(state[t] + offset);
        float sum2 = CalcWarpVecSum2(vec);
        if (sum2 > 0) {
            discrScale = sqrtf(sum2 / MM_TILE) * vecScale;
            vec = Scale(vec, 1 / discrScale);
        }
    } else {
        vec = ZeroWarpVec();
    }
    StoreWarpVec(normState[t] + offset, vec);
    if (h == 0) {
        normStateScale[tile][t] = discrScale;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// res can point to src or grad
inline __device__ void BackpropNormalizeSmem(int dim, float *src, float *grad, float *res)
{
    int h = threadIdx.x;
    float sum2 = 0;
    float dp = 0;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        float val = src[d];
        float valGrad = grad[d];
        sum2 += val * val;
        dp += val * valGrad;
    }
    sum2 = WarpSum(sum2);
    if (sum2 == 0) {
        for (int base = 0; base < dim; base += WARP_SIZE) {
            int d = base + h;
            res[d] = 0;
        }
    } else {
        // add gradient and update gradMax
        dp = WarpSum(dp);

        float sigma = dp / sum2;
        float scale = sqrtf(dim / sum2);
        for (int base = 0; base < dim; base += WARP_SIZE) {
            int d = base + h;
            float resGrad = scale * (grad[d] - src[d] * sigma);
            res[d] = resGrad;
        }
    }
}


template <class TSrc>
inline __device__ void LoadAddSmemVec(int dim, TSrc *src, float *buf)
{
    // could use async loads to fully utilize bandwidth with single warp?
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        buf[d] += float(src[d]);
    }
}

inline __device__ void ScaleSmemVec(int dim, float *buf, float mult)
{
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        buf[base + h] *= mult;
    }
}

template <class TSrc, class TGrad, class TGradFast>
__global__ void BackpropLayerNormalize(TCuda2DPtr<TSrc> normState, TCuda2DPtr<float> stateScale, TCuda2DPtr<float> dNormState,
    float *combinerScale, float *gradScale, float *gradMult, TCuda2DPtr<TGrad> pStateGrad, TCuda2DPtr<TGradFast> pStateGradFast,
    float *nextLayerGradMaxNorm)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;

    float4 vec = LoadWarpVec(normState[t] + offset);
    vec = Scale(vec, stateScale[tile][t]);

    float4 grad = LoadWarpVec(dNormState[t] + offset);
    grad = Scale(grad, *combinerScale * *gradScale);

    grad = TileNormalizeBackpropWarpVec(vec, grad);

    grad = grad + LoadWarpVec(pStateGrad[t] + offset);
    float gradMax = CalcWarpVecMaxAbsValue(grad);
    StoreWarpVec(pStateGrad[t] + offset, grad);
    StoreWarpVec(pStateGradFast[t] + offset, Scale(grad, *gradMult));

    if (h == 0) {
        atomicMax((int *)nextLayerGradMaxNorm, __float_as_int(gradMax));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TGrad, class TGradFast>
__global__ void BackpropFinalNormalize(int len, TCuda2DPtr<TSrc> normState, TCuda2DPtr<float> stateScale, TCuda2DPtr<TGrad> dNormState,
    TCuda2DPtr<TGrad> pStateGrad, TCuda2DPtr<TGradFast> pStateGradFast, float *nextLayerGradMaxNorm)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;

    if (t < len) {
        float4 vec = LoadWarpVec(normState[t] + offset);
        vec = Scale(vec, stateScale[tile][t]);

        float4 grad = LoadWarpVec(dNormState[t] + offset);

        grad = TileNormalizeBackpropWarpVec(vec, grad);

        float gradMax = CalcWarpVecMax(grad);
        StoreWarpVec(pStateGrad[t] + offset, grad);
        StoreWarpVec(pStateGradFast[t] + offset, grad);
        if (h == 0) {
            atomicMax((int *)nextLayerGradMaxNorm, __float_as_int(gradMax));
        }
    } else {
        StoreZeroWarpVec(pStateGrad[t] + offset);
        StoreZeroWarpVec(pStateGradFast[t] + offset);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TNormFloat>
struct TStoreLayerAddDelta : public TStoreRowBase
{
    struct TParams
    {
        float Scale;
        float *ScalePtr;
        float VecScale; // for normalization
        TCuda2DPtr<TNormFloat> NormState;
        TCuda2DPtr<float> NormStateScale;
    };

    __device__ static float GetScale(TParams &params)
    {
        float sumScale = params.Scale;
        if (params.ScalePtr) {
            sumScale *= *params.ScalePtr;
        }
        return sumScale;
    }

    template <class TRes>
    __device__ static void StoreRow(TParams &params, TWgShmem &shmem, int tile, int dstX, int dstY, float *mmRes, TCuda2DPtr<TRes> resBuf)
    {
        // add delta
        float4 vec = LoadWarpVecSmem(mmRes);
        vec = vec + LoadWarpVec(resBuf[dstY] + dstX);
        StoreWarpVec(resBuf[dstY] + dstX, vec);
        // normalize vec
        float sum2 = CalcWarpVecSum2(vec);
        float discrScale = 0;
        if (sum2 > 0) {
            discrScale = sqrt(sum2 / MM_TILE) * params.VecScale;
            vec = Scale(vec, 1 / discrScale);
        }
        StoreWarpVec(params.NormState[dstY] + dstX, vec);
        if (threadIdx.x == 0) {
            params.NormStateScale[tile][dstY] = discrScale;
        }
        resBuf.CheckCoords(dstX, dstY);
    }
};
}
