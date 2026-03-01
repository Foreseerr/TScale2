namespace NCUDA_Transformer
{
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TStateGradScale
{
    yint DeviceId = 0;
    TCudaVector<float> LayerGradScale; // number to multiply gradFast to obtain grad
    TCudaVector<float> LayerGradMult; // result of ComputeGradMult()
    TIntrusivePtr<TMultiDeviceVector<float>> LayerGradMaxNorm;

public:
    void AllocateCuda(TPtrArg<TMultiDeviceBuffers> multiBuffers, yint deviceId, int maxStepId)
    {
        DeviceId = deviceId;
        LayerGradMaxNorm = multiBuffers->Fab().CreateVector<float>("StateGradMaxNorm");
        LayerGradMaxNorm->AllocateCuda(DeviceId, RoundUp(maxStepId, MM_TILE), null_ptr_arg);
        //
        TVector<float> gradScale;
        ClearPodArray(&gradScale, maxStepId);
        gradScale[0] = 1;
        LayerGradScale.Init(gradScale);
        LayerGradMult.Init(gradScale);
    }
    TCudaPOD<float> GetGradScale(yint d) { return LayerGradScale.GetElement(d); }
    TCudaPOD<float> GetGradMult(yint d) { return LayerGradMult.GetElement(d); }
    TCudaPOD<float> GetGradMaxNorm(yint d) { return LayerGradMaxNorm->GetData(DeviceId).GetElement(d); }
    void ClearMaxNorm(TPtrArg<TGraph> c) { c->ClearMem(&LayerGradMaxNorm->GetData(DeviceId)); }
    void SyncMax(TPtrArg<TGraph> c, TPtrArg<TMultiDeviceBuffers> multiBuffers) { multiBuffers->Op().AllMax(c, LayerGradMaxNorm, DeviceId); }
};


// gradient scaling
__forceinline __device__ float ComputeGradMult(float gradMaxNorm)
{
    const float TARGET_MAX_NORM = 128;
    if (gradMaxNorm == 0) {
        return 1; // zero gradients can be multiplied by any number
    } else {
        return TruncateToPow2(TARGET_MAX_NORM / gradMaxNorm);
    }
}


__global__ void ScaleGrad(int layerId, float *gradMaxNorm, float *prevGradMult, TCuda2DPtr<TStateFloat> grad1,
    TCuda2DPtr<TFastGradientFloat> gradFast, float *pGradScale, float *pGradMult)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    float gradMult = ComputeGradMult(*gradMaxNorm);
    if (gradMult != *prevGradMult) {
        float4 v = LoadWarpVec(grad1[t] + offset);
        v = Scale(v, gradMult);
        StoreWarpVec(gradFast[t] + offset, v);
    }
    if (t == 0 && tile == 0 && threadIdx.x == 0) {
        CUDA_ASSERT(gradMult != 0);
        *pGradScale = 1 / gradMult;
        *pGradMult = gradMult;
    }
}


__global__ void RecvGradKernel(int xSize, int len, TCuda2DPtr<TStateFloat> grad, float *pGradMaxNorm, float *pGradMult)
{
    float4 maxVal = ZeroWarpVec();
    for (int t = threadIdx.y + blockIdx.x * blockDim.y; t < len; t += blockDim.y * gridDim.x) {
        for (int offset = 0; offset < xSize; offset += WARP_VEC_DIM) {
            float4 v = LoadWarpVec(grad[t] + offset);
            maxVal = Max(maxVal, Fabs(v));
        }
    }
    float gradMax = CalcWarpVecMaxAbsValue(maxVal);
    if (threadIdx.x == 0) {
        *pGradMult = 1e38f;
        atomicMax((int *)pGradMaxNorm, __float_as_int(gradMax));
    }
}
KERNEL_BLOCK_SIZE(RecvGradKernel, WARP_SIZE, MAX_WARPS)
}
