#include <util/pch.h>
#define KERNEL_UNIT "layer_moe/"
#include "layer_ffn.cuh"
#include "moe_kernels.cuh"
#include "moe_matmul.cuh"
#include "layer_cuda_base.cuh"
#include "matmul_fwdbwd.cuh"
#include "cpu_util.h"
#include "gpu_layer_norm.cuh"
#include <lib/math/linear.h>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace NCuda;

class TCudaMoELayer : public TCudaLayerBase
{
    int SelectDim = 0;
    // expert assignment
    TCudaVector<int> SampleExpertIds;
    TCudaVector<float> SampleExpertWeights;
    TCudaVector<int> SampleExpertOffset;
    TCudaVector<int> ExpertOffset;
    TCudaVector<int> XpSampleId;
    TCudaVector<int> XpSampleWeight;
    TCudaVector<int> XpTileExpertId;
    // buffers
    TPreTransposedArray<TNormStateFloat> NormStateMoe;
    TCuda2DArray<half> MoeSelect;
    TCuda2DArray<half> Wide;
    TCuda2DArray<float> WideScale;
    TCuda2DArray<half> Gate;
    TCuda2DArray<half> ContractResult;
    TPreTransposedArray<TFastGradientFloat> StateGradLocal;
    TCuda2DArray<half> StateGradFull;
    // matrices
    TMoeMatMulBuffers MatmulExpand;
    TMoeMatMulBuffers MatmulGate;
    TMoeMatMulBuffers MatmulContract;
    TMatMulBuffers MatmulSelect;
    TIntrusivePtr<TMultiDevice2DArray<TNormStateFloat>> RG;
    TPreTransposedArray<TFastGradientFloat> DWide;
    TPreTransposedArray<TFastGradientFloat> DGate;
    TCuda2DArray<half> DMoeSelect;
    TIntrusivePtr<TMultiDeviceVector<float>> DExpertWeight;
    TIntrusivePtr<TMultiDevice2DArray<half>> DRG;
    // pool
    TCudaLayerPools Pools;

private:
    TCudaSpan GetFullExpertSpan() { return TCudaSpan(0, Dims.GetMoeExpertCount()); }
    TCudaSpan GetExpertSpan()
    {
        yint expertCount = Dims.GetMoeExpertCount();
        const TDeviceGroup &deviceGroup = MultiBuffers->GetDeviceGroup();
        yint dgSize = deviceGroup.GetSize();
        if (dgSize > 1) {
            Y_VERIFY((expertCount % dgSize) == 0);
            int perDevice = expertCount / dgSize;
            yint dgRank = deviceGroup.Rank(DeviceId);
            return TCudaSpan(dgRank * perDevice, (dgRank + 1) * perDevice);
        } else {
            return TCudaSpan(0, expertCount);
        }
    }

public:
    TCudaMoELayer(TPtrArg<TMultiDeviceBuffers> multiBuffers, const void *groupPtr, yint deviceId, const TModelDims &dims, bool updateMatrix,
        yint len, const TVector<TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>>> &matrArr, TCudaLayerPools &pools)
        : TCudaLayerBase(multiBuffers, deviceId, dims, updateMatrix, matrArr), Pools(pools)
    {
        yint dgSize = MultiBuffers->GetDgSize();
        int stateDim = Dims.Dim;
        int localStateDim = stateDim / dgSize;
        int expertCount = Dims.GetMoeExpertCount();
        int localExpertCount = expertCount / dgSize;
        SelectDim = RoundUp(expertCount, MM_TILE);
        int selectedCount = Dims.GetMoeSelectedCount();
        int xpDim = Dims.GetExpertDim();
        int lenMoe = RoundDown(len, MM_TILE) * Dims.MoeSelected + MM_TILE * expertCount;

        // expert assignment
        SampleExpertIds.AllocateCuda(selectedCount * len, pools.Fwd);
        SampleExpertWeights.AllocateCuda(selectedCount * len, pools.Fwd);
        SampleExpertOffset.AllocateCuda(selectedCount * len, pools.Fwd);
        ExpertOffset.AllocateCuda(expertCount + 2, pools.Fwd);
        XpSampleId.AllocateCuda(lenMoe, pools.Fwd);
        XpSampleWeight.AllocateCuda(lenMoe, pools.Fwd);
        XpTileExpertId.AllocateCuda(lenMoe / MM_TILE, pools.Fwd);
        // buffers
        NormStateMoe.AllocateCuda(stateDim, lenMoe, pools.Fwd);
        MoeSelect.AllocateCuda(SelectDim, len, pools.Fwd);
        Wide.AllocateCuda(xpDim, lenMoe, pools.Fwd);
        WideScale.AllocateCuda(lenMoe, MMTiles(xpDim), pools.Fwd);
        Gate.AllocateCuda(xpDim, lenMoe, pools.Fwd);
        RG = MultiBuffers->Fab().Create2DArray<TNormStateFloat>(Sprintf("%p-RG", groupPtr));
        RG->AllocateCuda(deviceId, xpDim, lenMoe, pools.Fwd);
        ContractResult.AllocateCuda(localStateDim, lenMoe, pools.Fwd);
        StateGradLocal.AllocateCuda(localStateDim, lenMoe, pools.Bwd);
        StateGradFull.AllocateCuda(stateDim, lenMoe, pools.Bwd);
        DWide.AllocateCuda(xpDim, lenMoe, pools.Bwd);
        DGate.AllocateCuda(xpDim, lenMoe, pools.Bwd);
        DMoeSelect.AllocateCuda(SelectDim, len, pools.Bwd);
        DExpertWeight = MultiBuffers->Fab().CreateVector<float>(Sprintf("%p-DExpertWeight", groupPtr));
        DExpertWeight->AllocateCuda(deviceId, lenMoe, pools.Bwd);
        DRG = MultiBuffers->Fab().Create2DArray<half>(Sprintf("%p-DRG", groupPtr));
        DRG->AllocateCuda(DeviceId, xpDim, lenMoe, pools.Bwd);
        MatmulExpand.AllocateCuda(MatrArr[MP_MOE_EXPAND], lenMoe, localExpertCount, pools.Bwd);
        MatmulGate.AllocateCuda(MatrArr[MP_MOE_GATE], lenMoe, localExpertCount, pools.Bwd);
        MatmulContract.AllocateCuda(MatrArr[MP_MOE_CONTRACT], lenMoe, expertCount, pools.Bwd);
        MatmulContract.AllocateCudaSrcTranspose(lenMoe, pools.Bwd);
        MatmulSelect.AllocateCuda(MatrArr[MP_MOE_SELECT], len, pools.Bwd);
        MatmulSelect.AllocateTransformFp16(pools.Bwd);
    }


    void ComputeRG(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState, int saveTransposed)
    {
        int stateDim = Dims.Dim;
        int xpDim = Dims.GetExpertDim();
        int selectedCount = Dims.GetMoeSelectedCount();
        int expertCount = Dims.GetMoeExpertCount();
        float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;

        MatmulSelect.Forward<TStoreScaled>(c, pParams, normState, &MoeSelect).Read(nullptr, stateNormScale);

        CudaCall(c, MoeCalcSampleExpertLists)
            .Grid(pParams->Len)
            .Shmem(expertCount * 4 + selectedCount * 8)
            .Read(expertCount, selectedCount, MoeSelect)
            .Write(&SampleExpertIds, &SampleExpertWeights);

        constexpr yint GROUP_EXPERT_WARPS = 32;
        CudaCall(c, GroupPerExpertLists)
            .Shmem(expertCount * sizeof(int) * GROUP_EXPERT_WARPS)
            .Block(WARP_SIZE, GROUP_EXPERT_WARPS)
            .Read(pParams->Len, pParams->LenMoe, expertCount, selectedCount, SampleExpertIds)
            .Write(&ExpertOffset, &SampleExpertOffset, &XpTileExpertId);

        CudaCall(c, InitXpTables)
            .Block(WARP_SIZE, MM_TILE / WARP_SIZE)
            .Grid(MMTiles(pParams->LenMoe))
            .Write(&XpSampleId, &XpSampleWeight);

        CudaCall(c, FillXpTables)
            .Block(WARP_SIZE, MM_TILE / WARP_SIZE)
            .Grid(MMTiles(pParams->LenMoe))
            .Read(pParams->Len, TIntDivision(selectedCount), SampleExpertIds, SampleExpertWeights, SampleExpertOffset, ExpertOffset)
            .Write(&XpSampleId, &XpSampleWeight);

        TCudaSpan expertSpan = GetExpertSpan();
        CudaCall(c, MoeGroupSamples<TNormStateFloat>)
            .Grid(MMTiles(stateDim), MMTiles(pParams->LenMoe))
            .Block(WARP_SIZE, MOE_GROUP_DATA_WARPS)
            .Shmem(sizeof(TMoeGroupSamplesData<TNormStateFloat>))
            .Read(saveTransposed)
            .Read(expertSpan, ExpertOffset, normState, XpSampleId)
            .Write(&NormStateMoe.Arr, &NormStateMoe.ArrT);

        MatmulExpand.Forward<TStoreRowTileNormalize>(c, pParams, NormStateMoe.Arr, &Wide, expertSpan, XpTileExpertId)
            .Read(stateNormScale, MOE_VEC_SCALE)
            .Write(&WideScale);
        MatmulGate.Forward<TStoreScaled>(c, pParams, NormStateMoe.Arr, &Gate, expertSpan, XpTileExpertId).Read(nullptr, stateNormScale);

        CudaCall(c, MatrixLoLUmoe<TNormStateFloat>)
            .Grid(MMTiles(xpDim), MMTiles(pParams->LenMoe))
            .Read(expertSpan, XpTileExpertId, XpSampleWeight, Gate, Wide)
            .Write(&RG->GetData(DeviceId));

        MultiBuffers->Op().AllGatherMoe(c, RG, pParams->LenMoe, DeviceId, expertSpan, XpTileExpertId);
    }


    void AddForward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState, TCuda2DArray<TStateFloat> *pState,
        TFragmentStates *pNextNormState) // override
    {
        Pools.SetMemPools(c);
        yint dgSize = MultiBuffers->GetDgSize();
        int stateDim = Dims.Dim;
        int localStateDim = stateDim / dgSize;
        int xpDim = Dims.GetExpertDim();
        int selectedCount = Dims.GetMoeSelectedCount();
        int expertCount = Dims.GetMoeExpertCount();
        TCudaSpan fullExpertSpan = GetFullExpertSpan();

        ComputeRG(c, pParams, normState, 0);

        float contractNormScale = CalcDotScale(xpDim) * MODEL_DISCR_SCALE * MOE_VEC_SCALE;
        MatmulContract.Forward<TStoreScaled>(c, pParams, RG->GetData(DeviceId), &ContractResult, fullExpertSpan, XpTileExpertId)
            .Read(nullptr, contractNormScale);

        float combineExpertsScale = CalcDotScale(expertCount);
        CudaCall(c, CombineExperts<TStoreLayerAddDelta<TNormStateFloat>, TStateFloat>)
            .Grid(MMTiles(localStateDim), pParams->Len)
            .Read(fullExpertSpan, ContractResult, selectedCount, SampleExpertIds, SampleExpertOffset, ExpertOffset)
            .Write(pState)
            .Struct()
            .Read(combineExpertsScale, MatmulContract.GetScale(), STATE_VEC_SCALE)
            .Write(&pNextNormState->GetLocal(), &pNextNormState->GetScaleLocal());
    }


    TCudaPOD<float> AddBackward(TPtrArg<TGraph> c, TComputeParams *pParams, TCuda2DArray<TNormStateFloat> &normState,
        TCuda2DArray<TFastGradientFloat> *pStateGrad, TCudaPOD<float> gradScale,
        TPtrArg<TMultiDeviceFRed2DArray<float>> dNormStatePtr) override
    {
        Pools.SetMemPools(c);
        yint dgSize = MultiBuffers->GetDgSize();
        int stateDim = Dims.Dim;
        int localStateDim = stateDim / dgSize;
        int xpDim = Dims.GetExpertDim();
        int selectedCount = Dims.GetMoeSelectedCount();
        int expertCount = Dims.GetMoeExpertCount();
        int expertCountFull = expertCount;
        TCudaSpan expertSpan = GetExpertSpan();
        int saveTransposed = (BWD_MATMUL_TYPE == MATMUL_FP8);

        ComputeRG(c, pParams, normState, saveTransposed);

        CudaCall(c, MoeGroupSamples<TFastGradientFloat>)
            .Grid(MMTiles(localStateDim), MMTiles(pParams->LenMoe))
            .Block(WARP_SIZE, MOE_GROUP_DATA_WARPS)
            .Shmem(sizeof(TMoeGroupSamplesData<TFastGradientFloat>))
            .Read(saveTransposed)
            .Read(GetFullExpertSpan(), ExpertOffset, *pStateGrad, XpSampleId)
            .Write(&StateGradLocal.Arr, &StateGradLocal.ArrT);

        // contract mulback
        auto bm = MatmulContract.BackpropMulti(c, pParams, StateGradLocal.Arr, GetFullExpertSpan(), XpTileExpertId);
        BackpropMoe(c, pParams, MultiArg() + bm, &DRG->GetData(DeviceId));
        MultiBuffers->Op().ReduceMoe(c, DRG, pParams->LenMoe, DeviceId, expertSpan, XpTileExpertId); // use FRed array

        CudaCall(c, BackpropRowTileNormalizeLoLUmoe)
            .Grid(MMTiles(pParams->LenMoe))
            .Block(WARP_SIZE, 32)
            .Shmem(sizeof(TBackpropNormLoLUmoeData))
            .Read(saveTransposed)
            .Read(expertSpan, XpTileExpertId, MMTiles(xpDim), XpSampleWeight, Gate, Wide, MOE_VEC_SCALE, WideScale)
            .Read(DRG->GetData(DeviceId), CalcDotScale(expertCount))
            .Write(&DGate.Arr, &DGate.ArrT, &DExpertWeight->GetData(DeviceId), &DWide.Arr, &DWide.ArrT);

        MultiBuffers->Op().AllGatherMoe(c, DExpertWeight, pParams->LenMoe, DeviceId, expertSpan, XpTileExpertId);

        CudaCall(c, BackpropMoeKernel)
            .Grid(pParams->LenRound)
            .Shmem(expertCountFull * 2 * sizeof(float))
            .Read(pParams->Len, SelectDim, expertCountFull, selectedCount)
            .Read(SampleExpertIds, SampleExpertWeights, SampleExpertOffset, ExpertOffset, DExpertWeight->GetData(DeviceId))
            .Write(&DMoeSelect);

        auto bm1 = MatmulExpand.BackpropMulti(c, pParams, DWide.Arr, expertSpan, XpTileExpertId);
        auto bm2 = MatmulGate.BackpropMulti(c, pParams, DGate.Arr, expertSpan, XpTileExpertId);
        BackpropMoe(c, pParams, MultiArg() + bm1 + bm2, &StateGradFull);

        TCuda2DArray<float> &dNormState = dNormStatePtr->GetData(DeviceId);
        CudaCall(c, CombineExperts<TStore, float>)
            .Grid(MMTiles(stateDim), pParams->Len)
            .Read(expertSpan, StateGradFull, selectedCount, SampleExpertIds, SampleExpertOffset, ExpertOffset)
            .Write(&dNormState)
            .Struct();

        MultiBuffers->Op().ReduceXSplit(c, dNormStatePtr, pParams->Len, DeviceId);

        MatmulSelect.BackpropFp16(c, pParams, normState, DMoeSelect, gradScale, &dNormState, RESULT_GRAD_ADD, UpdateLayers);

        MatmulContract.BackpropDelta(
            c, pParams, RG->GetData(DeviceId), StateGradLocal, gradScale, ExpertOffset, GetFullExpertSpan(), XpTileExpertId, UpdateLayers);
        MatmulExpand.BackpropDelta(c, pParams, NormStateMoe, DWide, gradScale, ExpertOffset, expertSpan, XpTileExpertId, UpdateLayers);
        MatmulGate.BackpropDelta(c, pParams, NormStateMoe, DGate, gradScale, ExpertOffset, expertSpan, XpTileExpertId, UpdateLayers);

        return MatmulContract.GetScale();
    }

    void AddDelta(TPtrArg<TGraph> c, EBackpropMode bm) override
    {
        if (UpdateLayers) {
            MatmulContract.AddDelta(c, bm);
            MatmulExpand.AddDelta(c, bm);
            MatmulGate.AddDelta(c, bm);
            MatmulSelect.AddDelta(c, bm);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void CpuMoe(const TModelDims &dims, const TArray2D<float> &moe, TArray2D<float> *pRes)
{
    *pRes = moe;
    yint len = moe.GetYSize();
    yint expertCount = dims.GetMoeExpertCount();
    yint selectedCount = dims.MoeSelected;
    float moeScale = sqrtf(expertCount);
    Y_ASSERT(moe.GetXSize() == expertCount);

    for (yint t = 0; t < len; ++t) {
        TVector<float> srcVec;
        float maxVal = -1e10f;
        for (yint x = 0; x < expertCount; ++x) {
            float val = moe[t][x];
            srcVec.push_back(val);
            (*pRes)[t][x] = 0;
            maxVal = Max(maxVal, val);
        }

        float sum2 = 0;
        for (int z = 0; z < selectedCount; ++z) {
            float top = -1e10;
            int topIdx = 0;
            for (yint x = 0; x < expertCount; ++x) {
                if (srcVec[x] > top) {
                    top = srcVec[x];
                    topIdx = x;
                }
            }
            Y_ASSERT(top * MOE_SCALE < 20);
            float w = exp((top - maxVal) * MOE_SCALE);
            (*pRes)[t][topIdx] = w;
            srcVec[topIdx] = -1e10;
            sum2 += w * w;
        }

        float scale = moeScale * sqrt(1 / sum2);
        for (yint x = 0; x < expertCount; ++x) {
            (*pRes)[t][x] *= scale;
        }
    }
}


static void CpuBackpropMoe(const TModelDims &dims, const TArray2D<float> &moeSrc, const TArray2D<float> &moe, TArray2D<float> *pDMoe)
{
    yint len = moe.GetYSize();
    yint expertCount = dims.GetMoeExpertCount();
    float moeScale = sqrtf(expertCount);
    Y_ASSERT(moeSrc.GetXSize() == expertCount);
    Y_ASSERT(moe.GetXSize() == expertCount);
    Y_ASSERT(pDMoe->GetXSize() == expertCount);

    for (yint t = 0; t < len; ++t) {
        TVector<float> srcVec;
        TVector<float> gradVec;
        for (yint x = 0; x < expertCount; ++x) {
            srcVec.push_back(moe[t][x] / moeScale);
            gradVec.push_back((*pDMoe)[t][x] * moeScale);
        }
        float avrgGrad = 0;
        for (yint x = 0; x < expertCount; ++x) {
            avrgGrad += srcVec[x] * gradVec[x];
        }
        TVector<float> srcGradVec;
        for (yint x = 0; x < expertCount; ++x) {
            float w = srcVec[x];
            srcGradVec.push_back(w * gradVec[x] - w * w * avrgGrad);
        }
        for (yint x = 0; x < expertCount; ++x) {
            (*pDMoe)[t][x] = srcGradVec[x] * MOE_SCALE;
        }
    }
}


static void ScaleExperts(const TArray2D<float> &moe, TArray2D<float> *pRes)
{
    yint len = moe.GetYSize();
    Y_ASSERT(len == pRes->GetYSize());
    yint expertCount = moe.GetXSize();
    yint expertSize = pRes->GetXSize() / expertCount;

    for (yint t = 0; t < len; ++t) {
        for (int expertId = 0; expertId < expertCount; ++expertId) {
            float scale = moe[t][expertId];
            int offset = expertId * expertSize;
            for (yint x = offset; x < offset + expertSize; ++x) {
                (*pRes)[t][x] *= scale;
            }
        }
    }
}

static void BackpropScaleExperts(const TArray2D<float> &moe, TArray2D<float> &rg, TArray2D<float> *pGradMoe, TArray2D<float> *pGradRg)
{
    yint len = moe.GetYSize();
    Y_ASSERT(len == rg.GetYSize());
    Y_ASSERT(len == pGradRg->GetYSize());
    Y_ASSERT(rg.GetXSize() == pGradRg->GetXSize());
    InitDeltaMatrix(pGradMoe, moe);
    yint expertCount = moe.GetXSize();
    yint expertSize = pGradRg->GetXSize() / expertCount;

    for (yint t = 0; t < len; ++t) {
        for (int expertId = 0; expertId < expertCount; ++expertId) {
            float scale = moe[t][expertId];
            int offset = expertId * expertSize;
            float gradMoe = 0;
            for (yint x = offset; x < offset + expertSize; ++x) {
                float grad = (*pGradRg)[t][x];
                gradMoe += rg[t][x] * grad;
                (*pGradRg)[t][x] *= scale;
            }
            (*pGradMoe)[t][expertId] = gradMoe;
        }
    }
}


// src[ [expert][e] ][ x ] -> res[ e ][ [expert][x] ]
inline TArray2D<float> FwdContractMatrix(const TArray2D<float> &src, yint xpCount)
{
    yint xSize = src.GetXSize();
    yint eSize = src.GetYSize() / xpCount;
    TArray2D<float> res;
    res.SetSizes(xpCount * xSize, eSize);
    for (yint xp = 0; xp < xpCount; ++xp) {
        for (yint e = 0; e < eSize; ++e) {
            for (yint x = 0; x < xSize; ++x) {
                res[e][xp * xSize + x] = src[xp * eSize + e][x];
            }
        }
    }
    return res;
}

// src[ e ][ [expert][x] ] -> res[ [expert][e] ][ x ]
inline TArray2D<float> BwdContractMatrix(const TArray2D<float> &src, yint xpCount)
{
    yint xSize = src.GetXSize() / xpCount;
    yint eSize = src.GetYSize();
    TArray2D<float> res;
    res.SetSizes(xSize, xpCount * eSize);
    for (yint xp = 0; xp < xpCount; ++xp) {
        for (yint e = 0; e < eSize; ++e) {
            for (yint x = 0; x < xSize; ++x) {
                res[xp * eSize + e][x] = src[e][xp * xSize + x];
            }
        }
    }
    return res;
}


template <class T>
void ContractMatmulBackprop(T transform, TArray2D<float> &src, TArray2D<float> &dstGrad, yint xpCount, bool updateMatrix, TArray2D<float> *pSrcGrad)
{
    MulBackwardWithAccum(pSrcGrad, FwdContractMatrix(GetData(transform), xpCount), dstGrad);
    if (updateMatrix) {
        TArray2D<float> delta;
        SumRankOne(src, &delta, dstGrad);
        transform->GetHostCompute()->HostApplyDelta(BwdContractMatrix(delta, xpCount));
    }
}


class TMoELayer : public TLayerBase
{
    TModelDims Dims;

    void ComputeForward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pState) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        int moeDim = Dims.GetMoeDim();
        yint xpCount = Dims.GetMoeExpertCount();

        *pState = prevState;
        TArray2D<float> normState = NormalizeState(*pState, dim / STATE_NORM_TILE);

        auto &matrExpand = MatrArr[MP_MOE_EXPAND];
        auto &matrGate = MatrArr[MP_MOE_GATE];
        auto &matrContract = MatrArr[MP_MOE_CONTRACT];
        auto &matrSelect = MatrArr[MP_MOE_SELECT];

        TArray2D<float> wide = MulForward(normState, GetData(matrExpand));
        TArray2D<float> wideNorm = NormalizeState(wide, MMTiles(moeDim));
        TArray2D<float> gate = MulForward(normState, GetDataNoScale(matrGate));
        TArray2D<float> moeSrc = MulForward(normState, GetDataNoScale(matrSelect));
        TArray2D<float> rg;
        LoLU(gate, wideNorm, &rg);

        TArray2D<float> moe;
        CpuMoe(Dims, moeSrc, &moe);
        ScaleExperts(moe, &rg);

        TArray2D<float> deltaState = MulForward(rg, FwdContractMatrix(GetData(matrContract), xpCount));
        AddScaledMatrix(pState, deltaState, 1);
    }

    void ComputeBackward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pGrad) override
    {
        yint len = prevState.GetYSize();
        yint dim = Dims.Dim;
        int moeDim = Dims.GetMoeDim();
        yint xpCount = Dims.GetMoeExpertCount();

        TArray2D<float> normState = NormalizeState(prevState, dim / STATE_NORM_TILE);
        TArray2D<float> dNormState;
        InitDeltaMatrix(&dNormState, normState);

        auto &matrExpand = MatrArr[MP_MOE_EXPAND];
        auto &matrGate = MatrArr[MP_MOE_GATE];
        auto &matrContract = MatrArr[MP_MOE_CONTRACT];
        auto &matrSelect = MatrArr[MP_MOE_SELECT];

        TArray2D<float> wide = MulForward(normState, GetData(matrExpand));
        TArray2D<float> wideNorm = NormalizeState(wide, MMTiles(moeDim));
        TArray2D<float> gate = MulForward(normState, GetDataNoScale(matrGate));
        TArray2D<float> moeSrc = MulForward(normState, GetDataNoScale(matrSelect));
        TArray2D<float> rgSrc;
        LoLU(gate, wideNorm, &rgSrc);

        TArray2D<float> moe;
        CpuMoe(Dims, moeSrc, &moe);
        TArray2D<float> rg = rgSrc;
        ScaleExperts(moe, &rg);

        TArray2D<float> drg;
        InitDeltaMatrix(&drg, rg);
        ContractMatmulBackprop(matrContract, rg, *pGrad, xpCount, UpdateLayers, &drg);

        TArray2D<float> dMoe;
        BackpropScaleExperts(moe, rgSrc, &dMoe, &drg);

        CpuBackpropMoe(Dims, moeSrc, moe, &dMoe);
        MatmulBackpropNoScale(matrSelect, normState, dMoe, UpdateLayers, &dNormState);

        TArray2D<float> dgate;
        TArray2D<float> dwide;
        BackpropLoLU(gate, wideNorm, drg, &dgate, &dwide);
        NormalizeStateBackward(wide, MMTiles(moeDim), dwide, &dwide);

        MatmulBackprop(matrExpand, normState, dwide, UpdateLayers, &dNormState);
        MatmulBackpropNoScale(matrGate, normState, dgate, UpdateLayers, &dNormState);

        TArray2D<float> deltaStateGrad;
        NormalizeStateBackward(prevState, dim / STATE_NORM_TILE, dNormState, &deltaStateGrad);
        AddScaledMatrix(pGrad, deltaStateGrad, 1);
    }

    TCudaLayerBase *CreateCudaLayer(TPtrArg<TCudaMemoryAllocator> cudaMem, TPtrArg<TMultiDeviceBuffers> multiBuffers, TStream &stream,
        yint deviceId, yint len, const TVector<TIntrusivePtr<NCuda::ICudaModelMatrixBase<TFastModelFloat>>> &matrArr,
        TCudaLayerPools &pools) override
    {
        return new TCudaMoELayer(multiBuffers, this, deviceId, Dims, UpdateLayers, len, matrArr, pools);
    }

public:
    TMoELayer(const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr)
        : TLayerBase(modelDescr, matrArr), Dims(modelDescr.Dims)
    {
    }
};


TIntrusivePtr<TLayerBase> CreateMoELayer(
    const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr)
{
    return new TMoELayer(modelDescr, matrArr);
}
