#include <util/pch.h>
#define KERNEL_UNIT "gpt_cuda/"
#include "gpu_transformer.cuh"
#include "cfg_precision.h"
#include "layer.h"
#include "state_buf.cuh"
#include "layer_cuda_base.cuh"
#include <lib/cuda/cuda_util.cuh>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/cuda_mma.cuh>
#include <lib/cuda/cuda_i8.cuh>
#include <lib/cuda/cuda_fp8.cuh>
#include <lib/cuda/cuda_fp16.cuh>
#include <lib/cuda/vec_util.cuh>
#include <lib/cuda/cuda_init.h>
#include <gpt/data/data.h>
#include <gpt/att/nodes_batch.h>
#include <gpt/att/rope.h>
#include <lib/random/rand_utils.h>
#include <lib/math/linear.h>
#include <util/radix_sort.h>
#include <util/thread.h>
#include <emmintrin.h>
// kernels
#include "gpu_embedding.cuh"
#include "gpu_final.cuh"
#include "gpu_layer_norm.cuh"
#include "gpu_grad_scale.cuh"


using namespace NCuda;


namespace NCUDA_Transformer
{
///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMicroBatchCtx
{
    TBatchNodes Nodes;
    TArray2D<float> SampleEmbedVectors;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TSinlgeComputeContext : public TThrRefBase
{
    struct TMicroBatch : public TThrRefBase
    {
        TVector<TIntrusivePtr<TFragmentStates>> AllStates;
        TComputeParams ComputeParams;
        TVector<TAttentionInfoGrouped<ATT_GROUP>> AttGroupsArr;
        TVector<TAttentionInfoGrouped<ATT_GROUP>> RevAttGroupsArr;
    };

    TStream Stream;
    yint DeviceId = 0;
    yint LocationId = 0;
    TVector<yint> LocationDeviceId;
    bool IsHostMasterModel = false;
    TIntrusivePtr<TCudaMemoryAllocator> CudaMem;
    TIntrusivePtr<TCudaMemoryPool> PoolEmbed;
    TIntrusivePtr<TCudaMemoryPool> PoolLayers;
    TIntrusivePtr<TCudaMemoryPool> PoolFinal;
    TIntrusivePtr<TMultiDeviceBuffers> MultiBuffers;
    TModelDescr ModelDescr;
    TModelSplit MSplit;
    TIntrusivePtr<IModelOps> ModelOps;
    // layers
    TIntrusivePtr<ICudaModelMatrixBase<TEmbedFloat>> Embedding;
    TEmbeddingLayer EmbedLayer;
    TVector<TIntrusivePtr<TCudaLayerBase>> LayerArr;
    TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>> Final;
    TFianlLayer FinalLayer;
    TIntrusivePtr<TModelVector> Bias;
    // compute buffers
    TIntrusivePtr<TStateBuffer> State[2];
    TCuda2DArray<TFastGradientFloat> StateGradFast;
    TStateGradScale GradScale;
    TIntrusivePtr<TMultiDeviceFRed2DArray<float>> DNormState;
    TVector<TIntrusivePtr<TMicroBatch>> MBatchArr;

    bool ComputeInFly = false;
    bool NeedCopyToDevice = true;

    TIntrusivePtr<TGraph> ForwardComputer;
    TIntrusivePtr<TGraph> CMForwardComputer;
    TIntrusivePtr<TGraph> CopyModelToDeviceGraph;
    TVector<TIntrusivePtr<TGraph>> BackpropComputerArr;
    TVector<TIntrusivePtr<TGraph>> CMBackpropComputerArr;
    yint GraphId = 0;

public:
    TSinlgeComputeContext(yint deviceId, yint deviceGroupId, yint locationId, TVector<yint> &locationDeviceId, const TModelSplit &msplit,
        TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount, bool isHostMasterModel, ETargetType targetType,
        TPtrArg<ICustomLoss> pLoss, TIntrusivePtr<TStateBuffer> &state0, TIntrusivePtr<TStateBuffer> &state1)
        : DeviceId(deviceId), LocationId(locationId), LocationDeviceId(locationDeviceId), IsHostMasterModel(isHostMasterModel),
          ModelDescr(modelStorage->GetModelDescr()), MSplit(msplit), ModelOps(modelOps), EmbedLayer(modelStorage->GetModelDescr()),
          FinalLayer(deviceId, modelStorage->GetModelDescr(), targetType, pLoss)
    {
        // dimension restrictions
        Y_VERIFY(ModelDescr.Dims.Dim % MM_TILE == 0);
        Y_VERIFY(ModelDescr.Dims.QDim == Q_DIM);
        Y_VERIFY(ModelDescr.Dims.TTDim == TT_DIM);
        Y_VERIFY(Q_DIM == MM_TILE);
        Y_VERIFY(TT_DIM == MM_TILE);
        Y_VERIFY(STATE_NORM_TILE == MM_TILE);

        // multi buffers
        MultiBuffers = ModelOps->GetMultiBuffers(deviceGroupId).Get();

        // params
        yint dgSize = MultiBuffers->GetDgSize();
        yint dimFull = ModelDescr.Dims.Dim;
        yint dim = dimFull / dgSize;
        Y_VERIFY(dim % MM_TILE == 0);
        yint maxLen = RoundUp(nodeCount, MM_TILE);
        yint microBatchCount = MSplit.MicroBatchCount;
        bool isPipelineParallel = MSplit.PP > 1;

        //
        CudaMem = new TCudaMemoryAllocator();
        PoolEmbed = CudaMem->CreatePool();
        PoolLayers = CudaMem->CreatePool();
        PoolFinal = CudaMem->CreatePool();
        yint layerPoolCount = (dim > 1024 ? 1 : (dim > 512 ? 2 : 4));
        TVector<TCudaLayerPools> layerPoolArr;
        layerPoolArr.resize(layerPoolCount);
        for (yint k = 0; k < layerPoolCount; ++k) {
            layerPoolArr[k].CreateNonshared(CudaMem, PoolLayers);
        }

        //
        ModelOps->InitDevice(DeviceId, Stream, CudaMem);
        TVector<TIntrusivePtr<TLayerBase>> &allLayers = modelStorage->GetLayers();
        yint depth = YSize(allLayers);
        yint maxStepId = depth + 100; // upper cap
        //
        if (!state0.Get()) {
            state0 = new TStateBuffer(MultiBuffers, 0, isPipelineParallel);
            state1 = new TStateBuffer(MultiBuffers, 1, isPipelineParallel);
        }
        State[0] = state0;
        State[0]->AllocateCuda(DeviceId, dim, maxLen);
        if (isPipelineParallel || microBatchCount > 1) {
            State[1] = state1;
            State[1]->AllocateCuda(DeviceId, dim, maxLen);
        }
        StateGradFast.AllocateCuda(dim, maxLen);
        GradScale.AllocateCuda(MultiBuffers, DeviceId, maxStepId);
        DNormState = MultiBuffers->Fab().CreateFRed2DArray<float>("DNormState");
        DNormState->AllocateCuda(DeviceId, dimFull, maxLen, PoolLayers);

        // model buffers
        TVector<bool> useStateBuf;
        ClearPodArray(&useStateBuf, depth + 1);
        if (MSplit.LocEmbed.Loc == LocationId) {
            Embedding = modelStorage->GetEmbedding()->CreateCudaMatrix(deviceId);
            EmbedLayer.AllocateCuda(PoolEmbed, microBatchCount, dim, maxLen, true);
            useStateBuf[0] = true;
        }
        LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            if (MSplit.GetLayerLoc(d).Loc == LocationId) {
                useStateBuf[d] = true;
                useStateBuf[d + 1] = true; // actually not used, room for mem save
                TCudaLayerPools pools;
                pools.Create(CudaMem, layerPoolArr[d % layerPoolCount]);
                TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr = allLayers[d]->GetMatrArr();
                TVector<TIntrusivePtr<ICudaModelMatrixBase<TFastModelFloat>>> cudaMatrArr;
                yint count = YSize(matrArr);
                cudaMatrArr.resize(count);
                for (yint k = 0; k < count; ++k) {
                    cudaMatrArr[k] = matrArr[k]->CreateCudaMatrix(deviceId);
                }
                LayerArr[d] = allLayers[d]->CreateCudaLayer(CudaMem, MultiBuffers, Stream, DeviceId, maxLen, cudaMatrArr, pools);
            }
        }
        FinalLayer.Init(microBatchCount, maxLen);
        if (MSplit.LocFinal.Loc == LocationId) {
            useStateBuf[depth] = true;
            Final = modelStorage->GetFinalLayer()->CreateCudaMatrix(deviceId);
            FinalLayer.AllocateCuda(PoolFinal, Final);
            Bias = modelStorage->GetBias().Get();
        }

        // init micro batches
        Y_VERIFY(microBatchCount >= 1);
        MBatchArr.resize(microBatchCount);
        for (yint mbId = 0; mbId < microBatchCount; ++mbId) {
            MBatchArr[mbId] = new TMicroBatch;
            TMicroBatch &mb = *MBatchArr[mbId];
            mb.AllStates.resize(depth + 1);
            for (yint k = 0; k < YSize(mb.AllStates); ++k) {
                if (useStateBuf[k]) {
                    mb.AllStates[k] = new TFragmentStates;
                    mb.AllStates[k]->AllocateCuda(MultiBuffers, Sprintf("mb[%g], state[%g]", mbId * 1., k * 1.), deviceId, dimFull, maxLen);
                }
            }
            FinalLayer.SetFinalStates(mbId, mb.AllStates.back());

            // compute params
            mb.ComputeParams.Allocate(ModelDescr, maxLen);
            mb.ComputeParams.InitRope(Stream, ModelDescr, maxLen);
            yint attentionTypeCount = ModelDescr.GetAttentionTypeCount();
            mb.ComputeParams.AttGDArr.resize(attentionTypeCount);
            for (yint wa = 0; wa < attentionTypeCount; ++wa) {
                mb.ComputeParams.AttGDArr[wa] = new TAttentionGroupData<ATT_GROUP>();
                mb.ComputeParams.AttGDArr[wa]->Allocate(maxLen, 4); // upper cap
            }
        }

        // create compute graphs
        CudaMem->AllocateMemory();
    }


private:
    TIntrusivePtr<TGraph> MakeGraph(TPtrArg<TGraph> mergedGraph, yint reserveSMcount)
    {
        ++GraphId;
        if (mergedGraph.Get()) {
            return mergedGraph.Get();
        }
        TIntrusivePtr<TGraph> c = new TGraph;
        c->ReserveSM(reserveSMcount);
        return c;
    }

    template <class T>
    TCuda2DArrayFragment<T> XSplitWindow(TCuda2DArray<T> &data)
    {
        return CalcXSplitWindow(data, data.GetXSize(), DeviceId, MultiBuffers->GetDeviceGroup());
    }

    TString GetStateCpName(const char *name, yint devId, yint d, yint mbId) const
    {
        return Sprintf("graph[%g]-%s-dev[%g]-layer[%g]-mb[%g]", GraphId * 1., name, devId * 1., d * 1., mbId * 1.);
    }

    // create compute graphs
    TIntrusivePtr<TGraph> MakeCopyModelToDevice(TPtrArg<TGraph> mergedGraph)
    {
        TIntrusivePtr<TGraph> c = MakeGraph(mergedGraph, 0);
        if (MSplit.LocEmbed.Loc == LocationId) {
            Embedding->CopyToDevice(c);
        }
        if (MSplit.LocFinal.Loc == LocationId) {
            Final->CopyToDevice(c);
        }
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            if (MSplit.GetLayerLoc(d).Loc == LocationId) {
                LayerArr[d]->CopyToDevice(c);
            }
        }
        c->FinishDependencyChain();
        return mergedGraph.Get() ? nullptr : c;
    }

    void AddForwardGraph(TPtrArg<TGraph> c, TPtrArg<TStateBuffer> stateBuf, yint mbId, yint passId, bool doCopyModel)
    {
        TMicroBatch &mb = *MBatchArr[mbId];
        TComputeParams *pParams = &mb.ComputeParams;
        TCuda2DArray<float> &state = stateBuf->Get(DeviceId);
        TModelLayerLocation myLocation(passId, LocationId);

        if (MSplit.LocEmbed == myLocation) {
            if (doCopyModel) {
                Embedding->CopyToDevice(c);
            }
            c->SetMemPool(PoolEmbed);
            EmbedLayer.AddForwardGraph(c, mbId, pParams, PtrArg(Embedding), &state);
        }
        TModelLayerLocation prevLocation = MSplit.LocEmbed;

        bool needNormalize = true;
        Y_ASSERT(YSize(LayerArr) == YSize(ModelDescr.LayerArr));
        yint depth = YSize(LayerArr);
        // iterate depth + 1 times to perform copy data and normalization for final layer as well
        for (yint d = 0; d <= depth; ++d) {
            TModelLayerLocation layerLocation = (d < depth) ? MSplit.GetLayerLoc(d) : MSplit.LocFinal;
            if (layerLocation == myLocation) {
                if (prevLocation != myLocation) {
                    yint prevDeviceId = LocationDeviceId[prevLocation.Loc];
                    stateBuf->RecvData(c, GetStateCpName("fwd", DeviceId, d, mbId), DeviceId, prevDeviceId);
                    needNormalize = true;
                }
                if (needNormalize) {
                    CudaCall(c, LayerNormalizeStateVecs<TStateFloat, TNormStateFloat>)
                        .Grid(MMTiles(state.GetXSize()), pParams->LenRound)
                        .Read(pParams->Len, STATE_VEC_SCALE, state)
                        .Write(&mb.AllStates[d]->GetLocal(), &mb.AllStates[d]->GetScaleLocal());
                    mb.AllStates[d]->Sync(c, MultiBuffers);
                    needNormalize = false;
                }
                if (d < depth) {
                    if (doCopyModel) {
                        LayerArr[d]->CopyToDevice(c);
                    }
                    LayerArr[d]->AddForward(c, pParams, mb.AllStates[d]->GetFull(), &state, mb.AllStates[d + 1].Get());
                    mb.AllStates[d + 1]->Sync(c, MultiBuffers);
                }
            } else if (prevLocation == myLocation) {
                yint nextDeviceId = LocationDeviceId[layerLocation.Loc];
                stateBuf->CopyData(c, GetStateCpName("fwd", nextDeviceId, d, mbId), pParams->Len, DeviceId, nextDeviceId);
            }
            prevLocation = layerLocation;
        }
    }

    TIntrusivePtr<TGraph> CreateForwardGraph(bool doCopyModel, TPtrArg<TGraph> mergedGraph)
    {
        TIntrusivePtr<TGraph> c = MakeGraph(mergedGraph, 1);
        ModelOps->InitFwdPass(DeviceId, c, doCopyModel);
        yint microBatchId = 0;
        for (yint passId = 0; passId < MSplit.GetPassCount(); ++passId) {
            AddForwardGraph(c, State[0], microBatchId, passId, doCopyModel);
        }
        if (doCopyModel) {
            Final->CopyToDevice(c);
        }
        c->FinishDependencyChain();
        return mergedGraph.Get() ? nullptr : c;
    }

    TIntrusivePtr<TGraph> CreateBackpropGraph(EBackpropMode bmArg, bool doCopyModelArg, TPtrArg<TGraph> mergedGraph)
    {
        yint stateXSize = StateGradFast.GetXSize();
        int stateTiles = MMTiles(stateXSize);
        yint dgRank = MultiBuffers->GetDeviceGroup().Rank(DeviceId);
        yint rankXOffset = dgRank * stateXSize;

        // background sm count depends on config
        yint reserveSMcount = IsHostMasterModel ? 2 : BG_SM_COUNT; // pcie read/write or background nvlink
        TIntrusivePtr<TGraph> c = MakeGraph(mergedGraph, reserveSMcount);

        // new flag value (iterCounter) for deltas
        ModelOps->InitFwdPass(DeviceId, c, doCopyModelArg);
        ModelOps->InitBwdPass(DeviceId, c);

        // all forward passes
        for (yint passId = 0; passId < MSplit.GetPassCount(); ++passId) {
            for (yint mbId = 0; mbId < MSplit.MicroBatchCount; ++mbId) {
                bool doCopyModel = (mbId == 0) ? doCopyModelArg : false;
                AddForwardGraph(c, State[mbId & 1], mbId, passId, doCopyModel);
            }
        }
        if (doCopyModelArg) {
            Final->CopyToDevice(c);
        }

        // all data from embedding matrix is read, can modify it (this can be executed much earlier)
        if (Embedding.Get()) {
            Embedding->AllowDelayedUpdates(c);
        }

        // all bwd passes
        for (yint passId = MSplit.GetPassCount() - 1; passId >= 0; --passId) {
            TModelLayerLocation myLocation(passId, LocationId);
            for (yint mbId = 0; mbId < MSplit.MicroBatchCount; ++mbId) {
                TMicroBatch &mb = *MBatchArr[mbId];
                TComputeParams *pParams = &mb.ComputeParams;
                EBackpropMode bm = BM_NONE;
                if (bmArg & BM_GRAD_ADD) {
                    bm = SetFlag(bm, BM_GRAD_ADD);
                } else {
                    if (mbId != 0) {
                        bm = SetFlag(bm, BM_GRAD_ADD);
                    }
                }
                if (bmArg & BM_GRAD_APPLY) {
                    if (mbId == MSplit.MicroBatchCount - 1) {
                        bm = SetFlag(bm, BM_GRAD_APPLY);
                    }
                }

                int stepId = 0;
                TIntrusivePtr<TStateBuffer> stateBuf = State[mbId & 1];
                auto &stateGrad = stateBuf->Get(DeviceId);
                GradScale.ClearMaxNorm(c);
                Y_ASSERT(stateGrad.GetXSize() == stateXSize);

                if (MSplit.LocFinal == myLocation) {
                    FinalLayer.AddBackprop(c, mbId, bm, stateGrad, rankXOffset);

                    TCudaPOD<float> gradMaxNorm = GradScale.GetGradMaxNorm(stepId);
                    TFragmentStates &normStates = *mb.AllStates.back();
                    CudaCall(c, BackpropFinalNormalize<TNormStateFloat, TStateFloat, TFastGradientFloat>)
                        .Grid(stateTiles, pParams->LenRound)
                        .Read(pParams->Len)
                        .Read(normStates.GetLocal(), normStates.GetScaleLocal(), stateGrad)
                        .Write(&stateGrad, &StateGradFast, &gradMaxNorm);
                    GradScale.SyncMax(c, MultiBuffers);
                }
                TModelLayerLocation prevLocation = MSplit.LocFinal;

                // modify layers
                for (yint d = YSize(LayerArr) - 1; d >= -1; --d, ++stepId) {
                    TCudaPOD<float> prevGradMaxNorm = GradScale.GetGradMaxNorm(stepId);
                    TCudaPOD<float> prevGradMult = GradScale.GetGradMult(stepId);
                    TCudaPOD<float> gradMaxNorm = GradScale.GetGradMaxNorm(stepId + 1);
                    TCudaPOD<float> gradScale = GradScale.GetGradScale(stepId + 1);
                    TCudaPOD<float> gradMult = GradScale.GetGradMult(stepId + 1);

                    TModelLayerLocation layerLocation = (d >= 0) ? MSplit.GetLayerLoc(d) : MSplit.LocEmbed;
                    if (layerLocation == myLocation) {
                        // receive state grad
                        if (prevLocation != myLocation) {
                            yint prevDeviceId = LocationDeviceId[prevLocation.Loc];
                            stateBuf->RecvData(c, GetStateCpName("bwd", DeviceId, d, mbId), DeviceId, prevDeviceId);

                            CudaCall(c, RecvGradKernel)
                                .FullGrid(c)
                                .Read(stateXSize, pParams->Len, stateGrad)
                                .Write(&prevGradMaxNorm, &prevGradMult);
                            GradScale.SyncMax(c, MultiBuffers);
                        }

                        // backprop layers or embed
                        if (d >= 0) {
                        // udpdate scale
                            CudaCall(c, ScaleGrad)
                                .Grid(stateTiles, pParams->Len)
                                .Read((int)d, prevGradMaxNorm, prevGradMult)
                                .Write(&stateGrad, &StateGradFast, &gradScale, &gradMult);

                            // backprop layer
                            TFragmentStates &normStates = *mb.AllStates[d];
                            TCudaPOD<float> combinerScale =
                                LayerArr[d]->AddBackward(c, pParams, normStates.GetFull(), &StateGradFast, gradScale, DNormState);

                            // add state gradient
                            auto dNormState = XSplitWindow(DNormState->GetData(DeviceId));
                            CudaCall(c, BackpropLayerNormalize<TNormStateFloat, TStateFloat, TFastGradientFloat>)
                                .Grid(stateTiles, pParams->Len)
                                .Read(normStates.GetLocal(), normStates.GetScaleLocal(), dNormState)
                                .Read(combinerScale, gradScale, gradMult)
                                .Write(&stateGrad, &StateGradFast, &gradMaxNorm);
                            GradScale.SyncMax(c, MultiBuffers);

                        } else {
                            // backprop embed
                            if (!ModelDescr.HasFlag(MPF_DISABLE_TUNE_EMBED)) {
                                EmbedLayer.AddBackprop(c, mbId, PtrArg(Embedding), stateGrad);
                            }
                        }
                    } else if (prevLocation == myLocation) {
                        yint nextDeviceId = LocationDeviceId[layerLocation.Loc];
                        stateBuf->CopyData(c, GetStateCpName("bwd", nextDeviceId, d, mbId), pParams->Len, DeviceId, nextDeviceId);
                    }
                    prevLocation = layerLocation;
                }

                // add delta
                for (yint d = YSize(LayerArr) - 1; d >= -1; --d, ++stepId) {
                    TModelLayerLocation layerLocation = (d >= 0) ? MSplit.GetLayerLoc(d) : MSplit.LocEmbed;
                    if (layerLocation == myLocation) {
                        if (d >= 0) {
                            LayerArr[d]->AddDelta(c, bm);
                        } else {
                            if (!ModelDescr.HasFlag(MPF_DISABLE_TUNE_EMBED)) {
                                EmbedLayer.AddDelta(c, PtrArg(Embedding), bm);
                            }
                        }
                    }
                }
            }
        }
        c->FinishDependencyChain();
        return mergedGraph.Get() ? nullptr : c;
    }

    void RunGraph(TPtrArg<TGraph> c)
    {
        if (c.Get()) {
            c->Run(Stream);
        }
    }

public:
    void CreateGraphs(TPtrArg<TSinlgeComputeContext> rootInstance, TPtrArg<TSinlgeComputeContext> rootHost)
    {
        TIntrusivePtr<TSinlgeComputeContext> rootCopyParams = rootInstance.Get();
        TIntrusivePtr<TSinlgeComputeContext> rootFwd = rootInstance.Get();
        TIntrusivePtr<TSinlgeComputeContext> rootBwd = rootHost.Get();
        ForwardComputer = CreateForwardGraph(false, rootFwd->ForwardComputer);
        if (IsHostMasterModel) {
            CMForwardComputer = CreateForwardGraph(true, rootFwd->CMForwardComputer);
        }
        CopyModelToDeviceGraph = MakeCopyModelToDevice(rootCopyParams->CopyModelToDeviceGraph);
        yint bmCount = ModelOps->GetBackpropModeCount();
        BackpropComputerArr.resize(bmCount);
        CMBackpropComputerArr = BackpropComputerArr;
        for (yint bm = 0; bm < bmCount; ++bm) {
            BackpropComputerArr[bm] = CreateBackpropGraph(EBackpropMode(bm), false, rootBwd->BackpropComputerArr[bm]);
            if (IsHostMasterModel) {
                CMBackpropComputerArr[bm] =
                    CreateBackpropGraph(EBackpropMode(bm), true, rootBwd->CMBackpropComputerArr[bm]);
            }
        }
        FinalLayer.CreateGraphs();
    }

    void WaitCudaCompute()
    {
        if (ComputeInFly) {
            Stream.Sync();
            ComputeInFly = false;
        }
    }

    const TModelDescr &GetLocalModelDescr() { return ModelDescr; }

    void CopyModelParamsToDevice()
    {
        FinalLayer.CopyModelParams(Stream, Bias->Vec);
        if (IsHostMasterModel) {
            NeedCopyToDevice = true;
        } else {
            RunGraph(CopyModelToDeviceGraph);
            NeedCopyToDevice = false;
        }
        Stream.Sync();
    }

    void Init(TVector<TMicroBatchCtx> mCtxArr, IModelInstance::EInitType initType)
    {
        bool isBackprop = (initType == IModelInstance::INIT_BACKPROP);
        yint microBatchCount = isBackprop ? MSplit.MicroBatchCount : 1;
        yint attentionTypeCount = ModelDescr.GetAttentionTypeCount();
        for (yint mbId = 0; mbId < microBatchCount; ++mbId) {
            TMicroBatch &mb = *MBatchArr[mbId];
            TMicroBatchCtx &mbCtx = mCtxArr[mbId];
            yint len = mbCtx.Nodes.GetNodeCount();
            bool isCross = (mb.ComputeParams.CrossAttnShuffle.GetSize() > 0);
            mb.AttGroupsArr.resize(attentionTypeCount);
            mb.RevAttGroupsArr.resize(attentionTypeCount);
            for (yint wa = 0; wa < attentionTypeCount; ++wa) {
                GroupAttention<ATT_GROUP, ATT_ALIGN>(mbCtx.Nodes.AttArr[wa], &mb.AttGroupsArr[wa]);
                GroupAttention<ATT_GROUP, ATT_ALIGN>(TransposeAttention(mbCtx.Nodes.AttArr[wa]), &mb.RevAttGroupsArr[wa]);
            }
            if (isCross) {
                PutHost(&mb.ComputeParams.CrossAttnShuffle, mbCtx.Nodes.CrossShuffle.FwdShuffle);
            }

            EmbedLayer.BuildIndex(mbId, mbCtx.Nodes.Labels, isBackprop);
        }

        WaitCudaCompute();
        for (yint mbId = 0; mbId < microBatchCount; ++mbId) {
            TMicroBatch &mb = *MBatchArr[mbId];
            TMicroBatchCtx &mbCtx = mCtxArr[mbId];
            yint len = mbCtx.Nodes.GetNodeCount();
            bool isCross = (mb.ComputeParams.CrossAttnShuffle.GetSize() > 0);
            EmbedLayer.Init(Stream, mbId, mbCtx.SampleEmbedVectors, isBackprop);
            FinalLayer.Init(Stream, mbId, len, mbCtx.Nodes.Target, isBackprop);

            mb.ComputeParams.Init(Stream, len);
            for (yint wa = 0; wa < attentionTypeCount; ++wa) {
                mb.ComputeParams.AttGDArr[wa]->Init(Stream, len, &mb.AttGroupsArr[wa], &mb.RevAttGroupsArr[wa]);
            }
            if (isCross) {
                mb.ComputeParams.CrossAttnShuffle.CopyToDevice(Stream, len);
            }
        }
    }

    void RunForward()
    {
        if (NeedCopyToDevice) {
            RunGraph(CMForwardComputer);
        } else {
            RunGraph(ForwardComputer);
        }
        NeedCopyToDevice = false;
        ComputeInFly = true;
    }

    void RunBackprop(EBackpropMode bm)
    {
        if (NeedCopyToDevice) {
            RunGraph(CMBackpropComputerArr[bm]);
        } else {
            RunGraph(BackpropComputerArr[bm]);
        }
        NeedCopyToDevice = IsHostMasterModel;
        ComputeInFly = true;
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction)
    {
        yint microBatchId = 0;
        TMicroBatch &mb = *MBatchArr[microBatchId];
        FinalLayer.ComputeFragmentPredictions(Stream, mb.ComputeParams.Len.Get(), pPrediction);
    }

    void ComputeFragmentPredictions(TVector<float> *pPrediction)
    {
        yint microBatchId = 0;
        TMicroBatch &mb = *MBatchArr[microBatchId];
        FinalLayer.ComputeFragmentPredictions(Stream, mb.ComputeParams.Len.Get(), pPrediction);
    }

    float ComputeScore() { return FinalLayer.ComputeScore(Stream); }
    float GetAvrgTrainErr() { return FinalLayer.GetAvrgTrainErr(Stream); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// multi GPU support (for backprop only so far)
class TMultiComputeContext : public IModelImpl
{
    enum EJob
    {
        JOB_WAIT_CUDA_COMPUTE,
        JOB_COPY_MODEL_PARAMS_TO_DEVICE,
        JOB_INIT_BACKPROP,
        JOB_INIT_FWD,
        JOB_BACKPROP_INIT_PARAMS,
        JOB_BACKPROP_RUN,
        JOB_GET_AVRG_TRAIN_ERR,
        JOB_RUN_FORWARD,
    };

    struct TModelInstanceCtx : public TThrRefBase
    {
        TVector<TMicroBatchCtx> MicroBatchArr;

        TModelInstanceCtx(yint microBatchCount) { MicroBatchArr.resize(microBatchCount); }
    };

    struct TDeviceControlThread : public TThrRefBase
    {
        TThread Worker;
        TSingleConsumerJobQueue<EJob> JobQueue;
        std::atomic<int> JobQueueSize;
        TIntrusivePtr<TModelInstanceCtx> MCtx;
        TIntrusivePtr<TSinlgeComputeContext> Ctx;
        IModelInstance::EInitType InitType = IModelInstance::INIT_BACKPROP;
        float AvrgTrainErr = 0;

        TDeviceControlThread(TPtrArg<TModelInstanceCtx> mCtx) : JobQueueSize(0), MCtx(mCtx) {}
        
        void Run(TMultiComputeContext *pThis)
        {
            Worker.Create(pThis);
        }
        
        void AddOp(EJob op)
        {
            JobQueueSize.fetch_add(1);
            JobQueue.Enqueue(op);
        }
        
        void WaitDevice()
        {
            while (JobQueueSize.load() > 0) {
                _mm_pause();
            }
        }
    };


private:
    yint DeviceCount = 0;
    yint TP = 0;
    yint PP = 0;
    yint FinalLayerLoc = 0;
    bool IsHostMasterModel = false;
    TIntrusivePtr<IModelOps> ModelOps;
    bool ModelDeltaInFly = false;
    std::atomic<int> WorkerId;
    TVector<TIntrusivePtr<TDeviceControlThread>> ThrArr;
    volatile bool Exit = false;
    volatile EBackpropMode BackpropMode = BM_NONE;


public:
    void WorkerThread()
    {
        yint deviceId = WorkerId.fetch_add(1);
        SetDevice(deviceId);
        TDeviceControlThread *thr = ThrArr[deviceId].Get();
        while (!Exit) {
            EJob job;
            if (thr->JobQueue.DequeueFirst(&job)) {
                TSinlgeComputeContext *ctx = thr->Ctx.Get();
                switch (job) {
                case JOB_WAIT_CUDA_COMPUTE:
                    ctx->WaitCudaCompute();
                    break;
                case JOB_COPY_MODEL_PARAMS_TO_DEVICE:
                    ctx->CopyModelParamsToDevice();
                    break;
                case JOB_INIT_BACKPROP:
                    ctx->Init(thr->MCtx->MicroBatchArr, IModelInstance::INIT_BACKPROP);
                    break;
                case JOB_INIT_FWD:
                    ctx->Init(thr->MCtx->MicroBatchArr, IModelInstance::INIT_FWD);
                    break;
                case JOB_BACKPROP_RUN:
                    ctx->RunBackprop(BackpropMode);
                    break;
                case JOB_GET_AVRG_TRAIN_ERR:
                    thr->AvrgTrainErr = ctx->GetAvrgTrainErr();
                    break;
                case JOB_RUN_FORWARD:
                    ctx->RunForward();
                    break;
                }
                thr->JobQueueSize.fetch_add(-1);
            } else {
                _mm_pause();
            }
        }
    }

private:
    void SetDevice(yint deviceId) const
    {
        if (DeviceCount > 1) {
            CudaSetDevice(deviceId);
        }
    }

    void ForeachDevice(EJob func)
    {
        for (yint deviceId = 0; deviceId < YSize(ThrArr); ++deviceId) {
            ThrArr[deviceId]->AddOp(func);
        }
    }

    void WaitDevices()
    {
        for (yint deviceId = 0; deviceId < YSize(ThrArr); ++deviceId) {
            ThrArr[deviceId]->WaitDevice();
        }
    }

    void WaitActiveCompute()
    {
        // correct order is to wait gpu graph completion first, then wait cpu ops (gpu graphs launch cpu compute)
        ForeachDevice(JOB_WAIT_CUDA_COMPUTE);
        WaitDevices();
        ModelOps->WaitActiveCompute();
    }

private:
    // IModelInstance implementation

    void WaitInstanceOpLaunch(yint modelInstanceId)
    {
        yint miSize = TP * PP;
        for (yint k = 0; k < miSize; ++k) {
            ThrArr[modelInstanceId * miSize + k]->WaitDevice();
        }
    }

    void Init(yint modelInstanceId, IModelInstance::EInitType initType)
    {
        if (DeviceCount > 1 && SIMULATE_MULTI_GPU) {
            // ctx->Init() performs cuda wait, but for sim multi gpu mode we should wait all gpus
            ForeachDevice(JOB_WAIT_CUDA_COMPUTE);
            WaitDevices();
        }
        EJob job = (initType == IModelInstance::INIT_BACKPROP) ? JOB_INIT_BACKPROP : JOB_INIT_FWD;
        yint miSize = TP * PP;
        for (yint k = 0; k < miSize; ++k) {
            ThrArr[modelInstanceId * miSize + k]->AddOp(job);
        }
    }

    void RunForward(yint modelInstanceId)
    {
        if (ModelDeltaInFly) {
            WaitAllCompute();
        }
        yint miSize = TP * PP;
        if (miSize > 1 && SIMULATE_MULTI_GPU) {
            // wait all Init()
            for (yint k = 0; k < miSize; ++k) {
                ThrArr[modelInstanceId * miSize + k]->AddOp(JOB_WAIT_CUDA_COMPUTE);
                ThrArr[modelInstanceId * miSize + k]->WaitDevice();
            }
        }
        for (yint k = 0; k < miSize; ++k) {
            ThrArr[modelInstanceId * miSize + k]->AddOp(JOB_RUN_FORWARD);
        }
        for (yint k = 0; k < miSize; ++k) {
            if (miSize > 1 && SIMULATE_MULTI_GPU) {
                // wait merged graph
                ThrArr[modelInstanceId * miSize + k]->AddOp(JOB_WAIT_CUDA_COMPUTE);
            }
            ThrArr[modelInstanceId * miSize + k]->WaitDevice();
        }
    }

    void ComputeFragmentPredictions(yint modelInstanceId, TVector<TVector<float>> *pPrediction)
    {
        RunForward(modelInstanceId);
        yint finalLayerDeviceId = FinalLayerLoc * TP + modelInstanceId * TP * PP;
        ThrArr[finalLayerDeviceId]->Ctx->ComputeFragmentPredictions(pPrediction);
    }

    void ComputeFragmentPredictions(yint modelInstanceId, TVector<float> *pPrediction)
    {
        RunForward(modelInstanceId);
        yint finalLayerDeviceId = FinalLayerLoc * TP + modelInstanceId * TP * PP;
        ThrArr[finalLayerDeviceId]->Ctx->ComputeFragmentPredictions(pPrediction);
    }

    float ComputeScore(yint modelInstanceId)
    {
        RunForward(modelInstanceId);
        yint finalLayerDeviceId = FinalLayerLoc * TP + modelInstanceId * TP * PP;
        return ThrArr[finalLayerDeviceId]->Ctx->ComputeScore();
    }


private:
    struct TModelInstance : public IModelInstance
    {
        yint ModelInstanceId = 0;
        TIntrusivePtr<TMultiComputeContext> Parent;
        TIntrusivePtr<TModelInstanceCtx> MCtx;

        TModelInstance(yint modelInstanceId, TMultiComputeContext *parent, TPtrArg<TModelInstanceCtx> mCtx)
            : ModelInstanceId(modelInstanceId), Parent(parent), MCtx(mCtx)
        {
        }
        const TModelDescr &GetModelDescr() override { return Parent->ThrArr[0]->Ctx->GetLocalModelDescr(); }
        void WaitInstanceOpLaunch() override
        {
            Parent->WaitInstanceOpLaunch(ModelInstanceId);
        }
        TBatchNodes &GetNodes(yint microBatchId) override
        {
            WaitInstanceOpLaunch();
            return MCtx->MicroBatchArr[microBatchId].Nodes;
        }
        TArray2D<float> &GetSampleEmbedVectors(yint microBatchId) override
        {
            WaitInstanceOpLaunch();
            return MCtx->MicroBatchArr[microBatchId].SampleEmbedVectors;
        }
        void Init(EInitType initType) override { Parent->Init(ModelInstanceId, initType); }
        void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) override
        {
            Parent->ComputeFragmentPredictions(ModelInstanceId, pPrediction);
        }
        void ComputeFragmentPredictions(TVector<float> *pPrediction) override
        {
            Parent->ComputeFragmentPredictions(ModelInstanceId, pPrediction);
        }
        float ComputeScore() override { return Parent->ComputeScore(ModelInstanceId); }
    };


private:
    // IModelImpl implementation

    TIntrusivePtr<IModelInstance> GetInstance(yint modelInstanceId)
    {
        return new TModelInstance(modelInstanceId, this, ThrArr[modelInstanceId * TP * PP]->MCtx);
    }

    void Backprop(const TTrainingStep &step, EAddToModel addToModel) override
    {
        WaitActiveCompute(); // modify cuda graphs when cuda queue is empty, wait non delayed updates
        // no more then one queued backprop is possible, so using global BackpropMode is fine
        BackpropMode = ModelOps->StartIteration(step, addToModel); // no pending matrix ops at this point expected
        ForeachDevice(JOB_BACKPROP_RUN);
        ModelDeltaInFly = true;
    }

    void WaitAllCompute() override
    {
        WaitDevices();
        if (ModelDeltaInFly) {
            WaitActiveCompute();
            ModelOps->WaitDelayedCompute();
            ModelDeltaInFly = false;
        }
    }

    void CopyModelParamsToDevice() override
    {
        WaitAllCompute();
        ForeachDevice(JOB_COPY_MODEL_PARAMS_TO_DEVICE);
        WaitDevices();
    }

    float GetAvrgTrainErr() override
    {
        if (SIMULATE_MULTI_GPU) {
            ForeachDevice(JOB_WAIT_CUDA_COMPUTE);
            WaitDevices();
        }
        ForeachDevice(JOB_GET_AVRG_TRAIN_ERR);
        WaitDevices();
        yint deviceCount = YSize(ThrArr);
        float sum = 0;
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            sum += ThrArr[deviceId]->AvrgTrainErr;
        }
        return sum / (deviceCount / PP);
    }

    ~TMultiComputeContext()
    {
        Exit = true;
    }

public:
    TMultiComputeContext(TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, const TModelSplit &msplit, yint nodeCount,
        ETargetType targetType, TPtrArg<ICustomLoss> pLoss)
        : ModelOps(modelOps), WorkerId(0)
    {
        IsHostMasterModel = ModelOps->IsHostMasterModel();
        DeviceCount = ModelOps->GetDeviceCount();
        TP = msplit.TP;
        PP = msplit.PP;
        FinalLayerLoc = msplit.LocFinal.Loc;
        yint modelInstanceCount = DeviceCount / TP / PP;
        Y_ASSERT(ModelOps->GetDeviceGroupCount() * TP == DeviceCount);

        TVector<TIntrusivePtr<TModelInstanceCtx>> modelInstanceCtxArr;
        modelInstanceCtxArr.resize(modelInstanceCount);
        for (yint k = 0; k < modelInstanceCount; ++k) {
            modelInstanceCtxArr[k] = new TModelInstanceCtx(msplit.MicroBatchCount);
        }

        TVector<TIntrusivePtr<TSinlgeComputeContext>> ctxArr;
        ctxArr.resize(DeviceCount);
        TIntrusivePtr<TStateBuffer> state0;
        TIntrusivePtr<TStateBuffer> state1;
        for (yint deviceId = 0; deviceId < DeviceCount; ++deviceId) {
            yint deviceGroupId = deviceId / TP;
            yint deviceGroupOffset = deviceId % TP;
            yint modelInstanceId = deviceId / TP / PP;
            yint locationId = deviceGroupId % PP;
            TVector<yint> locationDeviceId;
            for (yint k = 0; k < PP; ++k) {
                locationDeviceId.push_back(modelInstanceId * TP * PP + k * TP + deviceGroupOffset);
            }
            SetDevice(deviceId);
            ctxArr[deviceId] = new TSinlgeComputeContext(deviceId, deviceGroupId, locationId, locationDeviceId, msplit, modelStorage,
                modelOps, nodeCount, IsHostMasterModel, targetType, pLoss, state0, state1);
        }
        // cross gpu kernels require all allocations complete
        for (yint deviceId = 0; deviceId < DeviceCount; ++deviceId) {
            SetDevice(deviceId);
            TIntrusivePtr<TSinlgeComputeContext> rootInstance = ctxArr[deviceId];
            TIntrusivePtr<TSinlgeComputeContext> rootHost = ctxArr[deviceId];
            if (SIMULATE_MULTI_GPU) {
                yint modelInstanceId = deviceId / TP / PP;
                rootInstance = ctxArr[modelInstanceId * TP * PP];
                rootHost = ctxArr[0];
            }
            ctxArr[deviceId]->CreateGraphs(rootInstance, rootHost);
        }
        SetDevice(0);

        ThrArr.resize(DeviceCount);
        for (yint deviceId = 0; deviceId < DeviceCount; ++deviceId) {
            yint modelInstanceId = deviceId / TP / PP;
            TIntrusivePtr<TModelInstanceCtx> mCtx = modelInstanceCtxArr[modelInstanceId];
            ThrArr[deviceId] = new TDeviceControlThread(mCtx);
            ThrArr[deviceId]->Ctx = ctxArr[deviceId];
        }
        for (yint deviceId = 0; deviceId < DeviceCount; ++deviceId) {
            ThrArr[deviceId]->Run(this);
        }

        // assign model params
        CopyModelParamsToDevice();
    }
};

TIntrusivePtr<IModelImpl> CreateContext(
    TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, const TModelSplit &msplit, yint nodeCount)
{
    return new TMultiComputeContext(modelStorage, modelOps, msplit, nodeCount, TARGET_TOKEN, null_ptr_arg);
}

TIntrusivePtr<IModelImpl> CreateContextNoSoftmax(TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount)
{
    return new TMultiComputeContext(modelStorage, modelOps, TModelSplit(), nodeCount, TARGET_NO_SOFTMAX, null_ptr_arg);
}

TIntrusivePtr<IModelImpl> CreateWithCustomLoss(
    TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount, TPtrArg<ICustomLoss> pLoss)
{
    return new TMultiComputeContext(modelStorage, modelOps, TModelSplit(), nodeCount, TARGET_CUSTOM, pLoss);
}

}
