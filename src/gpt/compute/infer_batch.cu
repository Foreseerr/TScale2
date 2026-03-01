#include <util/pch.h>
#define KERNEL_UNIT "gpt_infer_batch/"
#include "infer_batch.cuh"
#include "cfg_precision.h"
#include "model.h"
#include "layer_cuda_base.cuh"
#include <gpt/matrix/infer_matrix_cuda.cuh>
#include "layer_ffn.cuh"
#include "layer_att.cuh"
#include "layer_att_3lin.cuh"
#include <gpt/att/rope.h>
#include <gpt/att/nodes_batch.h>
#include <gpt/att/sliding_window.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_arrays.h>
// kernels
#include "gpu_embedding.cuh"
#include "gpu_final.cuh"
#include "gpu_layer_norm.cuh"


using namespace NCuda;

namespace NCUDA_Transformer
{

__global__ void GumbelMaxSampler(int vocabSize, TCuda2DPtr<half> logitBuf, TCuda1DPtr<ui32> rngSeed, TCuda1DPtr<ui32> resArr)
{
    int t = blockIdx.x;
    int h = threadIdx.x;
    TCudaRngLCG rng(t, h, rngSeed[t]);

    float bestScore = -1e38f;
    ui32 res = 0;
    for (int base = 0; base < vocabSize; base += WARP_SIZE) {
        int x = base + h;
        if (x < vocabSize) {
            float logitVal = float(logitBuf[t][x]) * LOG2;
            float score = logitVal - log(-log(rng.GenRandReal3()));
            if (score > bestScore) {
                bestScore = score;
                res = x;
            }
        }
    }
    res = WarpMaxIdx(bestScore, res);
    if (h == 0) {
        resArr[t] = res;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void InitKVcachePtrs(TStream &stream, TInferAttentionData *pInferAtt, yint maxHistory, const TVector<TBatchInferRequest> &qArr)
{
    // assume number of queries is constant, otherwise
    TInferAttentionData &att = *pInferAtt;
    TVector<int> readArr;
    TVector<int> readArrPtr;
    TVector<int> writeArr;
    TVector<yint> ropeTimeArr;
    readArrPtr.push_back(0);
    for (const TBatchInferRequest &q : qArr) {
        const TKVcacheReference &kv = q.KVcache;
        Y_ASSERT(kv.KVwrite >= 0);
        for (int id : kv.KVrefs) {
            readArr.push_back(id);
        }
        readArrPtr.push_back(YSize(readArr));
        writeArr.push_back(kv.KVwrite);
        ropeTimeArr.push_back(kv.Time);
    }
    TArray2D<float> ropeBuf;
    FillRopeBuf(&ropeBuf, att.RopeBuf.GetXSize(), ropeTimeArr);

    Put(stream, &att.ReadArr, readArr);
    Put(stream, &att.ReadArrPtr, readArrPtr);
    Put(stream, &att.WriteArr, writeArr);
    Put(stream, &att.RopeBuf, ropeBuf);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TBatchInfer : public IBatchInfer
{
    TStream Stream;
    TIntrusivePtr<TCudaMemoryAllocator> CudaMem;
    TIntrusivePtr<TCudaMemoryPool> PoolEmbed;
    TIntrusivePtr<TCudaMemoryPool> PoolLayers;
    TIntrusivePtr<TCudaMemoryPool> PoolFinal;
    TModelDescr ModelDescr;
    TIntrusivePtr<THostModelMatrixScale> HostMatrixScale;
    TIntrusivePtr<TCudaModelMatrixScale> CudaMatrixScale;
    TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>> FinalLayer;
    TIntrusivePtr<TCudaInferModelMatrix<TEmbedFloat>> Embedding;
    TVector<TIntrusivePtr<TCudaInferLayerBase>> LayerArr;
    // model params
    TCudaVector<float> Bias;
    // compute buffers
    TKVcacheTracker KVcache;
    TEmbeddingLayer EmbedLayer;
    TCuda2DArray<TStateFloat> State; // after forward pass contains state after all layers applied
    TCuda2DArray<TNormStateFloat> NormState;
    TCuda2DArray<float> NormStateScale;
    TCuda2DArray<half> LogitBuf;
    TCuda2DArray<float> LogitBufRowTileLogSum;
    TCuda2DArray<half> LogitBufHost;
    TCudaVector<ui32> RngSeed;
    TCudaVector<ui32> ResTokens;

    TComputeParams ComputeParams;
    yint PrevLen = -1;

    TIntrusivePtr<TGraph> PredictComputer;
    TIntrusivePtr<TGraph> SampleComputer;

private:
    yint GetFinalLayerSizeRounded() const
    {
        return RoundUp(ModelDescr.OutputTokenCount, MM_TILE);
    }

    // create compute graphs
    void AddFinalLayer(TPtrArg<TGraph> c)
    {
        TModelDims &dims = ModelDescr.Dims;
        TComputeParams *pParams = &ComputeParams;
        int stateDim = dims.Dim;
        int flSize = GetFinalLayerSizeRounded();
        int outputCount = ModelDescr.OutputTokenCount;
        TCudaPOD<float> scaleFinalLayer = FinalLayer->GetScale();

        float normScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;
        float flScale = CalcFinalLayerMult() * normScale;
        int computeLogit = 0;
        TKernelOp *pFinalOp = 0;
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            // using half precision sum here impairs quality beyond repair
            pFinalOp = &MulForwardFp16<TStoreFinalLayerLogits>(c, pParams, stateDim, flSize, NormState, PtrArg(FinalLayer), &LogitBuf);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            pFinalOp = &MulForwardFp8<TStoreFinalLayerLogits>(c, pParams, stateDim, flSize, NormState, PtrArg(FinalLayer), &LogitBuf);
        } else if (FWD_MATMUL_TYPE == MATMUL_INT8) {
            pFinalOp = &MulForwardInt8<TStoreFinalLayerLogits>(c, pParams, stateDim, flSize, NormState, PtrArg(FinalLayer), &LogitBuf);
        }
        (*pFinalOp).Read(scaleFinalLayer, flScale, outputCount, Bias, computeLogit).Write(&LogitBufRowTileLogSum);
    }

    void AddForwardGraph(TPtrArg<TGraph> c)
    {
        TModelDims &dims = ModelDescr.Dims;
        TComputeParams *pParams = &ComputeParams;
        int dim = dims.Dim;

        c->SetMemPool(PoolEmbed);
        EmbedLayer.AddForwardGraph(c, 0, &ComputeParams, PtrArg(Embedding), &State);

        CudaCall(c, LayerNormalizeStateVecs<TStateFloat, TNormStateFloat>)
            .Grid(MMTiles(dim), pParams->LenRound)
            .Read(pParams->Len, STATE_VEC_SCALE, State)
            .Write(&NormState, &NormStateScale);

        // apply layers
        Y_ASSERT(YSize(LayerArr) == YSize(ModelDescr.LayerArr));
        c->SetMemPool(PoolLayers);
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            LayerArr[d]->AddForward(c, pParams, &State, &NormState, &NormStateScale);
        }

        c->SetMemPool(PoolFinal);
        AddFinalLayer(c);
    }

    void CreateGraph()
    {
        TIntrusivePtr<TGraph> c = new TGraph;
        AddForwardGraph(c);
        c->KernelCopy(&LogitBufHost, LogitBuf, ComputeParams.Len);
        PredictComputer = c;
        //
        c = new TGraph;
        AddForwardGraph(c);
        CudaCall(c, GumbelMaxSampler).Grid(ComputeParams.Len).Read((int)ModelDescr.OutputTokenCount, LogitBuf, RngSeed).Write(&ResTokens);
        c->KernelCopy(&LogitBufHost, LogitBuf, ComputeParams.Len);
        SampleComputer = c;
    }

public:
    TBatchInfer(const TModelParams &params, yint maxBatchSize) : EmbedLayer(params.ModelDescr), ModelDescr(params.ModelDescr)
    {
        // dimension restrictions
        yint dim = ModelDescr.Dims.Dim;
        Y_VERIFY((dim % WARP_SIZE) == 0); // some kernels process states with warps
        Y_VERIFY((dim % I8_TILE_GROUP_SIZE) == 0);
        Y_VERIFY(STATE_NORM_TILE == MM_TILE);
        // params
        yint maxLen = RoundUp(maxBatchSize, MM_TILE);
        yint finalLayerRoundSize = GetFinalLayerSizeRounded();
        EModelMatrixQuant quant = GetQuant(ModelDescr);
        yint depth = YSize(ModelDescr.LayerArr);
        yint maxHistory = ModelDescr.GetMaxAttentionHistory();
        yint kvCacheSize = maxLen * (maxHistory + 1);
        //
        CudaMem = new TCudaMemoryAllocator();
        PoolEmbed = CudaMem->CreatePool();
        PoolLayers = CudaMem->CreatePool();
        PoolFinal = CudaMem->CreatePool();
        //
        KVcache.Init(maxHistory, kvCacheSize);
        //
        HostMatrixScale = new THostModelMatrixScale(depth * MP_MAX_COUNT + MP_MODEL_COUNT);
        CudaMatrixScale = new TCudaModelMatrixScale(HostMatrixScale);
        LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            const TModelDescr::TLayerParams &layerDescr = ModelDescr.LayerArr[d];
            const TModelParams::TAttentionMatrices &modelParamsLayer = params.LayerArr[d];
            yint count = YSize(modelParamsLayer.MatrArr);
            TIntrusivePtr<TCudaMemoryPool> layerPool = CudaMem->CreatePool(PoolLayers);
            TAttentionType attnType = ModelDescr.AttentionTypeArr[layerDescr.AttentionTypeId];

            TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> matrArr;
            matrArr.resize(count);
            for (yint k = 0; k < count; ++k) {
                matrArr[k] = new TCudaInferModelMatrix<TFastModelFloat>(
                    Stream, HostMatrixScale, CudaMatrixScale, modelParamsLayer.MatrArr[k], quant, MM_MEM_DEVICE);
            }

            switch (layerDescr.LayerType) {
            case MLT_ATT:
                LayerArr[d] = CreateAttLayerInference(//
                    ModelDescr, layerDescr.AlibiSlope, attnType, maxLen, kvCacheSize, matrArr, layerPool);
                break;
            case MLT_ATT_3LIN:
                LayerArr[d] = Create3LinAttLayerInference(Stream, //
                    ModelDescr, layerDescr.AlibiSlope, attnType, maxLen, kvCacheSize, matrArr, layerPool);
                break;
            case MLT_FFN:
                LayerArr[d] = CreateFFLayerInference(ModelDescr, maxLen, matrArr, layerPool);
                break;
            default:
                Y_VERIFY(0 && "unsupported layer type");
            }
        }
        Embedding = new TCudaInferModelMatrix<TEmbedFloat>(
            Stream, HostMatrixScale, CudaMatrixScale, params.MatrArr[MP_MODEL_EMBED], quant, MM_MEM_HOST);
        FinalLayer = new TCudaInferModelMatrix<TFastModelFloat>(
            Stream, HostMatrixScale, CudaMatrixScale, params.MatrArr[MP_MODEL_FINAL], quant, MM_MEM_DEVICE);
        Bias.Allocate(ModelDescr.OutputTokenCount);
        // initialize rest model params
        Put(Stream, &Bias, params.Bias);
        CudaMatrixScale->CopyToDevice(Stream);
        //
        bool needBackprop = false;
        EmbedLayer.AllocateCuda(PoolEmbed, 1, dim, maxLen, needBackprop);
        State.Allocate(dim, maxLen);
        NormState.AllocateCuda(dim, maxLen);
        NormStateScale.Allocate(maxLen, MMTiles(dim));
        LogitBuf.AllocateCuda(finalLayerRoundSize, maxLen, PoolFinal);
        LogitBufRowTileLogSum.AllocateCuda(maxLen, finalLayerRoundSize / MM_TILE, PoolFinal);
        LogitBufHost.AllocateHost(finalLayerRoundSize, maxLen);
        RngSeed.Allocate(maxLen);
        ResTokens.Allocate(maxLen);
        //
        // compute params & contexts
        ComputeParams.Allocate(ModelDescr, maxLen);
        ComputeParams.InferAtt = new TInferAttentionData();
        ComputeParams.InferAtt->Allocate(ModelDescr, maxLen, maxHistory);
        ComputeParams.InferCrossAtt = new TAttentionGroupData<ATT_GROUP>();
        ComputeParams.InferCrossAtt->Allocate(maxLen, 1);
        // create compute graphs
        CudaMem->AllocateMemory();
        CreateGraph();
    }


    void Init(const TVector<TBatchInferRequest> &qArr)
    {
        yint len = YSize(qArr);
        bool isTrain = false;

        TArray2D<float> sampleEmbedVectors;
        if (ModelDescr.HasFlag(MPF_SAMPLE_EMBED_VECTORS)) {
            yint dim = ModelDescr.Dims.Dim;
            sampleEmbedVectors.SetSizes(dim, len);
            for (yint t = 0; t < len; ++t) {
                for (yint x = 0; x < dim; ++x) {
                    sampleEmbedVectors[t][x] = qArr[t].EmbedVec[x];
                }
            }
        }

        TBatchLabels labels;
        labels.Init();
        for (yint t = 0; t < len; ++t) {
            TBPEToken prevToken = qArr[t].PrevToken;
            AddLabels(ModelDescr, &labels, prevToken);
        }
        yint microBatchId = 0;
        EmbedLayer.BuildIndex(microBatchId, labels, isTrain);
        EmbedLayer.Init(Stream, microBatchId, sampleEmbedVectors, isTrain);

        InitKVcachePtrs(Stream, ComputeParams.InferAtt.Get(), ModelDescr.GetMaxAttentionHistory(), qArr);

        if (len != PrevLen) {
            PrevLen = len;
            ComputeParams.Init(Stream, len);
            {
                TAttentionInfo crossAttInfo;
                crossAttInfo.Init();
                for (int i = 0; i < len; ++i) {
                    crossAttInfo.AddSpan(TAttentionSpan(0, len - 1));
                    crossAttInfo.AddSample();
                }
                TAttentionInfoGrouped<ATT_GROUP> crossAtt;
                GroupAttention<ATT_GROUP, ATT_ALIGN>(crossAttInfo, &crossAtt);
                ComputeParams.InferCrossAtt->InitForwardOnly(Stream, len, &crossAtt);
            }
        }
    }


    void InitRngSeed(const TVector<TBatchInferRequest> &qArr, ui32 seed)
    {
        TMersenne<ui64> rng(seed);
        TVector<ui32> rngSeed;
        yint len = YSize(qArr);
        for (yint t = 0; t < len; ++t) {
            rngSeed.push_back(rng.GenRand());
        }
        Put(Stream, &RngSeed, rngSeed);
    }


    void ComputeFragmentPredictions(const TVector<TBatchInferRequest> &qArr, TVector<TVector<float>> *pPrediction) override
    {
        yint len = YSize(qArr);

        Init(qArr);
        PredictComputer->Run(Stream);
        Stream.Sync();

        // fetch data
        TMemoryBlob hostMem = LogitBufHost.GetHostMem();
        pPrediction->resize(len);
        __m256 scaleMult = _mm256_set1_ps(1);
        for (yint t = 0; t < len; ++t) {
            yint width = ModelDescr.OutputTokenCount;
            yint width8 = (width + 7) & ~7;
            Y_ASSERT(width8 <= LogitBufHost.GetXSize());
            TVector<float> &dst = (*pPrediction)[t];
            dst.resize(width8);
            UnpackFp16Array(dst.data(), hostMem.GetElementAddress<fp16>(0, t), width8, scaleMult);
            dst.resize(width);
        }
    }


    void ComputeContinuation(const TVector<TBatchInferRequest> &qArr, ui32 seed, TVector<ui32> *pRes) override
    {
        yint len = YSize(qArr);

        Init(qArr);
        InitRngSeed(qArr, seed);
        SampleComputer->Run(Stream);
        ResTokens.CopyToHost(Stream, len);
        Stream.Sync();

        // fetch sampled
        GetData(ResTokens, pRes, len);
    }


    TKVcacheTracker &GetKVcache() override
    {
        return KVcache;
    }
};


TIntrusivePtr<IBatchInfer> CreateBatchInferencer(const TModelParams &params, yint maxBatchSize)
{
    return new TBatchInfer(params, maxBatchSize);
}

}
