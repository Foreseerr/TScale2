#pragma once
#include <gpt/model_params/model_params.h>
#include <gpt/data/bpe.h>
#include <gpt/att/kv_cache.h>


struct TBatchInferRequest
{
    TBPEToken PrevToken = UNDEFINED_TOKEN; // if token is invalid should start with fragment start label
    TVector<float> EmbedVec;
    TKVcacheReference KVcache;
};


struct IBatchInfer : public TThrRefBase
{
    virtual void ComputeFragmentPredictions(const TVector<TBatchInferRequest> &qArr, TVector<TVector<float>> *pPrediction) = 0;
    virtual void ComputeContinuation(const TVector<TBatchInferRequest> &qArr, ui32 seed, TVector<ui32> *pRes) = 0;
    virtual TKVcacheTracker &GetKVcache() = 0;
};


namespace NCUDA_Transformer
{
TIntrusivePtr<IBatchInfer> CreateBatchInferencer(const TModelParams &params, yint maxBatchSize);
}
