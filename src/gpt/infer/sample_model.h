#pragma once
#include <gpt/att/sliding_window.h>
#include <gpt/data/bpe.h>
#include <gpt/data/data.h>
#include <gpt/data/fragment_gen.h>
#include <gpt/data/ppm_lmatch.h>
#include <gpt/data/ppm_window.h>
#include <gpt/gpt2/gpt2_tokenizer.h>
#include <gpt/model_params/model_params.h>
#include <gpt/compute/model.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TSamplingModelBase : public TThrRefBase
{
    TIntrusivePtr<IModel> Model;
    yint MaxLen = 0;
    bool UsePPM = false;
    bool UseLMatch = false;

public:
    TSamplingModelBase(TIntrusivePtr<TModelParamsHolder> mph)
    {
        const TModelParams &params = mph->Params;
        MaxLen = params.ModelDescr.FragLen;
        UsePPM = params.ModelDescr.HasFlag(MPF_PPM);
        UseLMatch = params.ModelDescr.HasFlag(MPF_LMATCH);
        Model = CreateHostMatrixTransformer(params, 1, MaxLen);
    }
    virtual TString SampleFromModel(TXRng &rng, const TString &prefix) = 0;
    virtual float ComputeLogLoss(const TVector<char> &text) = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TSamplingModel : public TSamplingModelBase
{
    TTokenizer Tokenizer;
    TIntrusivePtr<TLMatchSearch> LMatchSearch;
    yint DocStartToken = -1;

public:
    TSamplingModel(TIntrusivePtr<TModelParamsHolder> mph, const TTokenizer &tokenizer, const TString &lmIndexDir)
        : TSamplingModelBase(mph), DocStartToken(tokenizer.HasDocStartToken() ? tokenizer.GetDocStartToken() : -1)
    {
        if (!lmIndexDir.empty()) {
            LMatchSearch = new TLMatchSearch(lmIndexDir, DocStartToken);
        }
        Tokenizer = tokenizer;
    }
    TString SampleFromModel(TXRng &rng, const TString &prefix) override;
    float ComputeLogLoss(const TVector<char> &text) override;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TGptSamplingModel : public TSamplingModelBase
{
    TGpt2Tokenizer Tokenizer;

public:
    TGptSamplingModel(TIntrusivePtr<TModelParamsHolder> mph, const TGpt2Tokenizer &tokenizer) : TSamplingModelBase(mph)
    {
        Y_ASSERT(!UsePPM && !UseLMatch);
        Tokenizer = tokenizer;
    }
    TString SampleFromModel(TXRng &rng, const TString &prefix) override;
    float ComputeLogLoss(const TVector<char> &text) override;
};
