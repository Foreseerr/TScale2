#include "model_dim.h"
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDescrString
{
    TModelDims Dims;
    yint MaxHistory = 64;

    TModelDescrString() {}
    TModelDescrString(const TString &modelDescrStr);
};

TModelDescrString::TModelDescrString(const TString &modelDescrStr)
{
    TStringParams sp(modelDescrStr);
    for (TStringParams::TParam &param : sp.Params) {
        if (param.Name == "e") {
            Dims.Dim = param.Value;
        } else if (param.Name == "xp") {
            Dims.ExpertDim = param.Value;
        } else if (param.Name == "h") {
            Dims.HeadCount = param.Value;
        } else if (param.Name == "d") {
            Dims.Depth = param.Value;
        } else if (param.Name == "w") {
            MaxHistory = param.Value;
        } else if (param.Name == "ffn") {
            Dims.FfnTiles = param.Value;
        } else if (param.Name == "moe") {
            Dims.MoeExpertCount = param.Value;
        } else if (param.Name == "moes") {
            Dims.MoeSelected = param.Value;
        } else if (param.Name == "rep") {
            Dims.KVrep = param.Value;
        } else {
            DebugPrintf("unknown model descr param %s\n", param.Name.c_str());
        }
    }
}


TString GetModelDescrString(const TModelDescr &modelDescr)
{
    TModelDims defDims;
    TString res = Sprintf("e%g", modelDescr.Dims.Dim * 1.);
    if (modelDescr.Dims.ExpertDim != defDims.ExpertDim) {
        res += Sprintf("xp%g", modelDescr.Dims.ExpertDim * 1.);
    }
    if (modelDescr.Dims.HeadCount != defDims.HeadCount) {
        res += Sprintf("h%g", modelDescr.Dims.HeadCount * 1.);
    }
    res += Sprintf("d%g", modelDescr.Dims.Depth * 1.);
    if (modelDescr.Dims.FfnTiles != defDims.FfnTiles) {
        res += Sprintf("ffn%g", modelDescr.Dims.FfnTiles * 1.);
    }
    if (modelDescr.Dims.MoeExpertCount != defDims.MoeExpertCount) {
        res += Sprintf("moe%g", modelDescr.Dims.MoeExpertCount * 1.);
    }
    if (modelDescr.Dims.MoeSelected != defDims.MoeSelected) {
        res += Sprintf("moes%g", modelDescr.Dims.MoeSelected * 1.);
    }
    if (modelDescr.Dims.KVrep != defDims.KVrep) {
        res += Sprintf("rep%g", modelDescr.Dims.KVrep * 1.);
    }
    yint maxHistory = modelDescr.GetMaxAttentionHistory();
    res += Sprintf("w%g", maxHistory * 1.);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EAlibi
{
    ALIBI_V3,
    ALIBI_RNN,
    ALIBI_RNN_CROSS,
};


static void InitAlibi(TModelDescr *p, EAlibi alibi, yint wideLimitWindow)
{
    yint depth = p->Dims.Depth;
    p->LayerArr.resize(0);
    p->AttentionTypeArr.clear();
    if (alibi == ALIBI_RNN) {
        p->AttentionTypeArr.push_back(CasualAttn(1));
        p->AttentionTypeArr.push_back(CasualAttn(128));
        // ELayerType attLayer = MLT_ATT;
        ELayerType attLayer = MLT_ATT_3LIN;
        for (yint d = 0; d < depth; ++d) {
            p->AddLayer(attLayer, 0, 0.0f);
            p->AddLayer(attLayer, 1, 1.f);
            p->AddLayer(attLayer, 1, 0.5f);
            p->AddLayer(attLayer, 1, 0.25f);
        }
    } else if (alibi == ALIBI_RNN_CROSS) {
        p->AttentionTypeArr.push_back(CasualAttn(1));
        p->AttentionTypeArr.push_back(CasualAttn(128));
        p->AttentionTypeArr.push_back(CrossBatchAttn());
        // ELayerType attLayer = MLT_ATT;
        ELayerType attLayer = MLT_ATT_3LIN;
        for (yint d = 0; d < depth; ++d) {
            p->AddLayer(attLayer, 0, 0.0f);
            p->AddLayer(attLayer, 2, 0); // cross attn
            p->AddLayer(attLayer, 1, 1.f);
            p->AddLayer(attLayer, 2, 0); // cross attn
            p->AddLayer(attLayer, 1, 0.5f);
            p->AddLayer(attLayer, 2, 0); // cross attn
            p->AddLayer(attLayer, 1, 0.25f);
            p->AddLayer(attLayer, 2, 0); // cross attn
        }
    } else {
        // useful in attention profiling
        if (depth == 1) {
            p->AttentionTypeArr.push_back(CasualAttn(wideLimitWindow));
            p->AddLayer(MLT_ATT, 0, 0.1f);
            //p->AddLayer(MLT_ATT_3LIN, 0, 0.1f);
            if (p->Dims.MoeExpertCount != 0) {
                p->AddLayer(MLT_MOE); // moe test
            } else {
                p->AddLayer(MLT_FFN);
            }
        } else {
            // configure position encoding
            if (alibi == ALIBI_V3) {
                //
                p->AttentionTypeArr.push_back(CasualAttn(1));
                p->AttentionTypeArr.push_back(CasualAttn(64));
                p->AttentionTypeArr.push_back(CasualAttn(wideLimitWindow));
                p->LayerArr.resize(0);
                ELayerType attLayer = MLT_ATT;
                // ELayerType attLayer = MLT_ATT_3LIN;
                if (depth > 0) {
                    yint groupCount = DivCeil(depth, 2);
                    yint w1count = Max<yint>(1, groupCount / 6);
                    for (yint k = 0; k < w1count; ++k) {
                        p->AddLayer(attLayer, 0, 0.0f);
                        p->AddLayer(MLT_FFN); // redundant for smaller models
                        p->AddLayer(attLayer, 0, 0.0f);
                        p->AddLayer(MLT_FFN);
                    }
                    ELayerType ffn = (p->Dims.IsUsingMoE()) ? MLT_MOE : MLT_FFN;
                    for (yint k = w1count; k < groupCount; ++k) {
                        p->AddLayer(attLayer, 1, 0.5f);
                        p->AddLayer(ffn);
                        p->AddLayer(attLayer, 2, 0.0f);
                        p->AddLayer(ffn);
                    }
                }

            } else {
                Y_VERIFY("unknown alibi version");
            }
        }
    }
}


void InitModelDescrImpl(TModelDescr *pRes, const TString &modelDescrStr, yint inputTokenCount, yint outputTokenCount, yint labelCount, ui64 flags)
{
    TModelDescrString dims(modelDescrStr);

    TModelDescr &modelDescr = *pRes;
    modelDescr = TModelDescr();
    modelDescr.Dims = dims.Dims;
    modelDescr.LabelCount = labelCount;
    modelDescr.InputTokenCount = inputTokenCount;
    modelDescr.OutputTokenCount = outputTokenCount;
    modelDescr.Flags = flags;
    EAlibi alibi = ALIBI_V3;
    if (modelDescr.HasFlag(MPF_RNN)) {
        alibi = ALIBI_RNN;
    }
    if (modelDescr.HasFlag(MPF_RNN_CROSS)) {
        alibi = ALIBI_RNN_CROSS;
    }
    InitAlibi(&modelDescr, alibi, dims.MaxHistory);
}
