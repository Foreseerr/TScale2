#pragma once
#include <lib/random/xrng.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
// model matrices discretization step
//constexpr float MODEL_DISCR_SCALE = 1.f / 16;
//constexpr float MODEL_DISCR_SCALE = 1.f / 24;
constexpr float MODEL_DISCR_SCALE = 1.f / 32;

///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint NOISE_LABELS_COUNT = 32;

///////////////////////////////////////////////////////////////////////////////////////////////////
// what scaling is optimal and why we need it all is unclear
constexpr float LOG2 = 0.693147f;
constexpr float FINAL_LAYER_SOFTMAX_SCALE = 1;
constexpr float ATT_DOTPRODUCT_SCALE = 0.5f;
constexpr float MOE_SCALE = 0.5f;
constexpr int STATE_NORM_TILE = 128;
constexpr float LOLU_SCALE = 2;

#define CalcDotScale(dim) (sqrtf(1.0f / (dim)))

#define CalcFinalLayerMult() (FINAL_LAYER_SOFTMAX_SCALE / LOG2)

#define CalcAttentionMult() (ATT_DOTPRODUCT_SCALE / LOG2)

#define GetAttentionDecay(dist, alibiSlope) (-(alibiSlope) * (dist))



///////////////////////////////////////////////////////////////////////////////////////////////////
// model params flags
const ui64 MPF_NOFLAGS = 0;
const ui64 MPF_HASHED_EMBED = 0x1;
const ui64 MPF_PPM = 0x2;
const ui64 MPF_USE_DOC_START_TOKEN = 0x4;
const ui64 MPF_DISABLE_TUNE_EMBED = 0x8;
const ui64 MPF_SAMPLE_EMBED_VECTORS = 0x10;
const ui64 MPF_TAIL_LOSS = 0x20;
const ui64 MPF_SIM_QUANT_2BIT = 0x40;
const ui64 MPF_SIM_QUANT_4BIT = 0x80;
const ui64 MPF_GROK_BINARY_OP = 0x100;
const ui64 MPF_MLM_BERT = 0x200;
const ui64 MPF_DISABLE_NOISE_LABELS = 0x800;
const ui64 MPF_LMATCH = 0x1000;
const ui64 MPF_SYNC_ALL_GRADIENTS = 0x2000;
const ui64 MPF_DISABLE_TUNE_FINAL_LAYER = 0x4000;
const ui64 MPF_DISABLE_TUNE_LAYERS = 0x8000;
const ui64 MPF_RNN = 0x10000;
const ui64 MPF_RNN_CROSS = 0x20000;


///////////////////////////////////////////////////////////////////////////////////////////////////
// layer types and matrices ids
enum ELayerType {
    MLT_ATT,
    MLT_ATT_3LIN,
    MLT_FFN,
    MLT_MOE,
};


enum {
    MP_ATT_Q = 0, // attention current token
    MP_ATT_K = 1, // attention remote token
    MP_ATT_V = 2, // remote token (lookup result)
    MP_ATT_GATE = 3,
    MP_ATT_COMBINER = 4,
    MP_FFN_EXPAND = 0,
    MP_FFN_GATE = 1,
    MP_FFN_CONTRACT = 2,
    MP_MOE_EXPAND = 0,
    MP_MOE_GATE = 1,
    MP_MOE_CONTRACT = 2,
    MP_MOE_SELECT = 3,
    MP_MAX_COUNT = 5
};


enum {
    MP_MODEL_EMBED = 0,
    MP_MODEL_FINAL = 1,
    MP_MODEL_COUNT = 2,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAttentionType
{
    enum {
        CASUAL = 0,
        CROSS_BATCH,
    };
    int Width = 0;
    int Type = CASUAL;

    TAttentionType() {}
    TAttentionType(int w, int t) : Width(w), Type(t) {}
    inline bool operator==(const TAttentionType &x) const { return Width == x.Width && Type == x.Type; }
};


inline TAttentionType CasualAttn(int width)
{
    return TAttentionType(width, TAttentionType::CASUAL);
}

inline TAttentionType CrossBatchAttn()
{
    return TAttentionType(0, TAttentionType::CROSS_BATCH);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDims
{
    int Dim = 256;
    int QDim = 128;
    int TTDim = 128;
    int ExpertDim = 128;
    int HeadCount = 1;
    int FfnTiles = 1;
    int MoeExpertCount = 0;
    int MoeSelected = 2; // selected experts
    int KVrep = 16;
    int Depth = 60;
    SAVELOAD(Dim, QDim, TTDim, ExpertDim, HeadCount, FfnTiles, MoeExpertCount, MoeSelected, KVrep, Depth);

    yint GetQSum() const { return QDim * HeadCount; }
    yint GetTTSum() const { return TTDim * HeadCount; }
    yint GetFfnDim() const { return FfnTiles * 128; }
    yint GetExpertDim() const { return ExpertDim; }
    yint GetMoeDim() const { return MoeExpertCount * ExpertDim; }
    yint GetMoeExpertCount() const { return MoeExpertCount; }
    yint GetMoeSelectedCount() const { return MoeSelected; }
    bool IsUsingMoE() const { return MoeExpertCount > 0; }
    yint GetKVrep() const { return KVrep; }
};

inline bool operator==(const TModelDims &a, const TModelDims &b)
{
    return a.Dim == b.Dim && a.QDim == b.QDim && a.TTDim == b.TTDim && a.ExpertDim == b.ExpertDim && a.HeadCount == b.HeadCount &&
           a.FfnTiles == b.FfnTiles && a.MoeExpertCount == b.MoeExpertCount && a.MoeSelected == b.MoeSelected && a.Depth == b.Depth;
}

inline bool operator!=(const TModelDims &a, const TModelDims &b)
{
    return !(a == b);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDescr
{
    struct TLayerParams
    {
        ELayerType LayerType = MLT_ATT;
        float AlibiSlope = 0;
        int AttentionTypeId = 0;

        TLayerParams() {}
        TLayerParams(ELayerType lt, float slope, yint attTypeId) : LayerType(lt), AlibiSlope(slope), AttentionTypeId(attTypeId) {}
    };

    TModelDims Dims;
    yint LabelCount = 0;
    yint InputTokenCount = 0;
    yint OutputTokenCount = 0;
    TVector<TAttentionType> AttentionTypeArr;
    TVector<TLayerParams> LayerArr;
    ui64 Flags = 0;
    ui64 DocStartToken = 0;
    yint FragLen = 0;
    SAVELOAD(Dims, LabelCount, InputTokenCount, OutputTokenCount, AttentionTypeArr, LayerArr, Flags, DocStartToken, FragLen);

    yint GetDepth() const { return YSize(LayerArr); }

    yint GetAttentionTypeCount() const { return YSize(AttentionTypeArr); }

    yint GetCrossAttentionId() const
    {
        for (yint k = 0; k < YSize(AttentionTypeArr); ++k) {
            if (AttentionTypeArr[k].Type == TAttentionType::CROSS_BATCH) {
                return k;
            }
        }
        return -1;
    }

    yint GetMaxAttentionHistory() const
    {
        yint res = 1;
        for (const TAttentionType &x : AttentionTypeArr) {
            if (x.Type == TAttentionType::CASUAL) {
                res = Max<yint>(x.Width, res);
            }
        }
        return res;
    }

    bool HasFlag(ui64 f) const { return (Flags & f) != 0; }

    void SetDocStartToken(ui64 token)
    {
        Flags |= MPF_USE_DOC_START_TOKEN;
        DocStartToken = token;
    }

    void AddLayer(ELayerType lt, yint attWidthId, float alibiSlope) { LayerArr.push_back(TLayerParams(lt, alibiSlope, attWidthId)); }

    void AddLayer(ELayerType lt) { LayerArr.push_back(TLayerParams(lt, 0, 0)); }

    void AddLayer(const TLayerParams &lp) { LayerArr.push_back(lp); }

    template <class Func>
    void EnumLayerMatrices(yint d, Func f) const
    {
        yint dim = Dims.Dim;
        yint qSum = Dims.GetQSum();
        yint ttSum = Dims.GetTTSum();
        yint ffnDim = Dims.GetFfnDim();
        yint xpDim = Dims.GetExpertDim();
        yint moeDim = Dims.GetMoeDim();
        yint kvRep = Dims.GetKVrep();
        yint xpCount = Dims.GetMoeExpertCount();
        float moeFraction = Dims.MoeSelected / (xpCount + 0.);
        switch (LayerArr[d].LayerType) {
        case MLT_ATT:
            f(dim, qSum, 1, 1);
            f(dim, qSum, 1, 1);
            f(dim, ttSum, 1, 1);
            f(dim, ttSum, 1, 1);
            f(ttSum, dim, 0, 1);
            break;
        case MLT_ATT_3LIN:
            f(dim, qSum, 1, 1);
            f(dim, qSum, 1, 1);
            f(dim, ttSum, 1, 1);
            f(dim, ttSum, 1, 1);
            f(ttSum * kvRep, dim, 0, 1);
            break;
        case MLT_FFN:
            f(dim, ffnDim, 1, 1);
            f(dim, ffnDim, 1, 1);
            f(ffnDim, dim, 0, 1);
            break;
        case MLT_MOE:
            f(dim, xpDim * xpCount, 1, moeFraction);
            f(dim, xpDim * xpCount, 1, moeFraction);
            f(xpDim, dim * xpCount, 0, moeFraction);
            f(dim, xpCount, 1, 1);
            break;
        }
    }
};

inline bool operator==(const TModelDescr::TLayerParams &a, const TModelDescr::TLayerParams &b)
{
    return a.LayerType == b.LayerType && a.AlibiSlope == b.AlibiSlope && a.AttentionTypeId == b.AttentionTypeId;
}

inline bool operator==(const TModelDescr &a, const TModelDescr &b)
{
    return a.Dims == b.Dims && a.LabelCount == b.LabelCount && a.InputTokenCount == b.InputTokenCount &&
           a.OutputTokenCount == b.OutputTokenCount && a.AttentionTypeArr == b.AttentionTypeArr && a.LayerArr == b.LayerArr &&
           a.Flags == b.Flags && a.DocStartToken == b.DocStartToken && a.FragLen == b.FragLen;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// quant
enum EModelMatrixQuant {
    MM_QUANT_NONE,
    MM_QUANT_158BIT, // -1 / 0 / 1
    MM_QUANT_2BIT,
    MM_QUANT_4BIT,
};


inline EModelMatrixQuant GetQuant(const TModelDescr &modelDescr)
{
    EModelMatrixQuant quant = MM_QUANT_NONE;
    if (modelDescr.HasFlag(MPF_SIM_QUANT_2BIT)) {
        quant = MM_QUANT_2BIT;
    } else if (modelDescr.HasFlag(MPF_SIM_QUANT_4BIT)) {
        quant = MM_QUANT_4BIT;
    }
    return quant;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
enum ECombinerInit
{
    COMBINER_INIT_RANDOM,
    COMBINER_INIT_ZERO,
};


void InitModelDescrImpl(TModelDescr *pRes, const TString &modelDescrStr, yint inputTokenCount, yint outputTokenCount, yint labelCount, ui64 flags);
TString GetModelDescrString(const TModelDescr &modelDescr);
