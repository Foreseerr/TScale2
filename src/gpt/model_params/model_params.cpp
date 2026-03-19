#include "model_params.h"
#include "sse_utils.h"
#include <lib/random/rand_utils.h>
#include <util/thread.h>


struct TFastNormalRNG : public TThrRefBase
{
    enum {
        BUF_SIZE = 1 << 16,
    };
    float Buf[BUF_SIZE];
    TXRng CoreRng;

    TFastNormalRNG(yint seed) : CoreRng(seed)
    {
        for (yint k = 0; k < BUF_SIZE; k += 2) {
            //(rng.GenRandReal3() * 2 - 1) * bound;
            float val = GenNormal(CoreRng);
            Buf[k] = val;
            Buf[k + 1] = -val;
        }
    }
    ui64 GenCoreRand() { return CoreRng.GenRand(); }
    float Gen(TXRng &rng)
    {
        return Buf[((ui32)rng.GenRand()) & (BUF_SIZE - 1)];
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
inline void InitMatrixImpl(TModelMatrix *pRes, TFastNormalRNG &normalRng, ui64 seed, float sko)
{
    TXRng rng(seed);
    yint xSize = pRes->GetXSize();
    yint ySize = pRes->GetYSize();
    TArray2D<float> mm;
    mm.SetSizes(xSize, ySize);
    float bound = sko * 0.5f;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            mm[y][x] = normalRng.Gen(rng) * bound;
        }
    }
    pRes->SetMatrix(mm);
}

inline void InitEmbedMatrixImpl(TModelMatrix *pRes, TFastNormalRNG &normalRng, ui64 seed, float noiseLabelsSko)
{
    TXRng rng(seed);
    yint xSize = pRes->GetXSize();
    yint ySize = pRes->GetYSize();
    TArray2D<float> embed;
    embed.SetSizes(xSize, ySize);
    embed.FillZero();
    for (yint y = 0; y < ySize; ++y) {
        float mult = 1;
        if (y < NOISE_LABELS_COUNT) { // zero init noise tokens
            mult = noiseLabelsSko;
        }
        for (yint x = 0; x < xSize; ++x) {
            embed[y][x] = normalRng.Gen(rng) * mult;
        }
    }
    pRes->SetMatrix(embed);
}

// inline void InitIdentity(TModelMatrix *pRes)
// {
//     yint xSize = pRes->GetXSize();
//     yint ySize = pRes->GetYSize();
//     TArray2D<float> mm;
//     mm.SetSizes(xSize, ySize);
//     for (yint y = 0; y < ySize; ++y) {
//         for (yint x = 0; x < xSize; ++x) {
//             mm[y][x] = (x == y) ? 1 : 0;
//         }
//     }
//     pRes->SetMatrix(mm);
// }


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMatrixInitProcessor : public TThrRefBase
{
    enum {
        THR_COUNT = 8
    };
    enum EJob {
        INIT_EMBED,
        INIT_MATRIX,
    };
    struct TJob
    {
        EJob Job = INIT_MATRIX;
        TModelMatrix *Matrix = 0;
        ui64 Seed = 0;
        float Sko = 0;

        TJob() {}
        TJob(EJob job, TModelMatrix *p, ui64 seed, float sko) : Job(job), Matrix(p), Seed(seed), Sko(sko) {}
    };

    TFastNormalRNG NormalRng;
    TThread ThrArr[THR_COUNT];
    TVector<TJob> JobArr;
    std::atomic<yint> JobId;

public:
    void WorkerThread()
    {
        for (;;) {
            yint jobId = JobId.fetch_add(1);
            if (jobId >= YSize(JobArr)) {
                return;
            }
            const TJob &job = JobArr[jobId];
            if (job.Job == INIT_EMBED) {
                InitEmbedMatrixImpl(job.Matrix, NormalRng, job.Seed, job.Sko);
            } else if (job.Job == INIT_MATRIX) {
                InitMatrixImpl(job.Matrix, NormalRng, job.Seed, job.Sko);
            } else {
                Y_VERIFY(0);
            }
        }
    }

public:
    TMatrixInitProcessor(yint seed) : NormalRng(seed), JobId(0) {}
    void InitEmbedMatrix(TModelMatrix *p, float noiseLabelSko) { JobArr.push_back(TJob(INIT_EMBED, p, NormalRng.GenCoreRand(), noiseLabelSko)); }
    void InitMatrix(TModelMatrix *p, float sko) { JobArr.push_back(TJob(INIT_MATRIX, p, NormalRng.GenCoreRand(), sko)); }
    void Run()
    {
        for (yint thrId = 0; thrId < THR_COUNT; ++thrId) {
            ThrArr[thrId].Create(this);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// Model Params Initialize

static void AllocateModel(TModelParams *p, const TModelDescr &descr)
{
    p->ModelDescr = descr;
    yint dim = descr.Dims.Dim;
    p->MatrArr.resize(MP_MODEL_COUNT);
    p->MatrArr[MP_MODEL_EMBED].SetSizesPerRowDisp(dim, descr.LabelCount, 1);
    p->MatrArr[MP_MODEL_FINAL].SetSizesPerRowDisp(dim, descr.OutputTokenCount, 1);
    yint depth = YSize(descr.LayerArr);
    p->LayerArr.resize(depth);
    for (yint d = 0; d < depth; ++d) {
        TModelParams::TAttentionMatrices &dst = p->LayerArr[d];
        dst.MatrArr.resize(MP_MAX_COUNT);
        yint ptr = 0;
        descr.EnumLayerMatrices(d, [&](yint xSize, yint ySize, float noise, float moeFraction) {
            dst.MatrArr[ptr++].SetSizesPerMatrixDisp(xSize, ySize, moeFraction);
        });
        dst.MatrArr.resize(ptr);
    }
    ClearPodArray(&p->Bias, descr.OutputTokenCount);
}


void InitModel(TModelParams *pParams, TXRng &rngArg, const TModelDescr &modelDescr, ECombinerInit combinerInit, const TVector<float> &biasArr)
{
    AllocateModel(pParams, modelDescr);

    TIntrusivePtr<TMatrixInitProcessor> matrixInit = new TMatrixInitProcessor(rngArg.GenRand());
    float noiseLabelsSko = modelDescr.HasFlag(MPF_DISABLE_NOISE_LABELS) ? 1 : 0;
    matrixInit->InitEmbedMatrix(&pParams->MatrArr[MP_MODEL_EMBED], noiseLabelsSko);
    matrixInit->InitMatrix(&pParams->MatrArr[MP_MODEL_FINAL], 1); // init for fixed final layer

    yint depth = YSize(modelDescr.LayerArr);
    pParams->LayerArr.resize(depth);
    for (yint d = 0; d < depth; ++d) {
        TModelParams::TAttentionMatrices &dst = pParams->LayerArr[d];
        dst.MatrArr.resize(MP_MAX_COUNT);
        yint ptr = 0;
        modelDescr.EnumLayerMatrices(d, [&](yint xSize, yint ySize, float noise, float moeFraction) {
            float sko = (combinerInit == COMBINER_INIT_RANDOM) ? 1 : noise;
            matrixInit->InitMatrix(&dst.MatrArr[ptr], sko);
            ++ptr;
        });
        dst.MatrArr.resize(ptr);
    }
    Y_VERIFY(YSize(biasArr) == modelDescr.OutputTokenCount);
    pParams->Bias = biasArr;
    matrixInit->Run();
    matrixInit = 0; // wait completion
}


void ReplaceHead(TModelParams *pParams, TXRng &rngArg, const TVector<float> &biasArr)
{
    TIntrusivePtr<TFastNormalRNG> rngHolder = new TFastNormalRNG(rngArg.GenRand());
    TFastNormalRNG &normalRng = *rngHolder.Get();
    yint dim = pParams->ModelDescr.Dims.Dim;
    yint sz = YSize(biasArr);
    TModelMatrix &head = pParams->MatrArr[MP_MODEL_FINAL];
    head.SetSizesPerRowDisp(dim, sz, 1);
    InitMatrixImpl(&head, normalRng, normalRng.GenCoreRand(), 1); // init for fixed final layer
    pParams->Bias = biasArr;
    pParams->ModelDescr.OutputTokenCount = sz;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// background save model
struct TModelSaveThread : public TThrRefBase
{
    TIntrusivePtr<TModelParamsHolder> Params;
    TString FName;
    TThread Thr;

public:
    TModelSaveThread(TPtrArg<TModelParamsHolder> params, const TString &fname) : Params(params.Get()), FName(fname)
    {
        Thr.Create(this);
    }
    void WorkerThread()
    {
        Serialize(IO_WRITE, FName, Params->Params);
        Params = 0;
    }
};

static TIntrusivePtr<TModelSaveThread> CurrentBackgroundModelSave;
void BackgroundSaveModel(TPtrArg<TModelParamsHolder> params, const TString &fname)
{
    CurrentBackgroundModelSave = new TModelSaveThread(params, fname);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// perform op on each model matrix
template <class TMP, class T>
void ForEachModelMatrix(TMP &modelParams, T func)
{
    for (auto &mm : modelParams.MatrArr) {
        func(mm);
    }
    for (auto &att : modelParams.LayerArr) {
        for (auto &mm : att.MatrArr) {
            func(mm);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// model ops
//
yint CountModelSize(const TModelParams &params)
{
    yint res = 0;
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        res += mm.GetXSize() * mm.GetYSize();
        });
    res += YSize(params.Bias);
    return res;
}

yint CountActiveModelSize(const TModelParams &params)
{
    yint res = 0;
    for (const TModelMatrix &mm : params.MatrArr) {
        res += mm.GetXSize() * mm.GetYSize();
    }
    for (yint d = 0; d < YSize(params.ModelDescr.LayerArr); ++d) {
        params.ModelDescr.EnumLayerMatrices(
            d, [&](yint xSize, yint ySize, float noise, float moeFraction) { res += xSize * ySize * moeFraction; });
    }
    res += YSize(params.Bias);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void InitZero(TAllModelMatrices *p, const TModelDescr &descr)
{
    p->Clear();
    const TModelDims &dims = descr.Dims;
    int dim = dims.Dim;
    p->AddMatrix(dim, descr.LabelCount);
    p->AddMatrix(dim, descr.OutputTokenCount);
    yint depth = YSize(descr.LayerArr);
    for (yint d = 0; d < depth; ++d) {
        descr.EnumLayerMatrices(
            d, [&](yint xSize, yint ySize, float noise, float moeFraction) { p->AddMatrix(xSize, ySize); });
    }
}

void GetMatrices(TAllModelMatrices *p, const TModelParams &params)
{
    p->Clear();
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        p->MatrArr.push_back(mm.GetMatrix());
        });
}

void SetMatrices(TModelParams *p, const TAllModelMatrices &params)
{
    yint k = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        mm.GetMatrix() = params.MatrArr[k++];
        });
}

void AddMatrices(TModelParams *p, const TAllModelMatrices &params, float scale)
{
    yint k = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        AddScaledMatrixAligned(&mm.GetMatrix(), params.MatrArr[k++], scale);
        });
}

void GetGradient(TAllModelMatrices *p, const TModelParams &params)
{
    p->Clear();
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        p->MatrArr.push_back(mm.GetGrad1());
        });
}

void SetGradient(TModelParams *p, const TAllModelMatrices &params)
{
    yint k = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        mm.GetGrad1() = params.MatrArr[k++];
        });
}




///////////////////////////////////////////////////////////////////////////////////////////////////
void PackMatrices(TBufferedStream &f, const TModelParams &params)
{
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        PackMatrix(f, mm.GetMatrix().GetHostPtr());
        });
}

void PackMatrices(TBufferedStream &f, const TAllModelMatrices &params)
{
    for (auto &mm : params.MatrArr) {
        PackMatrix(f, mm.GetHostPtr());
    }
}

void AddPackedMatrices(TAllModelMatrices *p, TBufferedStream &f, float scale)
{
    for (auto &mm : p->MatrArr) {
        AddPackedMatrix(mm.GetHostPtr(), f, scale);
    }
}

void AddPackedMatrices(TModelParams *p, TBufferedStream &f, float scale)
{
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        AddPackedMatrix(mm.GetMatrix().GetHostPtr(), f, scale);
        });
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void PackModelParams(TModelParams &params, TVector<char> *pBuf)
{
    TMemStream mem;
    {
        TBufferedStream bufIO(IO_WRITE, mem);
        WriteStruct(bufIO, params.ModelDescr);
        WriteStruct(bufIO, params.Bias);
        PackMatrices(bufIO, params);
    }
    mem.Swap(pBuf);
}

void UnpackModelParams(TVector<char> &buf, TModelParams *p)
{
    TMemStream mem(&buf);
    {
        TBufferedStream bufIO(IO_READ, mem);
        TModelDescr modelDescr;
        ReadStruct(bufIO, modelDescr);
        AllocateModel(p, modelDescr);
        ReadStruct(bufIO, p->Bias);
        AddPackedMatrices(p, bufIO, 1);
    }
    mem.Swap(&buf);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void GetRowDisp(TModelRowDisp *p, const TModelParams &params)
{
    p->Clear();
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        p->AddMatrixRowDisp(mm.GetRowDisp(), mm.GetSumWeight());
        });
}

void SetRowDisp(TModelParams *p, const TModelRowDisp &rd)
{
    yint ptr = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        yint sz = YSize(mm.GetRowDisp());
        TVector<float> rowDisp;
        for (yint k = 0; k < sz; ++k) {
            rowDisp.push_back(rd.RowDisp[ptr++]);
        }
        mm.SetRowDisp(rd.SumWeight, rowDisp);
        });
}
