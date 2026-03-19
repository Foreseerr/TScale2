#include "net_train.h"
#include "net_ib_train.h"
#include "mmlu_score.h"
#include <gpt/data/data.h>
#include <gpt/data/bpe.h>
#include <gpt/data/fragment_gen.h>
#include <gpt/compute/model_inst_init.h>
#include <gpt/compute/infer_batch.cuh>
#include <gpt/data_config/data_config.h>
#include <gpt/train_config/train_config.h>
#include <gpt/train_ctx/train_ctx.h>
#include <gpt/train_ctx/backprop.h>
#include <lib/cuda/cuda_init.h>
#include <lib/math/softmax.h>
#include <lib/math/linear.h>
#include <lib/random/rand_utils.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data script
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// load text or connect
static TString DATA_SCRIPT = " make_char_dataset('enwik8')";

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Train script
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static TString TRAIN_SCRIPT = " check_cpu_gpu_match()";


///////////////////////////////////////////////////////////////////////////////////////////////////
static TIntrusivePtr<IModel> CreateTrainModel(const TModelParams &params, yint deviceCount, const TModelSplit &msplit, bool cudaGradient, yint nodeCount)
{
    if (cudaGradient) {
        return CreateLocalTransformer(params, deviceCount, nodeCount, msplit);
    } else {
        if (msplit.TP == 1 && msplit.PP == 1 && msplit.MicroBatchCount == 1) {
            return CreateHostMatrixTransformer(params, deviceCount, nodeCount);
        } else {
            DebugPrintf("for TP/PP/MB use CUDA_GRADIENT\n");
            abort();
            return 0;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute model params distribution
static void ComputeMatrixParamDistr(TModelParams *pParams)
{
    for (TModelParams::TAttentionMatrices &att : pParams->LayerArr) {
        TModelMatrix &matr = att.MatrArr[MP_ATT_V];
        const TArray2D<float> &data = matr.GetMatrix();
        yint xSize = matr.GetXSize();
        yint ySize = matr.GetYSize();
        double sum2 = 0;
        for (yint y = 0; y < ySize; ++y) {
            for (yint x = 0; x < xSize; ++x) {
                sum2 += Sqr(data[y][x]);
            }
        }
        double sko = sqrt(sum2 / xSize / ySize);
        double count3 = 0;
        double count5 = 0;
        double count7 = 0;
        for (yint y = 0; y < ySize; ++y) {
            for (yint x = 0; x < xSize; ++x) {
                double val = fabs(data[y][x] / sko);
                if (val > 3) {
                    count3 += 1;
                }
                if (val > 5) {
                    count5 += 1;
                }
                if (val > 7) {
                    count7 += 1;
                }
            }
        }
        double scale = 100. / xSize / ySize;
        DebugPrintf("sko %g, 3sko %g%%, 5sko %g%%, 7sko %g%%\n", sko, count3 * scale, count5 * scale, count7 * scale);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void ComputeAverageModel(TModelParams *p, yint finishIter, yint iterInterval, yint step)
{
    TString pathTemplate = GetHomeDir() + "eden_gpt_%.8gk.bin";
    // TString pathTemplate = GetHomeDir() + "models/fed_small/model_%.8g.bin ";

    // model averaging boosts perf on test significantly
    int startIter = finishIter - iterInterval;
    double modelCount = 1;
    TModelParams &res = *p;
    Serialize(IO_READ, Sprintf(pathTemplate.c_str(), startIter / 1000.), res);
    TAllModelMatrices sum;
    GetMatrices(&sum, res);
    for (int iter = startIter + step; iter <= finishIter; iter += step) {
        TModelParams params;
        Serialize(IO_READ, Sprintf(pathTemplate.c_str(), iter / 1000.), params);
        TAllModelMatrices pp;
        GetMatrices(&pp, params);
        sum.AddScaled(pp, 1);
        modelCount += 1;
        printf(".");
    }
    printf("\n");
    sum.Scale(1 / modelCount);
    SetMatrices(&res, sum);
    //ComputeMatrixParamDistr(&startParams);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute score on test set
static void ComputeExactTest(TPtrArg<IDataSource> data, const TModelParams &params)
{
    yint fragLen = params.ModelDescr.FragLen;
    //yint testBatchSize = BUFFER_LEN / GetNodeCount(fragLen);
    //yint testBatchSize = 4;
    yint testBatchSize = 1;

    TIntrusivePtr<IModel> pModel = CreateTrainModel(params, 1, TModelSplit(), false, testBatchSize * fragLen);
    TIntrusivePtr<IModelInstance> pCtx = pModel->GetInstance(0);
    double sumTestErr = 0;
    double sumCount = 0;
    int rngSeed = 31331;
    for (yint iter = 1;; ++rngSeed, ++iter) {
        TVector<TFragment> batchArr;
        data->SampleFragments(IDataSource::TEST, rngSeed, testBatchSize, fragLen, &batchArr);
        float testErr = CalcModelErr(batchArr, pCtx) * data->GetStats().Compression;
        if (isnan(testErr)) {
            DebugPrintf("rseed %g, score is nan\n", rngSeed * 1.);
        }
        sumTestErr += testErr;
        sumCount += 1;
        if ((iter % 100) == 0) {
            DebugPrintf("iter %gk, avrg test score %g\n", iter / 1000., sumTestErr / sumCount); fflush(0);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// check if results are reproducible
template <class T>
double CalcDiffSqr(const TVector<T> &cpuPredArr, const TVector<T> &gpuPredArr)
{
    double totalDiff = 0;
    Y_ASSERT(YSize(cpuPredArr) == YSize(gpuPredArr));
    for (yint k = 0; k < YSize(cpuPredArr); ++k) {
        totalDiff += Sqr(cpuPredArr[k] - gpuPredArr[k]);
    }
    return totalDiff;
}

template <class T>
double CalcDiff(const TVector<TVector<T>> &cpuPredArr, const TVector<TVector<T>> &gpuPredArr)
{
    double totalDiff = 0;
    for (yint t = 0; t < YSize(cpuPredArr); ++t) {
        totalDiff += CalcDiffSqr(cpuPredArr[t], gpuPredArr[t]);
    }
    return sqrt(totalDiff / YSize(cpuPredArr) / YSize(cpuPredArr[0]));
}

// static bool TestMatch(const TArray2D<float> &a, const TArray2D<float> &b)
//{
//     for (yint y = 0; y < a.GetYSize(); ++y) {
//         for (yint x = 0; x < a.GetXSize(); ++x) {
//             if (a[y][x] != b[y][x]) {
//                 printf("%g != %g  (%x %x)\n", a[y][x], b[y][x], *(int*)&a[y][x], *(int*)&b[y][x]);
//                 return false;
//             }
//         }
//     }
//     return true;
// }
//
// void TestReproducibility(const TTrainContext &trainCtx, IComputeContext *pCtx, TXRng &rng, const TVector<TFragment> &fragArr)
//{
//     const TDescentConfig &dc = trainCtx.GetDescentConfig();
//
//     TModelParams point1;
//     pCtx->GetParams(&point1);
//     pCtx->SetParams(point1);
//
//     TXRng chkRng = rng;
//     TVector<TNodeTarget> batchTarget;
//     MakeTrain(rng, fragArr, dc.TokenDrop, pCtx, &batchTarget);
//     pCtx->Backprop(dc.Step, GRADIENT_APPLY);
//
//     TModelParams point2;
//     pCtx->GetParams(&point2);
//
//     for (yint testId = 0; testId < 5; ++testId) {
//         pCtx->SetParams(point1);
//
//         pCtx->Backprop(dc.Step, GRADIENT_APPLY);
//
//         TModelParams chk;
//         pCtx->GetParams(&chk);
//
//         bool hasMismatch = false;
//         if (!TestMatch(chk.LabelEmbed.GetMatrix(), point2.LabelEmbed.GetMatrix())) {
//             printf("Label embed mismatch\n");
//             hasMismatch = true;
//         }
//         for (yint d = 0; d < YSize(point2.LayerArr); ++d) {
//             for (yint k = 0; k < YSize(point2.LayerArr[d]); ++k) {
//                 const TModelParams::TAttentionMatrices &att1 = point2.LayerArr[d][k];
//                 const TModelParams::TAttentionMatrices &att2 = chk.LayerArr[d][k];
//                 if (!TestMatch(att1.QK, att2.QK)) {
//                     printf("Layer %g, att %g, QK mismatch\n", d * 1., k * 1.);
//                     hasMismatch = true;
//                 }
//                 if (!TestMatch(att1.QV, att2.QV)) {
//                     printf("Layer %g, att %g, QV mismatch\n", d * 1., k * 1.);
//                     hasMismatch = true;
//                 }
//                 if (!TestMatch(att1.V, att2.V)) {
//                     printf("Layer %g, att %g, V mismatch\n", d * 1., k * 1.);
//                     hasMismatch = true;
//                 }
//                 if (!TestMatch(att1.K, att2.K)) {
//                     printf("Layer %g, att %g, K mismatch\n", d * 1., k * 1.);
//                     hasMismatch = true;
//                 }
//                 if (!TestMatch(att1.Combiner, att2.Combiner)) {
//                     printf("Layer %g, att %g, Combiner mismatch\n", d * 1., k * 1.);
//                     hasMismatch = true;
//                 }
//             }
//         }
//         if (hasMismatch) {
//             printf("attempt %g\n", testId + 1.);
//             while (hasMismatch) {
//                 SchedYield();
//             }
//         }
//     }
// }


///////////////////////////////////////////////////////////////////////////////////////////////////
void CheckCpuGpuMatch(const TDescentConfig &dc, TPtrArg<IDataSource> data)
{
    const yint CHECK_BATCH_SIZE = 1;
    //const yint CHECK_BATCH_SIZE = 3;
    //const yint CHECK_BATCH_SIZE = 32;
    const yint CHECK_FRAG_LEN = 64 - 1;
    constexpr yint NODE_COUNT = CHECK_BATCH_SIZE * CHECK_FRAG_LEN;

    TXRng chkRng(1313);
    TModelParams params;
    yint vocabSize = data->GetStats().VocabSize;
    yint modelFlags = 0;
    // modelFlags |= MPF_RNN_CROSS;
    // modelFlags |= MPF_DISABLE_TUNE_EMBED;
    // modelFlags |= MPF_DISABLE_TUNE_LAYERS;
    // modelFlags |= MPF_DISABLE_TUNE_FINAL_LAYER;
    // TString modelDescrStr = "e384d0w64"; // embed + final
    // TString modelDescrStr = "e384d1h3ffn3w64"; // TP test
    // TString modelDescrStr = "e384d1h3ffn3moes5moe9w64"; // TP moe test
    TString modelDescrStr = "e384d1h3ffn3xp512moes3moe9w64"; // TP moe test, xp512
    // TString modelDescrStr = "e128h1d1w64";
    // TString modelDescrStr = "e128d1ffn2moe4w64"; // moe test
    // TString modelDescrStr = "e128xp256d1ffn2moe4w64"; // moe larger expert
    // TString modelDescrStr = "e384h2d1ffn2w64"; // default
    // TString modelDescrStr = "e384h2d4ffn2moe4w64";
    // TString modelDescrStr = "e128d6w64";
    TModelDescr modelDescr;
    InitModelDescr(&modelDescr, modelDescrStr, vocabSize, vocabSize, modelFlags);
    InitModel(&params, chkRng, modelDescr, COMBINER_INIT_RANDOM, data->GetStats().Bias);
    // Serialize(IO_READ, GetHomeDir() + "eden_gpt_134k.bin", params);
    // params.ResetGrad();

    TIntrusivePtr<IModel> refModel = CreateCpuTransformer(params, NODE_COUNT); // CPU
    //TIntrusivePtr<IModel> refModel = CreateLocalTransformer(params, 1, NODE_COUNT, TModelSplit()); // GPU
    TIntrusivePtr<IModelInstance> refCtx = refModel->GetInstance(0);

    yint tmInstanceCount = 1;
    //yint tmInstanceCount = 2;
    //TModelSplit msplit = SplitModel(modelDescr.GetDepth(), 1, 1, 1);
    //TModelSplit msplit = SplitModel(modelDescr.GetDepth(), 3, 1, 1); // TP
    //TModelSplit msplit = SplitModel(modelDescr.GetDepth(), 1, 3, 1); // PP
    TModelSplit msplit = SplitModel(modelDescr.GetDepth(), 3, 2, 1); // TP * PP
    yint deviceCount = msplit.GetLocationCount() * msplit.TP * tmInstanceCount;
    Y_VERIFY(deviceCount < MAX_NUM_DEVICES);
    //TIntrusivePtr<IModel> gpuModel = CreateHostMatrixTransformer(params, deviceCount, NODE_COUNT);
    TIntrusivePtr<IModel> gpuModel = CreateLocalTransformer(params, deviceCount, NODE_COUNT, msplit);
    //
    TIntrusivePtr<IModelInstance> gpuCtx = gpuModel->GetInstance(tmInstanceCount - 1);

    TVector<TFragment> fragArr;
    data->SampleFragments(IDataSource::TRAIN, 1313, CHECK_BATCH_SIZE, CHECK_FRAG_LEN, &fragArr);
    TVector<TVector<TFragment>> mbFragArr;
    mbFragArr.push_back(fragArr);

    MakeTest(fragArr, refCtx);
    MakeTest(fragArr, gpuCtx);

    TVector<TVector<float>> refPredArr;
    refCtx->ComputeFragmentPredictions(&refPredArr);
    TVector<TVector<float>> gpuPredArr;
    gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

    int t = 15;
    //int t = 0;
    for (yint k = 0; k < 5; ++k) {
        DebugPrintf("%g - %g\n", refPredArr[t][k], gpuPredArr[t][k]);
    }
    DebugPrintf("\nDiff %g bp\n", CalcDiff(refPredArr, gpuPredArr) * 10000);

    TXRng refRng = chkRng;
    TTrainingStep largeStep = dc.Step;
    largeStep.ScaleRate(10);
    MakeTrain(refRng, mbFragArr, dc.TokenDrop, refCtx);
    refModel->Backprop(largeStep, GRADIENT_APPLY);
    refCtx->ComputeFragmentPredictions(&refPredArr);

    for (yint m = 0; m < tmInstanceCount; ++m) {
        TXRng gpuRng = chkRng;
        MakeTrain(gpuRng, mbFragArr, dc.TokenDrop, gpuModel->GetInstance(m));
    }
    gpuModel->Backprop(largeStep, GRADIENT_APPLY);
    gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

    DebugPrintf("\nAfter backprop\n");
    for (yint k = 0; k < 5; ++k) {
        DebugPrintf("%g - %g\n", refPredArr[t][k], gpuPredArr[t][k]);
    }
    DebugPrintf("\nDiff %g bp\n\n", CalcDiff(refPredArr, gpuPredArr) * 10000);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void CheckInferMatch(const TDescentConfig &dc, TPtrArg<IDataSource> data)
{
    const yint CHECK_BATCH_SIZE = 3;
    const yint CHECK_FRAG_LEN = 100;

    TXRng chkRng(1314);
    TModelParams params;
    yint vocabSize = data->GetStats().VocabSize;
    //yint modelFlags = 0;
    yint modelFlags = MPF_RNN_CROSS;
    //TString modelDescrStr = "e128h1d1w64";
    //TString modelDescrStr = "e128d1ffn2moe2w64"; // moe test
    TString modelDescrStr = "e384h2d1ffn2w64"; // default
    //TString modelDescrStr = "e384h2d4ffn2moe2w64";
    //TString modelDescrStr = "e512h2d6w64";
    //TString modelDescrStr = "e128d6w64";
    TVector<float> bias = data->GetStats().Bias;
    ClearPodArray(&bias, YSize(bias));
    TModelDescr modelDescr;
    InitModelDescr(&modelDescr, modelDescrStr, vocabSize, vocabSize, modelFlags);
    InitModel(&params, chkRng, modelDescr, COMBINER_INIT_RANDOM, bias);
    // Serialize(IO_READ, GetHomeDir() + "eden_gpt_134k.bin", params);

    TIntrusivePtr<IBatchInfer> infer = NCUDA_Transformer::CreateBatchInferencer(params, CHECK_BATCH_SIZE);

    TIntrusivePtr<IModel> gpuModel = CreateNoSoftmaxTransformer(params, CHECK_BATCH_SIZE * CHECK_FRAG_LEN);
    TIntrusivePtr<IModelInstance> gpuCtx = gpuModel->GetInstance(0);

    TVector<TFragment> fragArr;
    data->SampleFragments(IDataSource::TRAIN, 1313, CHECK_BATCH_SIZE, CHECK_FRAG_LEN, &fragArr);
    for (const TFragment &frag : fragArr) {
        Y_ASSERT(frag.GetLength() == CHECK_FRAG_LEN);
    }

    TVector<TVector<float>> gpuPredArr;
    MakeTest(fragArr, gpuCtx);
    gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

    float totalDiff = 0;
    float totalSum2 = 0;
    TVector<TBatchInferRequest> qArr;
    qArr.resize(YSize(fragArr));
    for (yint t = 0; t < CHECK_FRAG_LEN; ++t) {
        TKVcacheTracker &kvCache = infer->GetKVcache();
        for (yint batchId = 0; batchId < CHECK_BATCH_SIZE; ++batchId) {
            TBatchInferRequest &q = qArr[batchId];
            q.KVcache.Next(&kvCache);
            q.PrevToken = fragArr[batchId].Text[t];
        }
        
        TVector<TVector<float>> inferPredArr;
        infer->ComputeFragmentPredictions(qArr, &inferPredArr);

        float sampleDiff = 0;
        for (yint batchId = 0; batchId < CHECK_BATCH_SIZE; ++batchId) {
            const TVector<float> &gpu = gpuPredArr[batchId * CHECK_FRAG_LEN + t];
            const TVector<float> &infer = inferPredArr[batchId];
            sampleDiff += CalcDiffSqr(infer, gpu);
            totalSum2 += Dot(gpu, gpu);
        }
        totalDiff += sampleDiff;
        DebugPrintf("t = %g, avrg diff %g\n", t * 1., sqrt(sampleDiff / CHECK_BATCH_SIZE));
    }
    DebugPrintf("\navrg diff %g, relative diff %g bp\n", sqrt(totalDiff / CHECK_FRAG_LEN / CHECK_BATCH_SIZE), sqrt(totalDiff / totalSum2) * 1e4f);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void TrainModel(yint startIteration, yint deviceCount, const TModelSplit &msplit, bool cudaGradient, bool printIterTrainErr,
    const TTrainContext &trainCtx, TIntrusivePtr<TModelParamsHolder> pParams)
{
    const TDescentConfig &dc = trainCtx.GetDescentConfig();
    const THostBatchConfig &bc = trainCtx.GetBatchConfig();

#ifdef _MSC_VER
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    SetProcessAffinityMask(GetCurrentProcess(), 0xffff); // use P-cores, mask is cpu dependent
#endif

    // create model
    TIntrusivePtr<IModel> pModel = CreateTrainModel(pParams->Params, deviceCount, msplit, cudaGradient, bc.GetInstanceMaxNodeCount());
    pParams = 0;
    TIntrusivePtr<IModelInstance> pCtx = pModel->GetInstance(0);

    // TOFStream fTrainLog(GetHomeDir() + "train_log.txt");
    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    for (yint iter = startIteration; iter <= trainCtx.GetMaxIters(); ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            if (trainCtx.IsSaveModel()) {
                TIntrusivePtr<TModelParamsHolder> params = new TModelParamsHolder;
                pModel->GetParams(&params->Params);
                BackgroundSaveModel(params, GetHomeDir() + Sprintf("eden_gpt_%.8gk.bin", iter / 1000.));
            }
            float trainErr = CalcModelErr(trainCtx.GetScoreTrainBatches(), pCtx) * trainCtx.GetCompression();
            float testErr = CalcModelErr(trainCtx.GetScoreTestBatches(), pCtx) * trainCtx.GetCompression();
            float timePassed = NHPTimer::GetTimePassed(&tStart);
            if (testErr != 0) {
                DebugPrintf("iter %.8gk, %g sec, train err %g, test err %g\n", iter / 1000., timePassed, trainErr, testErr);
                fflush(0);
            } else {
                DebugPrintf("iter %.8gk, %g sec, train err %g\n", iter / 1000., timePassed, trainErr);
                fflush(0);
            }
            // fTrainLog << trainErr << "\t" << testErr << Endl;
        }

        ui64 rngSeed = (iter + 0xbadf00d) * 0x39ef28172812ull;
        TVector<TFragment> fragArr;
        trainCtx.SampleTrainBatches(rngSeed, &fragArr);

        TXRng iterRng(iter);
        BackpropBatch(iterRng, dc, bc, trainCtx.GetStep(iter), fragArr, pModel);

        if (printIterTrainErr) {
            // adds Stream.Sync(), prohibits cpu/gpu compute overlap
            DebugPrintf("iter %g, train err %g\n", iter + 0., pModel->GetAvrgTrainErr()); fflush(0);
        }

        //printf("Iter %.8gk\n", iter / 1000.);
        //TestReproducibility(trainCtx, pCtx.Get(), iterRng, fragArr);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class TTrainScriptParser
{
    TTrainModelConfigParser TrainCfg;
    yint DeviceCount = 1;
    yint StartIteration = 0;
    bool SaveModel = true;
    yint MaxIters = 2000000;
    yint EvalInterval = 1000;
    yint EvalBatchCount = 20;
    bool PrintIterTrainErr = false;
    yint LimitNodeCount = 0;
    bool BiasReset = false;
    bool CudaGradient = false;
    yint TP = 1;
    yint PP = 1;
    yint MB = 1;

public:
    TTrainScriptParser(yint limitNodeCount) : LimitNodeCount(limitNodeCount)
    {
        DeviceCount = GetCudaDeviceCount();
    }
    void ParseScript(const TVector<TConfigFile::TOp> &opArr, TIntrusivePtr<IDataSource> data, const TString &lmIndexDir)
    {
        Y_VERIFY(data.Get() != 0);
        for (yint ptr = 0; ptr < YSize(opArr); ++ptr) {
            const TConfigFile::TOp &op = opArr[ptr];
            if (op.Op == CFG_OP_ASSIGNMENT) {
                if (op.Dst == "MAX_ITERS") {
                    MaxIters = atof(op.Args[0].c_str());
                } else if (op.Dst == "DEVICE_COUNT") {
                    DeviceCount = atof(op.Args[0].c_str());
                    Y_VERIFY(DeviceCount >= 1 && DeviceCount < 100);
                } else if (op.Dst == "EVAL_INTERVAL") {
                    EvalInterval = atof(op.Args[0].c_str());
                } else if (op.Dst == "EVAL_BATCH_COUNT") {
                    EvalBatchCount = atof(op.Args[0].c_str());
                } else if (op.Dst == "SAVE_MODEL") {
                    SaveModel = IsYes(op.Args[0]);
                } else if (op.Dst == "PRINT_ITER_TRAIN_ERR") {
                    PrintIterTrainErr = IsYes(op.Args[0]);
                } else if (op.Dst == "BIAS_RESET") {
                    BiasReset = IsYes(op.Args[0]);
                } else if (op.Dst == "CUDA_GRADIENT") {
                    CudaGradient = IsYes(op.Args[0]);
                } else if (op.Dst == "TP") {
                    TP = atoll(op.Args[0].c_str());
                } else if (op.Dst == "PP") {
                    PP = atoll(op.Args[0].c_str());
                } else if (op.Dst == "MB") {
                    MB = atoll(op.Args[0].c_str());
                } else if (TrainCfg.ParseScriptOp(op, data)) {
                    ;
                } else {
                    DebugPrintf("unknown config variable %s\n", op.Dst.c_str());
                }

            } else if (op.Op == CFG_OP_CALL) {
                if (op.Dst == "load_checkpoint") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    StartIteration = atoi(op.Args[0].c_str());
                    DebugPrintf("Load checkpoint %gk\n", StartIteration / 1000.);
                    TrainCfg.StartParams = new TModelParamsHolder();
                    TString fName = GetHomeDir() + Sprintf("eden_gpt_%.8gk.bin", StartIteration / 1000.);
                    Serialize(IO_READ, fName, TrainCfg.StartParams->Params);
                    Y_VERIFY(!TrainCfg.StartParams->Params.IsEmpty());

                    // process ops
                } else if (op.Dst == "train" || op.Dst == "net_train" || op.Dst == "ib_train") {
                    Y_VERIFY(!TrainCfg.StartParams->Params.IsEmpty());
                    yint microBatchCount = MB;
                    yint modelInstanceCount = DeviceCount / TP / PP;
                    if (DeviceCount != TP * PP * modelInstanceCount) {
                        DebugPrintf("Device count should be divisible by TP\n");
                        abort();
                    }
                    //(modelType == TMODEL_CUDA_TP) ? 1 : DeviceCount;
                    TDescentConfig dc = TrainCfg.MakeDescentConfig();
                    TTrainContext trainCtx(data, dc, modelInstanceCount, microBatchCount, LimitNodeCount, SaveModel, MaxIters, EvalInterval);

                    float modelSize = CountModelSize(TrainCfg.StartParams->Params) / 1000000.;
                    float activeModelSize = CountActiveModelSize(TrainCfg.StartParams->Params) / 1000000.;
                    TString szModelSize = (modelSize == activeModelSize) ? Sprintf("%gM", modelSize) : Sprintf("%gM of %gM", activeModelSize, modelSize);
                    DebugPrintf("%s %s %s 0x%x, size %s\n",
                        GetModelDescrString(TrainCfg.StartParams->Params.GetModelDescr()).c_str(),
                        dc.GetTrainConfig().c_str(),
                        dc.GetDropConfig().c_str(),
                        (int)TrainCfg.StartParams->Params.ModelDescr.Flags,
                        szModelSize.c_str());

                    // create batches for train & test score compute, can use different sizes
                    THostBatchConfig bc = trainCtx.GetBatchConfig();
                    trainCtx.MakeScoreBatches(EvalBatchCount);

                    // keep train params, split model
                    TModelSplit msplit;
                    {
                        TModelParams &params = TrainCfg.StartParams->Params;
                        params.ModelDescr.FragLen = dc.TrainFragLen;
                        if (BiasReset) {
                            ClearPodArray(&params.Bias, YSize(params.Bias));
                        }
                        msplit = SplitModel(params.ModelDescr.GetDepth(), TP, PP, microBatchCount);
                    }

                    if (op.Dst == "train") {
                        TrainModel(
                            StartIteration, DeviceCount, msplit, CudaGradient, PrintIterTrainErr, trainCtx, TrainCfg.StartParams.Release());
                    } else if (op.Dst == "net_train" || op.Dst == "ib_train") {
                        Y_VERIFY(YSize(op.Args) == 1);
                        TVector<TString> workerArr;
                        ReadNonEmptyLines(&workerArr, op.Args[0]);
                        if (op.Dst == "net_train") {
                            Y_VERIFY(TP == 1);
                            Y_VERIFY(PP == 1);
                            Y_VERIFY(MB == 1);
                            NNetTrain::RunMaster(StartIteration, DeviceCount, workerArr, trainCtx, TrainCfg.StartParams.Release());
                        } else {
                            NNetIbTrain::RunMaster(StartIteration, DeviceCount, msplit, workerArr, trainCtx, TrainCfg.StartParams.Release());
                        }
                    } else {
                        Y_ASSERT(0);
                    }

                } else if (op.Dst == "compute_exact_test") {
                    TModelParams params;
                    if (op.Args.empty()) {
                        Y_VERIFY(TrainCfg.StartParams.Get() && !TrainCfg.StartParams->Params.IsEmpty());
                        params = TrainCfg.StartParams->Params;
                    } else {
                        yint finishIter = atoi(op.Args[0].c_str());
                        yint iterInterval = YSize(op.Args) > 1 ? atoi(op.Args[1].c_str()) : 0;
                        yint iterStep = YSize(op.Args) > 2 ? atoi(op.Args[2].c_str()) : 1000;
                        ComputeAverageModel(&params, finishIter, iterInterval, iterStep);
                    }
                    ComputeExactTest(data, params);

                } else if (op.Dst == "compute_choice_score") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    Y_VERIFY(TrainCfg.StartParams.Get() && !TrainCfg.StartParams->Params.IsEmpty());
                    yint fragmentStartToken = data->GetStats().FragmentStartToken;
                    yint docStartToken = data->GetStats().DocStartToken;
                    ComputeChoiceScore(TrainCfg.StartParams->Params, op.Args[0], docStartToken, fragmentStartToken, lmIndexDir);

                } else if (op.Dst == "check_cpu_gpu_match") {
                    TDescentConfig dc = TrainCfg.MakeDescentConfig();
                    CheckCpuGpuMatch(dc, data);

                } else if (op.Dst == "check_infer_match") {
                    TDescentConfig dc = TrainCfg.MakeDescentConfig();
                    CheckInferMatch(dc, data);

                } else if (TrainCfg.ParseScriptOp(op, data)) {

                } else {
                    DebugPrintf("unknown function %s\n", op.Dst.c_str());
                    abort();
                }
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
extern yint MatrixAddWorkerThreadCount;

void TestMatMulFp16();
void TestMatMulInt8();
void TestMatMulFp8();
void TestAttFp16();
void TestAttGradKfp16();
void TestCudaSort();
void TestLMatch();
void RunMultideviceTest();
// void Repack();


int main(int argc, char **argv)
{
    //TestMatMulFp16();
    //TestMatMulInt8();
    //TestMatMulFp8();
    //TestAttFp16();
    //TestAttGradKfp16();
    //TestCudaSort();
    //Repack();
    //GenerateArithmetic();
    //GenerateArithmetic97();
    //TestLMatch();
    // RunMultideviceTest();
    // return 0;

    yint limitNodeCount = 24 * 1024;
    bool useInfiniband = true;
    TOpt cmdline("d:s:n:w:t:x:", argc, argv);
    TString workerPort = "";
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "s") {
            //DebugPrintf("Train script %s\n", param.Args[0].c_str());
            TVector<char> cfg;
            Y_VERIFY(ReadWholeFile(param.Args[0], &cfg));
            cfg.push_back(0);
            TRAIN_SCRIPT = cfg.data();
        } else if (param.Name == "d") {
            //DebugPrintf("Datasource script %s\n", param.Args[0].c_str());
            TVector<char> cfg;
            Y_VERIFY(ReadWholeFile(param.Args[0], &cfg));
            cfg.push_back(0);
            DATA_SCRIPT = cfg.data();
        } else if (param.Name == "n") {
            limitNodeCount = atoi(param.Args[0].c_str());
        } else if (param.Name == "w") {
            workerPort = param.Args[0];
        } else if (param.Name == "t") {
            MatrixAddWorkerThreadCount = atoi(param.Args[0].c_str());
        } else if (param.Name == "x") {
            useInfiniband = IsYes(param.Args[0]);
        }
    }
    if (!workerPort.empty() && workerPort != "master") {
        if (useInfiniband) {
            NNetIbTrain::RunWorker(atoi(workerPort.c_str()));
        } else {
            NNetTrain::RunWorker(atoi(workerPort.c_str()));
        }
        return 0;
    }

    // load data
    TString lmIndexDir;
    TIntrusivePtr<IDataSource> data = CreateDataSource(DATA_SCRIPT, &lmIndexDir);
    if (data.Get() == 0) {
        DebugPrintf("no dataset no train\n");
        return 0;
    }

    // train script
    TConfigFile trainCfg;
    ParseConfig(&trainCfg, TRAIN_SCRIPT);
    TTrainScriptParser trainScript(limitNodeCount);
    trainScript.ParseScript(trainCfg.OpArr, data, lmIndexDir);

    return 0;
}
