#pragma once
#include "batch_config.h"
#include <gpt/data/bpe.h>
#include <gpt/data/data.h>
#include <gpt/compute/model.h>
#include <gpt/train_config/train_config.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/random/mersenne.h>
#include <lib/random/rand_utils.h>


double CalcModelErr(const TVector<TFragment> &fragArr, TPtrArg<IModelInstance> pCtx);
double CalcModelErr(const TVector<TVector<TFragment>> &batchArr, TPtrArg<IModelInstance> pCtx);
double CalcTargetLoss(const TVector<TVector<float>> &predArr, const TVector<TNodeTarget> &target);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TTrainContext
{
    TIntrusivePtr<IDataSource> Data;
    TDescentConfig DescentConfig;
    THostBatchConfig HostBatchConfig;
    TVector<TVector<TFragment>> ScoreTrainBatches;
    TVector<TVector<TFragment>> ScoreTestBatches;
    bool SaveModel = false;
    yint MaxIters = 1000;
    yint EvalInterval = 100;

    void AddBatches(TVector<TVector<TFragment>> *p, const TVector<TFragment> &arr, yint mbSize)
    {
        Y_ASSERT((YSize(arr) % mbSize) == 0);
        for (yint base = 0; base < YSize(arr); base += mbSize) {
            TVector<TFragment> ff;
            for (yint i = 0; i < mbSize; ++i) {
                ff.push_back(arr[base + i]);
            }
            p->push_back(ff);
        }
    }


public:
    TTrainContext(TPtrArg<IDataSource> data, const TDescentConfig &descent, yint modelInstanceCount, yint microBatchCount, yint limitNodeCount,
        bool saveModel, yint maxIters, yint evalInterval)
        : Data(data), DescentConfig(descent),
          HostBatchConfig(modelInstanceCount, microBatchCount, limitNodeCount, descent.TrainBatchSize, descent.TrainFragLen), SaveModel(saveModel),
          MaxIters(maxIters), EvalInterval(evalInterval)
    {
    }
    const TDescentConfig &GetDescentConfig() const { return DescentConfig; }
    const THostBatchConfig &GetBatchConfig() const { return HostBatchConfig; }
    const TVector<TVector<TFragment>> &GetScoreTrainBatches() const { return ScoreTrainBatches; }
    const TVector<TVector<TFragment>> &GetScoreTestBatches() const { return ScoreTestBatches; }
    float GetCompression() const { return Data->GetStats().Compression; }
    bool IsSaveModel() const { return SaveModel; }
    yint GetMaxIters() const { return MaxIters; }
    yint GetEvalInterval() const { return EvalInterval; }
    TTrainingStep GetStep(yint iter) const
    {
        return DescentConfig.GetStep(iter, MaxIters);
    }

    void MakeScoreBatches(yint batchCount)
    {
        const THostBatchConfig &bc = HostBatchConfig;
        if (Data->GetStats().HasTest) {
            yint chkRngSeed = 1313;
            for (yint k = 0; k < batchCount; ++k) {
                TVector<TFragment> batchArr;
                Data->SampleFragments(IDataSource::TEST, chkRngSeed++, bc.BatchSize, bc.FragLen, &batchArr);
                AddBatches(&ScoreTestBatches, batchArr, bc.InstanceMicroBatchSize);
            }
        }
        yint chkRngSeed = 31313;
        for (yint k = 0; k < batchCount; ++k) {
            TVector<TFragment> batchArr;
            Data->SampleFragments(IDataSource::TRAIN, chkRngSeed++, bc.BatchSize, bc.FragLen, &batchArr);
            AddBatches(&ScoreTrainBatches, batchArr, bc.InstanceMicroBatchSize);
        }
    }

    void SampleTrainBatches(yint rngSeed, TVector<TFragment> *pRes) const
    {
        Data->SampleFragments(IDataSource::TRAIN, rngSeed, DescentConfig.TrainBatchSize, DescentConfig.TrainFragLen, pRes);
    }
};
