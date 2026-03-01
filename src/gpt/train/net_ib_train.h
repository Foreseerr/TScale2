#pragma once
#include <gpt/model_params/model_params.h>

class TTrainContext;
struct TModelSplit;

namespace NNetIbTrain
{
void RunWorker(yint port);
void RunMaster(yint startIteration, yint deviceCount, const TModelSplit &msplit, const TVector<TString> &workerAddrArr,
    const TTrainContext &trainCtx, TIntrusivePtr<TModelParamsHolder> pParams);
}
