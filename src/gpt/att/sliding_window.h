#pragma once
#include "nodes_batch.h"
#include <gpt/data/bpe.h>
#include <gpt/model_params/model_dim.h>
#include <lib/random/xrng.h>


struct TFragment;

void InitModelDescr(TModelDescr *pRes, const TString &modelDescrStr, yint inputTokenCount, yint outputTokenCount, ui64 flags);

void AddLabels(const TModelDescr &modelDescr, TBatchLabels *p, TBPEToken token);

enum {
    ATT_GRAPH_TRAIN_LOSS,
    ATT_GRAPH_TEST_LOSS,
};

void InitLabelData(const TModelDescr &modelDescr, TXRng &rng, float tokenDrop,
    const TVector<TFragment> &fragArr, yint lossType,
    TBatchNodes *pNodes);

