#pragma once
#include "model.h"
#include <gpt/att/nodes_batch.h>
#include <gpt/data/data.h>
#include <gpt/att/sliding_window.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
// process set of fragments and init context
inline void MakeTrain(TXRng &rng, const TVector<TVector<TFragment>> &fragArr, float tokenDrop, TPtrArg<IModelInstance> pCtx)
{
    yint microBatchCount = YSize(fragArr);
    for (yint microBatchId = 0; microBatchId < microBatchCount; ++microBatchId) {
        TBatchNodes &nodes = pCtx->GetNodes(microBatchId);
        const TModelDescr &modelDescr = pCtx->GetModelDescr();
        InitLabelData(modelDescr, rng, tokenDrop, fragArr[microBatchId], ATT_GRAPH_TRAIN_LOSS, &nodes);
    }
    pCtx->Init(IModelInstance::INIT_BACKPROP);
}


inline void MakeTest(const TVector<TFragment> &fragArr, TPtrArg<IModelInstance> pCtx)
{
    TBatchNodes &nodes = pCtx->GetNodes(0);
    const TModelDescr &modelDescr = pCtx->GetModelDescr();
    InitLabelData(modelDescr, NopRng, 1., fragArr, ATT_GRAPH_TEST_LOSS, &nodes);
    pCtx->Init(IModelInstance::INIT_FWD);
}
