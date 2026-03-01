#pragma once
#include "batch_config.h"
#include <gpt/compute/model.h>
#include <gpt/train_config/train_config.h>
#include <lib/random/xrng.h>


void BackpropBatch(TXRng &rng, const TDescentConfig &dc, const THostBatchConfig &bc, const TTrainingStep &step,
    const TVector<TFragment> &fragArr, TPtrArg<IModel> model);
