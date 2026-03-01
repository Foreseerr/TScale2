#include "batch_config.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
THostBatchConfig::THostBatchConfig(yint modelInstanceCount, yint microBatchCount, yint limitNodeCount, yint batchSize, yint fragLen)
    : BatchSize(batchSize), ModelInstanceCount(modelInstanceCount), MicroBatchCount(microBatchCount), FragLen(fragLen)
{
    if (fragLen > limitNodeCount) {
        DebugPrintf("train frag length %g does not fit into available %g nodes\n", fragLen * 1., limitNodeCount * 1.);
        fflush(0);
        abort();
    }
    yint trainFragPerMicroBatch = batchSize / ModelInstanceCount / MicroBatchCount;
    if (trainFragPerMicroBatch == 0 || trainFragPerMicroBatch * ModelInstanceCount * MicroBatchCount != batchSize) {
        DebugPrintf("suboptimal configuration, %g fragments per microbatch (%g fragments, %g instances, %g microbatch)\n", //
            trainFragPerMicroBatch * 1., batchSize * 1., ModelInstanceCount * 1., MicroBatchCount * 1.);
        fflush(0);
        abort();
    }
    yint maxFragPerStep = limitNodeCount / fragLen;
    Y_VERIFY(maxFragPerStep > 0);
    for (yint fragPerStep = Min<yint>(maxFragPerStep, trainFragPerMicroBatch); fragPerStep >= 1; --fragPerStep) {
        if ((trainFragPerMicroBatch % fragPerStep) == 0) {
            InstanceMicroBatchSize = fragPerStep;
            AccumulateSteps = trainFragPerMicroBatch / fragPerStep;
            TString strAccSteps = (AccumulateSteps > 1) ? Sprintf("%g accumulation steps, ", AccumulateSteps * 1.).c_str() : "";
            TString strModelInstanceCount = (ModelInstanceCount > 1) ? Sprintf("%g instances, ", ModelInstanceCount * 1.).c_str() : "";
            TString strMicroBatches = (MicroBatchCount > 1) ? Sprintf("%g micro batches, ", MicroBatchCount * 1.).c_str() : "";
            DebugPrintf("%s%s%s%g fragments per step, %g fragment legnth\n", //
                strAccSteps.c_str(), strModelInstanceCount.c_str(), strMicroBatches.c_str(), InstanceMicroBatchSize * 1., FragLen * 1.);
            fflush(0);
            return;
        }
    }
}


yint THostBatchConfig::GetInstanceMaxNodeCount() const
{
    return FragLen * InstanceMicroBatchSize;
}
