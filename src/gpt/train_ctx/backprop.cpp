#include "backprop.h"
#include <gpt/compute/model_inst_init.h>


void BackpropBatch(TXRng &rng, const TDescentConfig &dc, const THostBatchConfig &bc, const TTrainingStep &step,
    const TVector<TFragment> &fragArr, TPtrArg<IModel> model)
{
    for (yint accStep = 0; accStep < bc.AccumulateSteps; ++accStep) {
        EAddToModel addToModel = (accStep == bc.AccumulateSteps - 1) ? GRADIENT_APPLY : GRADIENT_ACCUMULATE;
        yint base = accStep * bc.ModelInstanceCount * bc.MicroBatchCount * bc.InstanceMicroBatchSize;
        // provide train data to devices
        for (yint instanceId = 0; instanceId < bc.ModelInstanceCount; ++instanceId) {
            TVector<TVector<TFragment>> allDevFrags;
            for (yint mbId = 0; mbId < bc.MicroBatchCount; ++mbId) {
                yint mbBase = base + (instanceId * bc.MicroBatchCount + mbId) * bc.InstanceMicroBatchSize;
                TVector<TFragment> devFrags;
                for (yint k = 0; k < bc.InstanceMicroBatchSize; ++k) {
                    devFrags.push_back(fragArr[mbBase + k]);
                }
                allDevFrags.push_back(devFrags);
            }
            MakeTrain(rng, allDevFrags, dc.TokenDrop, model->GetInstance(instanceId));
        }
        // backprop
        model->Backprop(step, addToModel);
    }
}
