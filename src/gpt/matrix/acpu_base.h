#pragma once
#include "base_host.h"
#include <gpt/train_config/train_step.h>


namespace NCuda
{
template <class T>
class TCudaPOD;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMatrixBaseOps : public IBitDelta
{
    virtual void AttachOp(NCuda::TCudaPOD<int> currentIteration, const TVector<NCuda::TCudaPOD<int>> &cudaDeltaFlags,
        const TVector<NCuda::TCudaPOD<int>> &cudaAllowDelayedFlags) = 0;
    virtual bool IsDelayGradient() const = 0;
    virtual void AddDeviceToSumDelta(yint deviceId) = 0;
    virtual void AddDelta(const TTrainingStep &step) = 0;
    virtual bool AddBitDelta(const TTrainingStep &step) = 0;
    virtual void Convert() = 0;
};
