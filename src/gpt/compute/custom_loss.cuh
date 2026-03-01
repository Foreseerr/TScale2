#pragma once
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_graph.cuh>


namespace NCUDA_Transformer
{
using namespace NCuda;

struct ICustomLoss : public TThrRefBase
{
    virtual void Allocate(yint deviceId, yint len) = 0; // not thread safe, should be called from single thread
    virtual void CopyDataToDevice(yint deviceId, TStream &stream) = 0;
    virtual void ComputeGradient(yint deviceId, TPtrArg<TGraph> c, TKernelParameter<int> &len, int logitBufWidth, int logitBufHeightBlock,
        TCuda2DArray<half> *pLogitBuf) = 0;
};
}
