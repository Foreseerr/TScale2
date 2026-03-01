#pragma once
#include "cfg_precision.h"
#include <gpt/att/att.h>
#include <gpt/matrix/base_host.h>
#include <lib/cuda/multi_device_buf.h>


namespace NCuda
{
    class TCudaModelMatrixScale;
    class TCudaLayerBase;
    class TCudaMemoryAllocator;
    class TCudaMemoryPool;
    class TStream;
    template <class T>
    class ICudaModelMatrixBase;
    class TMultiDeviceBuffers;
    struct TCudaLayerPools;
}

struct TModelDescr;
struct TBatchNodes;
using NCuda::TCudaLayerBase;
using NCuda::TCudaLayerPools;
using NCuda::TCudaMemoryAllocator;
using NCuda::TCudaMemoryPool;
using NCuda::TCudaModelMatrixScale;
using NCuda::TMultiDeviceBuffers;
using NCuda::TStream;


struct TAttentionFB
{
    TAttentionInfo Att;
    TAttentionInfo RevAtt;

    void Assign(const TAttentionInfo &att)
    {
        Att = att;
        RevAtt = TransposeAttention(att);
    }
};


struct TCommonDataCPU
{
    TArray2D<float> RopeBuf;
    TVector<TAttentionFB> AttArr;
    TAttentionCrossShuffle Cross;

    void Init(const TModelDescr &modelDescr, yint nodeCount);
    void InitAttention(const TBatchNodes &nodes);
};


class TLayerBase : public TThrRefBase
{
protected:
    bool UpdateLayers = false;
    TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> MatrArr;

protected:
    TLayerBase(const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr);

public:
    virtual void ComputeForward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pState) = 0;
    virtual void ComputeBackward(const TCommonDataCPU &common, const TArray2D<float> &prevState, TArray2D<float> *pGrad) = 0;
    virtual TCudaLayerBase *CreateCudaLayer(TPtrArg<TCudaMemoryAllocator> cudaMem, TPtrArg<TMultiDeviceBuffers> multiBuffers,
        TStream &stream, yint deviceId, yint len, const TVector<TIntrusivePtr<NCuda::ICudaModelMatrixBase<TFastModelFloat>>> &matrArr,
        TCudaLayerPools &pools) = 0;
    TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &GetMatrArr() { return MatrArr; }
};
