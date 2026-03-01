#pragma once
#include "layer.h"


TIntrusivePtr<TLayerBase> Create3LinAttLayer(const TModelDescr &modelDescr,
    const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr, float alibiSlope, yint attTypeId);

namespace NCuda
{
class TCudaInferLayerBase;
template <class T>
class TCudaInferModelMatrix;

TIntrusivePtr<TCudaInferLayerBase> Create3LinAttLayerInference(TStream &stream, const TModelDescr &modelDescr, float alibiSlope,
    const TAttentionType &attnType, yint len, yint kvCacheSize,
    const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr, TPtrArg<TCudaMemoryPool> pool);
}
