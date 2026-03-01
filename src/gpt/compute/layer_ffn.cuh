#pragma once
#include "layer.h"


TIntrusivePtr<TLayerBase> CreateFFLayer(
    const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr);

namespace NCuda
{
class TCudaInferLayerBase;
template <class T>
class TCudaInferModelMatrix;

TIntrusivePtr<TCudaInferLayerBase> CreateFFLayerInference(const TModelDescr &modelDescr, yint len,
    const TVector<TIntrusivePtr<TCudaInferModelMatrix<TFastModelFloat>>> &matrArr, TPtrArg<TCudaMemoryPool> pool);
}
