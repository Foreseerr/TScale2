#pragma once
#include "layer.h"


TIntrusivePtr<TLayerBase> CreateMoELayer(
    const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr);
