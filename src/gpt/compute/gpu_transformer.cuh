#pragma once
#include "model.h"

namespace NCUDA_Transformer
{
struct ICustomLoss;
TIntrusivePtr<IModelImpl> CreateContext(
    TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, const TModelSplit &msplit, yint nodeCount);

TIntrusivePtr<IModelImpl> CreateContextNoSoftmax(TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount);

TIntrusivePtr<IModelImpl> CreateWithCustomLoss(
    TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount, TPtrArg<ICustomLoss> pLoss);
}
