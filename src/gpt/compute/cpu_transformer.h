#pragma once
#include "model.h"

namespace NCPU_Transformer
{
TIntrusivePtr<IModelImpl> CreateContext(TPtrArg<IModelStorage> modelStorage, TPtrArg<IModelOps> modelOps, yint nodeCount);
}
