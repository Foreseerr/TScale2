#include "layer.h"
#include <gpt/att/rope.h>
#include <gpt/att/nodes_batch.h>
#include <gpt/model_params/model_dim.h>


void TCommonDataCPU::Init(const TModelDescr &modelDescr, yint nodeCount)
{
    FillRopeBuf(&RopeBuf, modelDescr.Dims.QDim, nodeCount);

    yint attentionTypeCount = modelDescr.GetAttentionTypeCount();
    AttArr.resize(attentionTypeCount);
}


void TCommonDataCPU::InitAttention(const TBatchNodes &nodes)
{
    yint count = YSize(AttArr);
    for (yint wa = 0; wa < count; ++wa) {
        AttArr[wa].Assign(nodes.AttArr[wa]);
    }
    Cross = nodes.CrossShuffle;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
TLayerBase::TLayerBase(const TModelDescr &modelDescr, const TVector<TIntrusivePtr<IModelMatrixBase<TFastModelFloat>>> &matrArr)
    : UpdateLayers(!modelDescr.HasFlag(MPF_DISABLE_TUNE_LAYERS)), MatrArr(matrArr)
{
}
