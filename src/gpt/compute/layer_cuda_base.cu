#include <util/pch.h>
#define KERNEL_UNIT "layer_cuda_base/"
#include "layer_cuda_base.cuh"
#include <gpt/att/rope.h>

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
TComputeParams::TComputeParams() : Len(), LenRound(RoundUp(Len, MM_TILE)) {}


void TComputeParams::Allocate(const TModelDescr &modelDescr, yint maxLen)
{
    Dims = modelDescr.Dims;
    if (modelDescr.GetCrossAttentionId() >= 0) {
        CrossAttnShuffle.Allocate(maxLen);
    }
}


void TComputeParams::InitRope(TStream &stream, const TModelDescr &modelDescr, int ropeLen)
{
    yint width = modelDescr.Dims.QDim;
    TArray2D<float> rope;
    FillRopeBuf(&rope, width, ropeLen);
    RopeBuf.Allocate(width, ropeLen);
    Put(stream, &RopeBuf, rope);
}


void TComputeParams::Init(TStream &stream, yint len)
{
    Len.Set(len);
    LenMoe.Set(RoundDown(len, MM_TILE) * Dims.MoeSelected + MM_TILE * Dims.MoeExpertCount);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TCudaLayerBase::CopyToDevice(TPtrArg<NCuda::TGraph> c)
{
    for (auto &mm : MatrArr) {
        mm->CopyToDevice(c);
    }
}
}
