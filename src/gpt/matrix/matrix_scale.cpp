#include "matrix_scale.h"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
THostModelMatrixScale::THostModelMatrixScale(yint sz)
{
    MatrixScaleHost.AllocateHost(sz);
}

void THostModelMatrixScale::SetScale(yint index, float val)
{
    Y_ASSERT(index >= 0 && index < MatrixScaleHost.GetSize());
    MatrixScaleHost.GetHostPtr()[index] = val;
}

float THostModelMatrixScale::GetScale(yint index)
{
    Y_ASSERT(index >= 0 && index < MatrixScaleHost.GetSize());
    return MatrixScaleHost.GetHostPtr()[index];
}

int THostModelMatrixScale::AddScale(float val)
{
    Y_VERIFY(IndexCount < MatrixScaleHost.GetSize());
    MatrixScaleHost.GetHostPtr()[IndexCount] = val;
    return IndexCount++;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
TCudaModelMatrixScale::TCudaModelMatrixScale(TPtrArg<THostModelMatrixScale> pScale) : HostMatrixScale(pScale.Get())
{
    DeviceMatrixScale.Allocate(pScale->GetSize());
}
}
