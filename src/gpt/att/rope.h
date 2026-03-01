#pragma once

void FillRopeBuf(TArray2D<float> *pRopeBuf, yint width, yint ropeLen);
void FillRopeBuf(TArray2D<float> *pRopeBuf, yint width, const TVector<yint> tArr);

void ApplyRope(const TArray2D<float> &ropeBuf, float ropeRotateDir, yint t, TVector<float> *p);
void ApplyRope(const TArray2D<float> &ropeBuf, float ropeRotateDir, TArray2D<float> *p);
