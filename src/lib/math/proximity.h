#pragma once


void Shrink(float lambda, TVector<float> *p);
void ShrinkToPrev(const TVector<float> &prevVec, float lambda, TVector<float> *p);
