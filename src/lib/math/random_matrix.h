#pragma once
#include "linear.h"


template<class T, class TRnd>
inline void MakeRandomPermutation(TArray2D<T> *res, TArray2D<T> *res1, yint sz, TRnd &&rnd)
{
    TVector<int> cc;
    for (yint k = 0; k < sz; ++k) {
        cc.push_back(k);
    }
    Shuffle(cc.begin(), cc.end(), rnd);
    res->SetSizes(sz, sz);
    res->FillZero();
    *res1 = *res;
    for (yint k = 0; k < sz; ++k) {
        (*res)[k][cc[k]] = 1;
        (*res1)[cc[k]][k] = 1;
    }
}


template<class TRng>
inline void BuildRandomOrthonormalMatrix(TArray2D<double> *res, yint featureCount, yint basisSize, TRng &rng)
{
    res->SetSizes(featureCount, basisSize);
    for (yint b = 0; b < basisSize; ++b) {
        TVector<double> dst;
        dst.resize(featureCount);
        for (yint f = 0; f < featureCount; ++f) {
            dst[f] = GenNormal(rng);
        }
        Normalize(&dst);
        for (yint z = 0; z < b; ++z) {
            double dot = 0;
            for (yint f = 0; f < featureCount; ++f) {
                dot += dst[f] * (*res)[z][f];
            }
            for (yint f = 0; f < featureCount; ++f) {
                dst[f] -= dot * (*res)[z][f];
            }
            Normalize(&dst);
        }
        for (yint f = 0; f < featureCount; ++f) {
            (*res)[b][f] = dst[f];
        }
    }
}


template<class TRng>
void MakeRandomReflection(TArray2D<double> *res, yint sz, TRng &rng)
{
    *res = MakeE<double>(sz);
    TVector<double> dir;
    dir.resize(sz);
    for (yint f = 0; f < sz; ++f) {
        dir[f] = GenNormal(rng);
    }
    Normalize(&dir);
    for (yint y = 0; y < sz; ++y) {
        for (yint x = 0; x < sz; ++x) {
            (*res)[y][x] -= 2 * dir[y] * dir[x];
        }
    }
}


template<class TRng>
void MakeRandomRotation(TArray2D<double> *res, yint sz, TRng &rng)
{
    TArray2D<double> a, b;
    MakeRandomReflection(&a, sz, rng);
    MakeRandomReflection(&b, sz, rng);
    MatrixMult(a, b, res);
}


