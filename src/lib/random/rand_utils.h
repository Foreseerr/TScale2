#pragma once
#include "mersenne.h"
#include "random250.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
inline double Logit(double x)
{
    return log(x / (1 - x));
}

inline double Logistic(double x)
{
    return 1 / (1 + exp(-x));
}

// two independent evidences, equivalent to logistic add
inline double IndepMult(double a, double b)
{
    double p1 = a * b;
    double p0 = (1 - a) * (1 - b);
    return p1 / (p1 + p0);
}

inline double Shrink(double x, double a)
{
    if (x > 0) {
        return Max(0., x - a);
    } else {
        return Min(0., x + a);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TRng>
inline double GenNormal(TRng &rng)
{
    for(;;) {
        double x = rng.GenRandReal3() * 2 - 1;
        double y = rng.GenRandReal3() * 2 - 1;
        double r = x*x + y*y;
        if (r > 1 || r == 0)
            continue;
        double fac = sqrt(-2 * log(r) / r);
        return x * fac;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void Shuffle(TVector<T> *res)
{
    yint sz = YSize(*res);
    for (yint i = 0; i < sz - 1; ++i) {
        DoSwap((*res)[i], (*res)[i + r250n(sz - i)]);
    }
}

template <class T, class TRnd>
void Shuffle(T beg, T fin, TRnd &&rnd)
{
    for (T ptr = beg; ptr != fin; ++ptr) {
        DoSwap(*ptr, ptr[rnd.Uniform(fin - ptr)]);
    }
}

template <class T, class TRng>
void SelectRandomSubset(TVector<T> *p, yint count, TRng &&rng)
{
    yint sz = YSize(*p);
    if (count < sz) {
        for (yint k = 0; k < count; ++k) {
            DoSwap((*p)[k], (*p)[k + rng.Uniform(sz - k)]);
        }
        p->resize(count);
    }
}
