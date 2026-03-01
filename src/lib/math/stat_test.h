#pragma once

namespace NStatTest
{
struct TGroupStat
{
    double Sum = 0;
    double Sum2 = 0;
    double Count = 0;
};

template <class T>
TGroupStat ComputeStats(const TVector<T> &vec)
{
    TGroupStat res;
    for (T x : vec) {
        res.Sum += x;
        res.Sum2 += x * x;
        res.Count += 1;
    }
    return res;
}

// stat tests with non-informative Jeffrey priors for dispersion (ln(sigma) in (-inf;inf)) and expectation (mu in (-inf;inf))
double ProbNormalZeroAverage(const TGroupStat &gs);
double ProbNormal(const TGroupStat &gs);
double ProbNormal(const TVector<TGroupStat> &vec);
}
