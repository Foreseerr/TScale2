#include "stat_test.h"
#include "gamma_func.h"



namespace NStatTest
{
double ProbNormalZeroAverage(const TGroupStat &gs)
{
    double n = gs.Count;
    double n2 = 0.5 * n;
    double res = 0
        + log(0.5)
        - n2 * log(PI)
        + lngamma(n2)
        - n2 * log(gs.Sum2);
    return res;
}


double ProbNormal(const TGroupStat &gs)
{
    double n = gs.Count;
    double n2 = 0.5 * n;
    double res = 0
        + log(0.5)
        - n2 * log(PI)
        + lngamma((n - 1) / 2.)
        + lngamma(1 / 2.)
        - 0.5 * log(n)
        - (n2 - 0.5) * log(gs.Sum2 - Sqr(gs.Sum) / n);
    return res;
}


double ProbNormal(const TVector<TGroupStat> &vec)
{
    double n = 0;
    double m = 0;
    double sum = 0;
    double prod = 0;
    for (const TGroupStat &gs : vec) {
        n += gs.Count;
        sum += gs.Sum2 - Sqr(gs.Sum) / gs.Count;
        m += 1;
        prod += log(gs.Count);
    }
    double n2 = 0.5 * n;
    double res = 0
        + log(0.5)
        - n2 * log(PI)
        + lngamma((n - m) / 2.)
        + m * lngamma(1 / 2.)
        - 0.5 * prod
        - (n2 - 0.5 * m) * log(sum);
    return res;
}

}
