#pragma once

namespace NCuda
{
class TIntDivision
{
    int d;
    int M;
    int s;
    int n_add_sign;

public:
    TIntDivision(int arg) : d(arg)
    {
        // Hacker's Delight, Second Edition, Chapter 10, Integer Division By Constants
        if (d == 1) {
            M = 0;
            s = -1;
            n_add_sign = 1;
            return;
        } else if (d == -1) {
            M = 0;
            s = -1;
            n_add_sign = -1;
            return;
        }

        int p;
        unsigned int ad, anc, delta, q1, r1, q2, r2, t;
        const unsigned two31 = 0x80000000;
        ad = (d == 0) ? 1 : abs(d);
        t = two31 + ((unsigned int)d >> 31);
        anc = t - 1 - t % ad;
        p = 31;
        q1 = two31 / anc;
        r1 = two31 - q1 * anc;
        q2 = two31 / ad;
        r2 = two31 - q2 * ad;
        do {
            ++p;
            q1 = 2 * q1;
            r1 = 2 * r1;
            if (r1 >= anc) {
                ++q1;
                r1 -= anc;
            }
            q2 = 2 * q2;
            r2 = 2 * r2;
            if (r2 >= ad) {
                ++q2;
                r2 -= ad;
            }
            delta = ad - r2;
        } while (q1 < delta || (q1 == delta && r1 == 0));
        M = q2 + 1;
        if (d < 0) {
            M = -M;
        }
        s = p - 32;

        if ((d > 0) && (M < 0)) {
            n_add_sign = 1;
        } else if ((d < 0) && (M > 0)) {
            n_add_sign = -1;
        } else {
            n_add_sign = 0;
        }
    }

    // return arg / *this
    __device__ __forceinline__ int Div(int arg) const
    {
        int q;
        asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(M), "r"(arg));
        q += arg * n_add_sign;
        if (s >= 0) {
            q >>= s;
            q += (((unsigned int)q) >> 31);
        }
        return q;
    }
    __device__ __forceinline__ int DivCeil(int arg) const
    {
        return Div(arg + d - 1);
    }

    __device__ __forceinline__ operator int() const
    {
        return d;
    }
};
}
