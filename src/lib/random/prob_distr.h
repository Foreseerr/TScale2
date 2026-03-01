#pragma once

template <yint N>
struct TFixedProbDistr
{
    double Prob[N];

    TFixedProbDistr() { Clear(); }
    void Clear() { Zero(*this); }
    void Normalize()
    {
        double sum = 0;
        for (yint h = 0; h < N; ++h) {
            sum += Prob[h];
        }
        double mult = 1 / sum;
        for (yint h = 0; h < N; ++h) {
            Prob[h] *= mult;
        }
    }
    void Scale(double x)
    {
        for (yint pp = 0; pp < N; ++pp) {
            Prob[pp] *= x;
        }
    }
    void AddScaled(const TFixedProbDistr &x, double a)
    {
        for (yint pp = 0; pp < N; ++pp) {
            Prob[pp] += x.Prob[pp] * a;
        }
    }
    template <typename F>
    void Interpolate(const TFixedProbDistr &s, F func)
    {
        for (yint k = 0; k < N; ++k) {
            Prob[k] = func(Prob[k], s.Prob[k]);
        }
        Normalize();
    }
    template <class TRng>
    void MakeRandom(TRng &rng)
    {
        for (yint pp = 0; pp < N; ++pp) {
            Prob[pp] = -log(rng.GenRandReal3());
        }
        Normalize();
    }
    template <class TRng>
    void Mutate(TRng &rng, double noise)
    {
        for (yint pp = 0; pp < N; ++pp) {
            Prob[pp] *= exp(GenNormal(rng) * noise);
        }
        Normalize();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TProbDistr
{
    TVector<double> Prob;
    SAVELOAD(Prob);

    TProbDistr() {}
    TProbDistr(yint dim) { Prob.resize(dim, 0.); }
    void Clear()
    {
        yint dim = YSize(Prob);
        memset(Prob.data(), 0, sizeof(Prob[0]) * dim);
    }
    void Clear(yint dim)
    {
        Prob.resize(dim);
        Clear();
    }
    yint GetDim() const { return YSize(Prob); }
    void Normalize()
    {
        yint dim = YSize(Prob);
        double sum = 0;
        for (yint h = 0; h < dim; ++h) {
            sum += Prob[h];
        }
        double mult = 1 / sum;
        for (yint h = 0; h < dim; ++h) {
            Prob[h] *= mult;
        }
    }
    void AddScaled(const TProbDistr &x, double a)
    {
        yint dim = YSize(Prob);
        Y_ASSERT(dim == YSize(x.Prob));
        for (yint pp = 0; pp < dim; ++pp) {
            Prob[pp] += x.Prob[pp] * a;
        }
    }
    template <typename F>
    void Interpolate(const TProbDistr &x, F func)
    {
        yint dim = YSize(Prob);
        Y_ASSERT(dim == YSize(x.Prob));
        for (yint k = 0; k < dim; ++k) {
            Prob[k] = func(Prob[k], x.Prob[k]);
        }
        Normalize();
    }
    template <class TRng>
    void MakeRandom(TRng &rng, yint dim)
    {
        Prob.resize(dim);
        for (yint pp = 0; pp < dim; ++pp) {
            Prob[pp] = -log(rng.GenRandReal3());
        }
        Normalize();
    }
    template <class TRng>
    void Mutate(TRng &rng, double noise)
    {
        yint dim = YSize(Prob);
        for (yint pp = 0; pp < dim; ++pp) {
            Prob[pp] *= exp(GenNormal(rng) * noise);
        }
        Normalize();
    }
};
