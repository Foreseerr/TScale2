#pragma once
#include <lib/math/linear.h>


const yint TRACK_KOEF_COUNT = 8;

// tracks are approximated with linear combination of base tracks
struct TTrackParam
{
    double Koefs[TRACK_KOEF_COUNT];

    TTrackParam() { Clear(); }
    void Scale(double mult)
    {
        for (yint z = 0; z < TRACK_KOEF_COUNT; ++z) {
            Koefs[z] *= mult;
        }
    }
    void AddScaled(const TTrackParam &a, double mult)
    {
        for (yint z = 0; z < TRACK_KOEF_COUNT; ++z) {
            Koefs[z] += a.Koefs[z] * mult;
        }
    }
    void Assign(const TVector<double> &koefs)
    {
        Y_ASSERT(YSize(koefs) == TRACK_KOEF_COUNT);
        for (yint z = 0; z < TRACK_KOEF_COUNT; ++z) {
            Koefs[z] = koefs[z];
        }
    }
    TVector<double> MakeKoefs() const
    {
        TVector<double> res;
        res.resize(TRACK_KOEF_COUNT);
        for (yint z = 0; z < TRACK_KOEF_COUNT; ++z) {
            res[z] = Koefs[z];
        }
        return res;
    }
    void Clear() { Zero(*this); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TTrackInfo
{
    double DT = 0;
    TVector<TVector<double>> TrackHistory;
    SAVELOAD(DT, TrackHistory);
};


struct TTrackApproximator
{
    TVector<TVector<double>> BaseTracks;
    TArray2D<double> Cov1;

    void ComputeCov1()
    {
        yint sz = YSize(BaseTracks);
        TArray2D<double> cov;
        cov.SetSizes(sz, sz);
        for (yint y = 0; y < sz; ++y) {
            for (yint x = y; x < sz; ++x) {
                double dp = Dot(BaseTracks[y], BaseTracks[x]);
                cov[y][x] = dp;
                cov[x][y] = dp;
            }
            cov[y][y] *= 1.00000001; // more stable results
        }
        Cov1 = cov;
        //InvertRobust(&Cov1);
        InvertMatrix(&Cov1);
    }
    void ComputeKoefs(const TVector<double> &track, TVector<double> *res) const
    {
        yint sz = Cov1.GetXSize();
        TVector<double> dpArr;
        dpArr.resize(sz);
        for (yint i = 0; i < sz; ++i) {
            dpArr[i] = Dot(track, BaseTracks[i]);
        }
        Mul(res, Cov1, dpArr);
    }
    void ComputeApprox(const TVector<double> &koefs, TVector<double> *res) const
    {
        yint sz = YSize(koefs);
        Y_ASSERT(sz == Cov1.GetXSize());
        yint len = YSize(BaseTracks[0]);
        ClearPodArray(res, len);
        for (yint k = 0; k < sz; ++k) {
            const TVector<double> &b = BaseTracks[k];
            double mult = koefs[k];
            for (yint t = 0; t < len; ++t) {
                (*res)[t] += b[t] * mult;
            }
        }
    }
    double ComputeValue(const TVector<double> &koefs, yint t) const
    {
        yint sz = YSize(koefs);
        Y_ASSERT(sz == Cov1.GetXSize());
        double x = 0;
        for (yint k = 0; k < sz; ++k) {
            const TVector<double> &b = BaseTracks[k];
            x += b[t] * koefs[k];
        }
        return x;
    }
    double CalcLastValue(const TTrackParam &p) const
    {
        double res = 0;
        for (yint z = 0; z < TRACK_KOEF_COUNT; ++z) {
            res += BaseTracks[z].back() * p.Koefs[z];
        }
        return res;
    }
    yint GetLen() const { return YSize(BaseTracks[0]); }
    yint GetBaseCount() const { return YSize(BaseTracks); }
};


void MakeMyBase(TTrackApproximator *res, yint len, double dt);
void MakeOptimalBase(TTrackApproximator *res, yint len, yint sz, const TVector<TVector<double>> &allTracks);
void ApproximateTracks();


