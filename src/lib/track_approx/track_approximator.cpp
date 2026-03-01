#include "track_approximator.h"
#include <lib/math/eigen.h>


void MakeMyBase(TTrackApproximator *res, yint len, double dt)
{
    TVector<double> vec;
    vec.resize(len);

    for (yint t = 0; t < len; ++t) {
        vec[t] = t;
    }
    res->BaseTracks.push_back(vec);

    for (yint t = 0; t < len; ++t) {
        vec[t] = 1;
    }
    res->BaseTracks.push_back(vec);

    //const double expMultArr[] = { 1 }; // 1.83
    //const double expMultArr[] = { 1, 2 }; // 0.25
    //const double expMultArr[] = { 0.3, 1, 3 }; // 0.024
    //const double expMultArr[] = { 0.3, 1, 3, 9 }; // 0.003
    const double expMultArr[] = { 0.1, 0.3, 1, 3, 9, 27 };
    for (yint z = 0; z < ARRAY_SIZE(expMultArr); ++z) {
        double em = expMultArr[z];
        for (yint t = 0; t < len; ++t) {
            vec[t] = log(1 + exp((t + 1) * em * dt)) - log(2.);
        }
        res->BaseTracks.push_back(vec);
    }
    Y_VERIFY(YSize(res->BaseTracks) == TRACK_KOEF_COUNT);
}


void MakeOptimalBase(TTrackApproximator *res, yint len, yint sz, const TVector<TVector<double>> &allTracks)
{
    yint total = YSize(allTracks);
    TArray2D<double> cc;
    cc.SetSizes(total, total);
    for (yint y = 0; y < total; ++y) {
        for (yint x = y; x < total; ++x) {
            double dp = Dot(allTracks[y], allTracks[x]);
            cc[y][x] = dp;
            cc[x][y] = dp;
        }
    }
    TVector<double> eigenVals;
    TVector<TVector<double>> eigenVecs;
    NEigen::CalcEigenVectors(&eigenVals, &eigenVecs, cc);

    for (yint k = 0; k < sz; ++k) {
        const TVector<double> &ev = eigenVecs[YSize(eigenVals) - 1 - k];
        TVector<double> vec;
        ClearPodArray(&vec, len);
        for (yint trackId = 0; trackId < total; ++trackId) {
            double mult = ev[trackId];
            for (yint t = 0; t < len; ++t) {
                vec[t] += allTracks[trackId][t] * mult;
            }
        }
        res->BaseTracks.push_back(vec);
    }
}


void ApproximateTracks()
{
    TTrackInfo trackInfo;
    Serialize(IO_READ, "D:/zParamsTracks.bin", trackInfo);
    yint len = YSize(trackInfo.TrackHistory[0]);

    TTrackApproximator tapx;
    MakeMyBase(&tapx, len, trackInfo.DT);
    //MakeOptimalBase(&tapx, len, 2, trackInfo.TrackHistory); // 1.2
    //MakeOptimalBase(&tapx, len, 3, trackInfo.TrackHistory); // 0.09
    //MakeOptimalBase(&tapx, len, 4, trackInfo.TrackHistory); // 0.009
    //MakeOptimalBase(&tapx, len, 5, trackInfo.TrackHistory); // 0.0009
    //MakeOptimalBase(&tapx, len, 6, trackInfo.TrackHistory); // 0.0001
    tapx.ComputeCov1();

    double err = 0;
    for (const TVector<double> &track : trackInfo.TrackHistory) {
        TVector<double> koef;
        tapx.ComputeKoefs(track, &koef);
        TVector<double> approx;
        tapx.ComputeApprox(koef, &approx);
        Y_ASSERT(YSize(track) == YSize(approx));
        for (yint t = 0; t < YSize(track); ++t) {
            err += Sqr(track[t] - approx[t]);
        }
    }
    printf("Total approx err = %g\n", err);
    for (;;) {
        SchedYield();
    }
}
