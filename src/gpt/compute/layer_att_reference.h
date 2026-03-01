#pragma once
#include <gpt/model_params/model_dim.h>
#include <gpt/att/att.h>


inline void ComputeDScale(const TModelDims &dims, const TArray2D<float> &valLookup, const TArray2D<float> &dValLookup, TArray2D<float> *pDScale)
{
    yint ttSum = dims.GetTTSum();
    yint headTTDim = dims.TTDim;
    yint len = valLookup.GetYSize();
    TArray2D<float> &dScale = *pDScale;
    dScale.SetSizes(len, dims.HeadCount);
    dScale.FillZero();
    for (yint t = 0; t < len; ++t) {
        for (yint k = 0; k < ttSum; ++k) {
            yint head = k / headTTDim;
            dScale[head][t] += dValLookup[t][k] * valLookup[t][k];
        }
    }
}


struct TAttentionComputer
{
    TArray2D<float> SumWeight;
    TArray2D<float> MaxDP;
    float AttDotScale = 0;
    float AlibiSlope = 0;

    TAttentionComputer(const TModelDims &dims, float alibiSlope, float attScaleMult) : AlibiSlope(alibiSlope)
    {
        AttDotScale = CalcDotScale(dims.QDim) * CalcAttentionMult() * attScaleMult;
    }

    float CalcDP(const TArray2D<float> &q, const TArray2D<float> &k,
        yint headQDim, yint qOffset, yint from, yint to)
    {
        float sum = 0;
        for (yint x = 0; x < headQDim; ++x) {
            sum += q[from][qOffset + x] * k[to][qOffset + x];
        }
        float dp = sum * AttDotScale;
        dp += GetAttentionDecay(from - to, AlibiSlope);
        return dp;
    }

    void ComputeValLookup(yint len, const TModelDims &dims,
        const TArray2D<float> &q, const TArray2D<float> &k, const TArray2D<float> &v,
        const TAttentionInfo &attInfo,
        TArray2D<float> *pValLookup)
    {
        yint ttSum = dims.GetTTSum();
        yint headQDim = dims.QDim;
        yint headTTDim = dims.TTDim;
        TVector<float> valLookup;
        pValLookup->SetSizes(ttSum, len);
        SumWeight.SetSizes(len, dims.HeadCount);
        MaxDP.SetSizes(len, dims.HeadCount);

        // compute weighted sum of val vectors
        for (yint head = 0; head < dims.HeadCount; ++head) {
            yint qOffset = head * headQDim;
            yint ttOffset = head * headTTDim;
            for (yint from = 0; from < len; ++from) {
                // find max dp
                float maxDP = 0;
                for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                    const TAttentionSpan &span = attInfo.Spans[attIndex];
                    for (yint to = span.Start; to <= span.Finish; ++to) {
                        float dp = CalcDP(q, k, headQDim, qOffset, from, to);
                        maxDP = fmaxf(maxDP, dp);
                    }
                }
                // sum
                float sumWeight = exp2(-maxDP); // initialize with zero vector of weight 1
                ClearPodArray(&valLookup, headTTDim);
                for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                    const TAttentionSpan &span = attInfo.Spans[attIndex];
                    for (yint to = span.Start; to <= span.Finish; ++to) {
                        float dp = CalcDP(q, k, headQDim, qOffset, from, to);
                        float w = exp2(dp - maxDP);
                        sumWeight += w;
                        for (yint x = 0; x < headTTDim; ++x) {
                            valLookup[x] += w * v[to][ttOffset + x];
                        }
                    }
                }
                Y_ASSERT(sumWeight > 0);
                SumWeight[head][from] = sumWeight;
                MaxDP[head][from] = maxDP;
                float sumWeight1 = 1 / sumWeight;
                for (yint x = 0; x < headTTDim; ++x) {
                    (*pValLookup)[from][ttOffset + x] = valLookup[x] * sumWeight1;
                }
            }
        }
    }

    struct TGradData
    {
        const TArray2D<float> &Q;
        const TArray2D<float> &K;
        const TArray2D<float> &V;
        const TArray2D<float> &DValLookupArr;
        const TArray2D<float> &DScaleArr;
        yint HeadQDim = 0;
        yint HeadTTDim = 0;

        TGradData(const TModelDims &dims,
            const TArray2D<float> &q, const TArray2D<float> &k, const TArray2D<float> &v,
            const TArray2D<float> &dValLookupArr, const TArray2D<float> &dScaleArr)
            : Q(q), K(k), V(v)
            , DValLookupArr(dValLookupArr)
            , DScaleArr(dScaleArr)
        {
            HeadQDim = dims.QDim;
            HeadTTDim = dims.TTDim;
        }
    };

    float CalcDDot(const TGradData &data,
        yint head, yint qOffset, yint ttOffset, yint from, yint to,
        float dp, float maxDP, float sumWeight)
    {
        float w = exp2(dp - maxDP) / sumWeight;
        Y_ASSERT(!isnan(w) && isfinite(w));

        float dW = 0;
        for (yint x = 0; x < data.HeadTTDim; ++x) {
            float dValLookup = data.DValLookupArr[from][ttOffset + x];
            float val = data.V[to][ttOffset + x]; // val2
            dW += dValLookup * val;
        }

        float dScale = data.DScaleArr[head][from];
        float dDot = w * (dW - dScale) * AttDotScale * LOG2;
        return dDot;
    }

    void GradQ(yint len, const TModelDims &dims,
        const TGradData &data,
        const TAttentionInfo &attInfo,
        TArray2D<float> *pDQ)
    {
        yint qSum = dims.GetQSum();
        yint ttSum = dims.GetTTSum();
        yint headQDim = dims.QDim;
        yint headTTDim = dims.TTDim;
        TArray2D<float> &dq = *pDQ;
        dq.SetSizes(qSum, len);
        dq.FillZero();
        for (yint head = 0; head < dims.HeadCount; ++head) {
            yint qOffset = head * headQDim;
            yint ttOffset = head * headTTDim;
            for (yint from = 0; from < len; ++from) {
                for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                    const TAttentionSpan &span = attInfo.Spans[attIndex];
                    float sumWeight = SumWeight[head][from];
                    float maxDP = MaxDP[head][from];
                    for (yint to = span.Start; to <= span.Finish; ++to) {
                        float dp = CalcDP(data.Q, data.K, data.HeadQDim, qOffset, from, to);
                        float dDot = CalcDDot(data, head, qOffset, ttOffset, from, to, dp, maxDP, sumWeight);
                        for (yint x = 0; x < headQDim; ++x) {
                            dq[from][qOffset + x] += dDot * data.K[to][qOffset + x];
                        }
                    }
                }
            }
        }
    }

    void GradK(yint len, const TModelDims &dims,
        const TGradData &data,
        const TAttentionInfo &revAttInfo,
        TArray2D<float> *pDK,
        TArray2D<float> *pDV)
    {
        yint qSum = dims.GetQSum();
        yint ttSum = dims.GetTTSum();
        TArray2D<float> &dk = *pDK;
        dk.SetSizes(qSum, len);
        dk.FillZero();
        TArray2D<float> &dv = *pDV;
        dv.SetSizes(ttSum, len);
        dv.FillZero();
        for (yint head = 0; head < dims.HeadCount; ++head) {
            yint qOffset = head * data.HeadQDim;
            yint ttOffset = head * data.HeadTTDim;
            for (yint to = 0; to < len; ++to) {
                for (yint attIndex = revAttInfo.SpanPtr[to]; attIndex < revAttInfo.SpanPtr[to + 1]; ++attIndex) {
                    const TAttentionSpan &span = revAttInfo.Spans[attIndex];
                    for (yint from = span.Start; from <= span.Finish; ++from) {
                        float sumWeight = SumWeight[head][from];
                        float maxDP = MaxDP[head][from];

                        float dp = CalcDP(data.Q, data.K, data.HeadQDim, qOffset, from, to);
                        float dDot = CalcDDot(data, head, qOffset, ttOffset, from, to, dp, maxDP, sumWeight);
                        float w = exp2(dp - maxDP) / sumWeight;

                        for (yint x = 0; x < data.HeadQDim; ++x) {
                            dk[to][qOffset + x] += dDot * data.Q[from][qOffset + x];
                        }

                        for (yint x = 0; x < data.HeadTTDim; ++x) {
                            dv[to][ttOffset + x] += w * data.DValLookupArr[from][ttOffset + x];
                        }
                    }
                }
            }
        }
    }
};


