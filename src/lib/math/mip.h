#pragma once
#include "linear.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
// using MIP, older history is stored in aggregated form
template <class TElement, int MAX_LAYER_SIZE>
class TGroupedHistory
{
    struct TLayer
    {
        TVector<TIntrusivePtr<TElement>> History;
    };

private:
    TVector<TLayer> LayerArr;

private:
    void CombineLayer(yint d)
    {
        Y_VERIFY(d + 1 < YSize(LayerArr));
        TLayer &layer = LayerArr[d];
        yint len = YSize(layer.History);
        if (len > MAX_LAYER_SIZE) {
            TIntrusivePtr<TElement> sum = layer.History[len - 1];
            sum->Add(*layer.History[len - 2]);
            layer.History.resize(len - 2);
            Add(d + 1, sum);
        }
    }

    void Add(yint d, TPtrArg<TElement> element)
    {
        TLayer &layer = LayerArr[d];
        layer.History.insert(layer.History.begin(), element.Get());
        CombineLayer(d);
    }

public:
    TGroupedHistory()
    {
        Init();
    }

    void Init()
    {
        LayerArr.resize(0);
        LayerArr.resize(100); // 2^100 is sufficient length
    }

    void Add(TIntrusivePtr<TElement> element)
    {
        Add(0, element);
    }

    template <class TFunc>
    void WalkHistory(TFunc func) const
    {
        yint len = 1;
        yint step = 1;
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            const TLayer &layer = LayerArr[d];
            if (layer.History.empty()) {
                break;
            }
            for (yint k = 0; k < YSize(layer.History); ++k) {
                yint dt = len + step / 2; // mid interval
                func(dt, *layer.History[k]);
                len += step;
            }
            step *= 2;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TExpWeightMIP
{
    float Alpha = 0;
    TVector<float> WeightArr;
public:
    TExpWeightMIP(float alpha) : Alpha(alpha) {}
    float GetWeight(yint dt)
    {
        Y_ASSERT(dt >= 0);
        while (dt >= YSize(WeightArr)) {
            yint sz = YSize(WeightArr);
            WeightArr.push_back(exp(log(sz * 1.) * Alpha));
        }
        return WeightArr[dt];
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TFloatHistoryMIP
{
    enum {
        MAX_LAYER_SIZE = 8
    };

    struct TElement : public TThrRefBase
    {
        float Val = 0;

        TElement(const float val) : Val(val) {}
        void Add(const TElement &x) { Val += x.Val; }
    };

private:
    TGroupedHistory<TElement, MAX_LAYER_SIZE> History;
    yint Dim = 0;

public:
    void AddFloat(float val) { History.Add(new TElement(val)); }

    void ScaleFreeAvrg(TExpWeightMIP *pWeights, float *p, float *pSumWeight) const
    {
        float sum = 0;
        float sumWeight = 0;
        History.WalkHistory([&](yint dt, TElement &element) {
            float w = pWeights->GetWeight(dt);
            sum += element.Val * w;
            sumWeight += w;
        });
        *p = sum;
        *pSumWeight = sumWeight;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TVecHistoryMIP
{
    enum {
        MAX_LAYER_SIZE = 8
    };

    struct TElement : public TThrRefBase
    {
        TVector<float> Vals;

        TElement(const TVector<float> &vals) : Vals(vals)
        {
        }

        void Add(const TElement &x)
        {
            AddScaled(&Vals, x.Vals, 1);
        }
    };

private:
    TGroupedHistory<TElement, MAX_LAYER_SIZE> History;
    yint Dim = 0;

public:
    TVecHistoryMIP(yint dim) : Dim(dim)
    {
    }

    void AddVec(const TVector<float> &vec)
    {
        Y_ASSERT(YSize(vec) == Dim);
        History.Add(new TElement(vec));
    }

    void ScaleFreeAvrg(TExpWeightMIP *pWeights, TVector<float> *p) const
    {
        TVector<float> &sum = *p;
        ClearPodArray(&sum, Dim);
        float sumWeight = 0;
        History.WalkHistory([&](yint dt, TElement &element) {
            float w = pWeights->GetWeight(dt);
            AddScaled(&sum, element.Vals, w);
            sumWeight += w;
        });
        if (sumWeight > 0) {
            Scale(&sum, 1 / sumWeight);
        }
    }
};
