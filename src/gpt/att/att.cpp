#include "att.h"


void SortAttentionSpans(TAttentionInfo *p)
{
    yint sampleCount = p->GetSampleCount();
    for (yint from = 0; from < sampleCount; ++from) {
        Sort(
            p->Spans.begin() + p->SpanPtr[from],
            p->Spans.begin() + p->SpanPtr[from + 1],
            [](const TAttentionSpan &a, const TAttentionSpan &b) { return a.Start < b.Start; }
        );
    }
}


static void AddSpans(TAttentionInfo *p, const TVector<TAttentionSpan> &vec)
{
    for (const TAttentionSpan &span : vec) {
        p->AddSpan(span);
    }
}


TAttentionInfo TransposeAttention(const TAttentionInfo &att)
{
    if (att.IsEmpty()) {
        return TAttentionInfo();
    }

    yint sampleCount = att.GetSampleCount();
    TVector<TVector<TAttentionSpan>> resSpans;
    resSpans.resize(sampleCount);
    for (yint from = 0; from < sampleCount; ++from) {
        for (yint k = att.SpanPtr[from]; k < att.SpanPtr[from + 1]; ++k) {
            const TAttentionSpan &span = att.Spans[k];
            for (yint to = span.Start; to <= span.Finish; ++to) {
                TVector<TAttentionSpan> *dst = &resSpans[to];
                if (!dst->empty() && dst->back().Finish == from - 1) {
                    dst->back().Finish = from;
                } else {
                    TAttentionSpan newSpan(from, from);
                    dst->push_back(newSpan);
                }
            }
        }
    }
    TAttentionInfo res;
    res.Init();
    for (yint to = 0; to < sampleCount; ++to) {
        AddSpans(&res, resSpans[to]);
        res.AddSample();
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void BuildCrossShuffle(const TAttentionCrossBatchInfo &cross, TAttentionCrossShuffle *pShuffle, TAttentionInfo *pAttSpans)
{
    TVector<int> offsets;
    int sum = 0;
    offsets.push_back(sum);
    for (int x : cross.GroupSize) {
        sum += x;
        offsets.push_back(sum);
    }
    // att spans
    yint groupCount = YSize(cross.GroupSize);
    yint sampleCount = YSize(cross.SampleGroup);
    pAttSpans->Init();
    for (yint g = 0; g < groupCount; ++g) {
        int beg = offsets[g];
        int fin = offsets[g + 1];
        TAttentionSpan span(beg, fin - 1);
        for (yint k = beg; k < fin; ++k) {
            pAttSpans->AddSpan(span);
            pAttSpans->AddSample();
        }
    }
    // shuffle
    pShuffle->FwdShuffle.resize(sampleCount);
    for (yint k = 0; k < sampleCount; ++k) {
        int groupId = cross.SampleGroup[k];
        pShuffle->FwdShuffle[offsets[groupId]++] = k;
    }
}