#include "nodes_batch.h"


void TBatchLabels::Init()
{
    LabelArr.resize(0);
    LabelPtr.resize(0);
    LabelPtr.push_back(0);
}


void TBatchLabels::AddSample(const TVector<TLabelIndex> &labels)
{
    LabelArr.insert(LabelArr.end(), labels.begin(), labels.end());
    LabelPtr.push_back(YSize(LabelArr));
}


void TBatchLabels::AddSample(TLabelIndex lbl)
{
    LabelArr.push_back(lbl);
    LabelPtr.push_back(YSize(LabelArr));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TBatchNodes::Init(yint attentionWidthCount, yint crossAttentionId)
{
    CrossAttentionId = crossAttentionId;
    Labels.Init();
    Target.resize(0);
    AttArr.resize(attentionWidthCount);
    for (TAttentionInfo &att : AttArr) {
        att.Init();
    }
    if (CrossAttentionId >= 0) {
        Cross.Init();
    }
}


void TBatchNodes::AddSample(const TVector<TLabelIndex> &labels, const TVector<TVector<TAttentionSpan>> &attSpansArr, int groupId)
{
    Labels.AddSample(labels);
    Y_VERIFY(YSize(AttArr) == YSize(attSpansArr));
    for (yint k = 0; k < YSize(AttArr); ++k) {
        AttArr[k].AddSpans(attSpansArr[k]);
        AttArr[k].AddSample();
    }
    if (CrossAttentionId >= 0) {
        Cross.AddSample(groupId);
    }
}


void TBatchNodes::Finish()
{
    if (CrossAttentionId >= 0) {
        BuildCrossShuffle(Cross, &CrossShuffle, &AttArr[CrossAttentionId]);
    }
}