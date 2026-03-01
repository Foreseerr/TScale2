#pragma once
#include "att.h"

typedef ui32 TLabelIndex;
const TLabelIndex LABEL_INVALID_INDEX = 0xffffffff;
const TLabelIndex LABEL_NEGATIVE = 0x80000000;
const TLabelIndex LABEL_MASK = 0x7fffffff;


struct TNodeTarget
{
    yint Node = 0;
    yint TargetId = 0;

    TNodeTarget() {}
    TNodeTarget(yint nodeId, yint targetId) : Node(nodeId), TargetId(targetId) {}
};

inline bool operator==(const TNodeTarget &a, const TNodeTarget &b)
{
    return a.Node == b.Node && a.TargetId == b.TargetId;
}


struct TBatchLabels
{
    TVector<TLabelIndex> LabelArr;
    TVector<ui32> LabelPtr;

    void Init();
    void AddSample(const TVector<TLabelIndex> &labels);
    void AddSample(TLabelIndex lbl);
};


struct TBatchNodes
{
    yint CrossAttentionId = -1;
    TBatchLabels Labels;
    TVector<TNodeTarget> Target;
    TVector<TAttentionInfo> AttArr;
    TAttentionCrossShuffle CrossShuffle;
    TAttentionCrossBatchInfo Cross;

    void Init(yint attentionWidthCount, yint crossAttentionId);
    void AddSample(const TVector<TLabelIndex> &labels, const TVector<TVector<TAttentionSpan>> &attSpansArr, int groupId);
    void Finish();
    yint GetNodeCount() const { return YSize(Labels.LabelPtr) - 1; }
};

