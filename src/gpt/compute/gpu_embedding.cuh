#pragma once
#include <util/radix_sort.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// embedding

struct TLabelInverseIndex
{
    struct TLabelPos
    {
        union {
            struct {
                int Pos;
                TLabelIndex Label;
            };
            ui64 SortValue;
        };

        TLabelPos() {}
        TLabelPos(TLabelIndex label, int pos) : Label(label), Pos(pos) {}
    };

    static const ui64 *GetSortValue(const TLabelPos &x)
    {
        return &x.SortValue;
    }

    TVector<TLabelPos> LabelPosArr;
    TVector<TLabelPos> LabelPosArrTempBuf;
    TVector<TLabelIndex> InvLabelArr;
    TVector<ui32> InvLabelPos;
    TVector<ui32> InvLabelPosPtr;

    // make list of references for each label
    // for labels with positions in previous iteration and no positions in current iteration create empty list
    void BuildInverseIndex(const TVector<TLabelIndex> &labelArr, const TVector<ui32> &labelPtr)
    {
        Y_ASSERT(sizeof(LABEL_NEGATIVE) == sizeof(InvLabelPos[0]));
        yint labelCount = YSize(labelArr);
        Y_ASSERT(labelCount == labelPtr.back());

        // add current iteration positions
        LabelPosArr.resize(labelCount);
        for (yint pos = 0; pos < YSize(labelPtr) - 1; ++pos) {
            for (yint k = labelPtr[pos]; k < labelPtr[pos + 1]; ++k) {
                TLabelIndex label = labelArr[k];
                TLabelIndex negMask = label & LABEL_NEGATIVE;
                LabelPosArr[k] = TLabelPos(label & LABEL_MASK, pos | negMask);
            }
        }

        // invert index
        LabelPosArrTempBuf.resize(labelCount);
        RadixUI64SortAscending(&LabelPosArr, &LabelPosArrTempBuf, GetSortValue);

        // make lists
        InvLabelArr.resize(0);
        InvLabelPos.resize(labelCount);
        InvLabelPosPtr.resize(0);
        TLabelIndex prevLabel = LABEL_INVALID_INDEX;
        for (yint k = 0; k < labelCount; ++k) {
            const TLabelPos &labelPos = LabelPosArr[k];
            TLabelIndex label = labelPos.Label;
            if (label != prevLabel) {
                InvLabelPosPtr.push_back(k);
                InvLabelArr.push_back(label);
                prevLabel = label;
            }
            InvLabelPos[k] = labelPos.Pos;
        }
        InvLabelPosPtr.push_back(labelCount);
    }
};


struct TLabelForwardIndex
{
    struct TLabelInfo
    {
        int Index = 0;
        int Timestamp = 0;
    };
    TVector<TLabelInfo> LabelIndex;
    TVector<TLabelIndex> RecodedLabelArr;
    TVector<TLabelIndex> UsedLabels;
    int Timestamp = 0;

public:
    void Init(int labelCount)
    {
        ClearPodArray(&LabelIndex, labelCount);
        Timestamp = 0;
    }
    void BuildUsedIndex(const TVector<TLabelIndex> &labelArr)
    {
        int curTimestamp = ++Timestamp;
        int sz = YSize(labelArr);
        RecodedLabelArr.yresize(sz);
        UsedLabels.yresize(sz);
        int dst = 0;
        for (int i = 0; i < sz; ++i) {
            TLabelIndex srcLabel = labelArr[i];
            TLabelIndex label = srcLabel & LABEL_MASK;
            TLabelInfo &info = LabelIndex[label];
            if (info.Timestamp != curTimestamp) {
                UsedLabels[dst] = label;
                info.Index = dst++;
                info.Timestamp = curTimestamp;
            }
            RecodedLabelArr[i] = info.Index | (srcLabel & LABEL_NEGATIVE);
        }
        UsedLabels.resize(dst);
        if (UsedLabels.empty()) {
            // avoid zero UsedLabelCount
            UsedLabels.push_back(LABEL_INVALID_INDEX);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// embed kernels
static __global__ void CopyUsedEmbeddings(int labelCount, TCuda1DPtr<TLabelIndex> labelArr, TCuda2DPtr<TEmbedFloat> labelEmbedding, TCuda2DPtr<TEmbedFloat> dst)
{
    int tile = blockIdx.x;
    int k = blockIdx.y;
    int label = labelArr[k];
    int offset = tile * MM_TILE;

    float4 vec;
    if (label >= 0 && label < labelCount) {
        vec = LoadWarpVec(labelEmbedding[label] + offset);
    } else {
        vec = ZeroWarpVec();
    }
    StoreWarpVec(dst[k] + offset, vec);
}


template <class T>
__device__ void AddEmbedVec(float4 *dst, TCuda2DPtr<T> labelEmbedding, int offset, TLabelIndex idx)
{
    float4 vec = LoadWarpVec(labelEmbedding[idx & LABEL_MASK] + offset);
    if (idx & LABEL_NEGATIVE) {
        vec = Scale(vec, -1);
    }
    *dst = *dst + vec;
}

static __global__ void AddEmbeddings(
    int len,
    TCuda1DPtr<TLabelIndex> labelArr, TCuda1DPtr<ui32> labelPtr, TCuda2DPtr<TEmbedFloat> labelEmbedding, float *labelEmbeddingScale, float labelEmbeddingStaticScale,
    int isUsingSampleEmbedVectors, TCuda2DPtr<TEmbedFloat> sampleEmbedVectors,
    TCuda2DPtr<TStateFloat> state
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;

    float4 vec = ZeroWarpVec();
    if (t < len) {
        int start = labelPtr[t];
        int finish = labelPtr[t + 1];
        if (finish > start) {
            for (int z = start; z < finish; ++z) {
                AddEmbedVec(&vec, labelEmbedding, offset, labelArr[z]);
            }
            vec = Scale(vec, *labelEmbeddingScale * labelEmbeddingStaticScale);
        }
        if (isUsingSampleEmbedVectors) {
            vec = vec + LoadWarpVec(sampleEmbedVectors[t] + offset);
        }
    }
    StoreWarpVec(state[t] + offset, vec);
}


template <class TGradFloat>
__global__ void BackpropEmbeddings(
    TCuda1DPtr<TLabelIndex> invLabelArr, TCuda1DPtr<ui32> invLabelPos, TCuda1DPtr<ui32> invLabelPosPtr,
    TCuda2DPtr<TGradFloat> stateGrad,
    TCuda2DPtr<i8> deltaLabelEmbedding, TCuda2DPtr<float> deltaLabelTileScale
)
{
    CUDA_STATIC_ASSERT(MM_TILE == MODEL_INT8_DELTA_TILE);
    int tile = blockIdx.x;
    int labelId = blockIdx.y;
    int offset = tile * MM_TILE;

    int start = invLabelPosPtr[labelId];
    int finish = invLabelPosPtr[labelId + 1];
    if (start == finish) {
        return;
    }

    // compute gradient
    float4 delta = ZeroWarpVec();
    for (int z = start; z < finish; ++z) {
        AddEmbedVec(&delta, stateGrad, offset, invLabelPos[z]);
    }

    // compute scale
    float maxVal = CalcWarpVecMaxAbsValue(delta);
    float scale = (maxVal > 0) ? maxVal / 127 : 0;
    float mult = (maxVal > 0) ? 1 / scale : 0;

    // write
    int label = invLabelArr[labelId];
    StoreWarpVec(deltaLabelEmbedding[label] + offset, Scale(delta, mult));
    if (threadIdx.x == 0) {
        deltaLabelTileScale[tile][label] = scale;
    }
    __threadfence_system(); // neccessary since we write to host memory
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TEmbeddingLayer
{
    struct TMicroBatch : public TThrRefBase
    {
        TCudaVector<TLabelIndex> LabelArr;
        TCudaVector<TLabelIndex> UsedLabelArr;
        TCudaVector<ui32> LabelPtr;
        TCuda2DArray<TEmbedFloat> SampleEmbedVectors;
        TKernelParameter<int> UsedLabelCount;
        TKernelParameter<int> InvLabelCount;
        TVector<ui32> HoldLabelPtr;
        // indexing
        TLabelForwardIndex LabelForwardIndex;
        TLabelInverseIndex LabelInverseIndex;
        // backprop
        TCudaVector<TLabelIndex> InvLabelArr;
        TCudaVector<ui32> InvLabelPos;
        TCudaVector<ui32> InvLabelPosPtr;
    };

    bool IsInitialized = false;
    TModelDescr ModelDescr;
    TCuda2DArray<TEmbedFloat> UsedEmbedBuffer; // temporary use, can reuse memory from other buffers (like gradient) for this
    TVector<TIntrusivePtr<TMicroBatch>> MBatchArr;

public:
    TEmbeddingLayer(const TModelDescr &modelDescr) : ModelDescr(modelDescr)
    {
    }

    template <class TEmbeddingMatrix>
    void AddForwardGraph(
        TPtrArg<TGraph> c, yint microBatchId, TComputeParams *pParams, TPtrArg<TEmbeddingMatrix> pEmbedding, TCuda2DArray<TStateFloat> *pState)
    {
        Y_ASSERT(IsInitialized);
        int stateDim = pState->GetXSize();
        int stateTiles = stateDim / MM_TILE;
        int labelCount = ModelDescr.LabelCount;
        TMicroBatch &mb = *MBatchArr[microBatchId];
        Y_ASSERT(stateDim == pEmbedding->GetLocalXSize());

        // copy used embeddings
        CudaCall(c, CopyUsedEmbeddings)
            .Grid(stateTiles, mb.UsedLabelCount)
            .Read(labelCount, mb.UsedLabelArr, pEmbedding->GetFast())
            .Write(&UsedEmbedBuffer);

        // compute token vectors
        TCudaPOD<float> scaleEmbed = pEmbedding->GetScale();
        int isUsingSampleEmbedVectors = ModelDescr.HasFlag(MPF_SAMPLE_EMBED_VECTORS);
        CudaCall(c, AddEmbeddings)
            .Grid(MMTiles(stateDim), pParams->LenRound)
            .Read(pParams->Len)
            .Read(mb.LabelArr, mb.LabelPtr, UsedEmbedBuffer, scaleEmbed, MODEL_DISCR_SCALE)
            .Read(isUsingSampleEmbedVectors, mb.SampleEmbedVectors)
            .Write(pState);
    }


    template <class TEmbeddingMatrix, class TSomeFloat>
    void AddBackprop(
        TPtrArg<TGraph> c, yint microBatchId, TPtrArg<TEmbeddingMatrix> pEmbedding, TCuda2DArray<TSomeFloat> &stateGrad)
    {
        Y_ASSERT(IsInitialized);
        int stateDim = stateGrad.GetXSize();
        int stateTiles = stateDim / MM_TILE;
        TMicroBatch &mb = *MBatchArr[microBatchId];
        Y_ASSERT(stateDim == pEmbedding->GetLocalXSize());

        // tune embed
        TCudaPackedDeltaMatrix &hostDelta = pEmbedding->GetHostDelta();
        c->ClearMem(&hostDelta.TileScale);
        CudaCall(c, BackpropEmbeddings<TSomeFloat>)
            .Grid(stateTiles, mb.InvLabelCount)
            .Read(mb.InvLabelArr, mb.InvLabelPos, mb.InvLabelPosPtr)
            .Read(stateGrad)
            .Write(&hostDelta.Delta, &hostDelta.TileScale);
    }

    template <class TEmbeddingMatrix>
    void AddDelta(TPtrArg<TGraph> c, TPtrArg<TEmbeddingMatrix> pEmbedding, EBackpropMode bm) { pEmbedding->AddHostDelta(c, bm); }


    void AllocateCuda(TPtrArg<TCudaMemoryPool> poolEmbed, yint microBatchCount, yint dim, yint maxLen, bool needBackprop)
    {
        yint maxLabels = maxLen * 8; // upper cap
        yint maxUsedLabels = maxLen * 2; // upper cap
        yint labelCount = ModelDescr.LabelCount;

        UsedEmbedBuffer.AllocateCuda(dim, maxUsedLabels, poolEmbed);

        // micro batches
        MBatchArr.resize(microBatchCount);
        for (yint mbId = 0; mbId < microBatchCount; ++mbId) {
            MBatchArr[mbId] = new TMicroBatch;
            TMicroBatch &mb = *MBatchArr[mbId];
            mb.LabelArr.Allocate(maxLabels); // upper cap
            mb.UsedLabelArr.Allocate(maxUsedLabels); // upper cap
            mb.LabelPtr.Allocate(maxLen + 1);

            if (ModelDescr.HasFlag(MPF_SAMPLE_EMBED_VECTORS)) {
                mb.SampleEmbedVectors.Allocate(dim, maxLen);
            }

            mb.LabelForwardIndex.Init(labelCount);

            if (needBackprop) {
                mb.InvLabelArr.Allocate(labelCount); // upper cap
                mb.InvLabelPos.Allocate(maxLabels); // upper cap
                mb.InvLabelPosPtr.Allocate(labelCount + 1);
            }
        }
        IsInitialized = true;
    }


    void BuildIndex(yint microBatchId, const TBatchLabels &labels, bool isBackprop)
    {
        if (IsInitialized) {
            TMicroBatch &mb = *MBatchArr[microBatchId];
            Y_VERIFY(YSize(labels.LabelPtr) <= mb.LabelPtr.GetSize());
            Y_VERIFY(labels.LabelPtr.back() <= mb.LabelArr.GetSize());
            for (yint pos : labels.LabelArr) {
                Y_ASSERT((pos & LABEL_MASK) < ModelDescr.LabelCount);
            }
            if (isBackprop) {
                // inverse index is not needed for forward computations (inference)
                mb.LabelInverseIndex.BuildInverseIndex(labels.LabelArr, labels.LabelPtr);
            }
            mb.LabelForwardIndex.BuildUsedIndex(labels.LabelArr);
            mb.HoldLabelPtr = labels.LabelPtr;
        }
    }


    void Init(TStream &stream, yint microBatchId, const TArray2D<float> &sampleEmbedVectors, bool isBackprop)
    {
        if (IsInitialized) {
            TMicroBatch &mb = *MBatchArr[microBatchId];
            Put(stream, &mb.LabelArr, mb.LabelForwardIndex.RecodedLabelArr);
            Put(stream, &mb.UsedLabelArr, mb.LabelForwardIndex.UsedLabels);
            Put(stream, &mb.LabelPtr, mb.HoldLabelPtr);
            mb.UsedLabelCount.Set(YSize(mb.LabelForwardIndex.UsedLabels));

            if (ModelDescr.HasFlag(MPF_SAMPLE_EMBED_VECTORS)) {
                Put(stream, &mb.SampleEmbedVectors, sampleEmbedVectors);
            }

            if (isBackprop) {
                yint invLabelCount = YSize(mb.LabelInverseIndex.InvLabelArr);
                mb.InvLabelCount.Set(invLabelCount);

                Put(stream, &mb.InvLabelArr, mb.LabelInverseIndex.InvLabelArr);
                Put(stream, &mb.InvLabelPos, mb.LabelInverseIndex.InvLabelPos);
                Put(stream, &mb.InvLabelPosPtr, mb.LabelInverseIndex.InvLabelPosPtr);
            }
        }
    }
};


}
