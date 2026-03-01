#pragma once
#include <gpt/model_params/sse_utils.h>
#include <lib/cuda/cuda_arrays.h>


constexpr yint MODEL_INT8_DELTA_TILE = 128;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct THostPackedDeltaPtr
{
    THost2DPtr<i8> Delta; // [y][x]
    THost2DPtr<float> TileScale; // [tile][y]

    THostPackedDeltaPtr(const THost2DPtr<i8> &delta, const THost2DPtr<float> &tileScale) : Delta(delta), TileScale(tileScale)
    {
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelMatrixHalfDelta
{
    struct TRow
    {
        float Scale;
        float Sum2; // sum2 of scaled values
    };
    yint SizeX = 0;
    yint SizeY = 0;
    TVector<fp16> Delta;
    TVector<TRow> Rows;

public:
    void Init(yint xSize, yint ySize)
    {
        SizeX = xSize;
        SizeY = ySize;
        ClearPodArray(&Delta, xSize * ySize);
        ClearPodArray(&Rows, ySize);
    }
    const fp16 *GetRow(yint y) const { return &Delta[y * SizeX]; }
    fp16 *GetRow(yint y) { return &Delta[y * SizeX]; }
    void GetAllData(TArray2D<float> *p) const;
};

void Copy(TModelMatrixHalfDelta *p, const THostPackedDeltaPtr &delta);
void Add(TModelMatrixHalfDelta *p, const THostPackedDeltaPtr &delta);
void Compress(THostPackedDeltaPtr *p, const TArray2D<float> &data);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelMatrixBitDelta
{
    TVector<float> DeltaRowDisp;
    TVector<ui64> BitDelta;
    SAVELOAD(DeltaRowDisp, BitDelta);

    bool IsEmpty() const { return BitDelta.empty(); }
    void Clear()
    {
        DeltaRowDisp.resize(0);
        BitDelta.resize(0);
    }
    void Swap(TModelMatrixBitDelta *p)
    {
        DeltaRowDisp.swap(p->DeltaRowDisp);
        BitDelta.swap(p->BitDelta);
    }
};


struct TModelMatrixBitDeltaTail
{
    TVector<ui64> BitDelta;

    void Init(yint xSize, yint ySize)
    {
        Y_VERIFY((xSize % 64) == 0);
        ClearPodArray(&BitDelta, ySize * xSize / 64);
    }
};


void SumBitDelta(const TModelMatrixBitDelta &a, const TModelMatrixBitDelta &b, TModelMatrixBitDeltaTail *pTail, TModelMatrixBitDelta *pRes);
