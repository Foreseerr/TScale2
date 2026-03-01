#include "host_matrix.h"
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
static ui64 ByteMaskToInt[256];
static struct TInitByteMaskToInt
{
    TInitByteMaskToInt()
    {
        for (yint k = 0; k < 256; ++k) {
            ui64 res = 0;
            for (yint b = 0; b < 8; ++b) {
                if (k & (1ll << b)) {
                    res |= 0xffull << (b * 8);
                }
            }
            ByteMaskToInt[k] = res;
        }
    }
} initByteMaskToInt;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMatrixDeltaAddCtx
{
    THost2DPtr<float> Matr;
    THost2DPtr<float> AvrgDelta1;
    __m256 NewSum2;
    __m256 Beta1;
    __m256 BetaMult1;
    __m256 Weight0;
    __m256 Weight1;
    __m256 StepMult;
    __m256 ShrinkMult;
    __m256 AllSignBits;

    TMatrixDeltaAddCtx(THost2DPtr<float> matr, THost2DPtr<float> avrgDelta1, float stepArg, float sparsity, float l2reg, float beta1, float w0, float w1)
        : Matr(matr), AvrgDelta1(avrgDelta1)
    {
        float step = stepArg * sqrt(sparsity);
        NewSum2 = _mm256_setzero_ps();
        Beta1 = _mm256_set1_ps(beta1);
        BetaMult1 = _mm256_set1_ps(1 - beta1);
        Weight0 = _mm256_set1_ps(w0);
        Weight1 = _mm256_set1_ps(w1);
        StepMult = _mm256_set1_ps(step);
        ShrinkMult = _mm256_set1_ps(GetShrinkMult(step, l2reg));
        AllSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    }

    void AddRow(yint y, yint xSize)
    {
        __m256 *avrgDelta1Ptr = (__m256 *) & AvrgDelta1[y][0];
        __m256 *matrPtr = (__m256 *) & Matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            __m256 oldAvrgDelta1 = avrgDelta1Ptr[x8];
            __m256 avrgDelta1 = _mm256_mul_ps(oldAvrgDelta1, Beta1);
            __m256 totalDelta = _mm256_mul_ps(avrgDelta1, Weight1);
            __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], ShrinkMult), _mm256_mul_ps(totalDelta, StepMult));
            avrgDelta1Ptr[x8] = avrgDelta1;
            matrPtr[x8] = val;
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        NewSum2 = _mm256_add_ps(NewSum2, rowSum2);
    }

    void AddRow(yint y, yint xSize, const __m128i *deltaPtr, __m256 deltaScale)
    {
        __m256 *avrgDelta1Ptr = (__m256 *) & AvrgDelta1[y][0];
        __m256 *matrPtr = (__m256 *) & Matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            __m256 deltaVal = _mm256_mul_ps(_mm256_cvtph_ps(deltaPtr[x8]), deltaScale);
            __m256 oldAvrgDelta1 = avrgDelta1Ptr[x8];
            __m256 avrgDelta1 = _mm256_add_ps(_mm256_mul_ps(oldAvrgDelta1, Beta1), _mm256_mul_ps(deltaVal, BetaMult1));
            __m256 totalDelta12 = _mm256_mul_ps(avrgDelta1, Weight1);
            __m256 totalDelta = _mm256_add_ps(totalDelta12, _mm256_mul_ps(deltaVal, Weight0));
            __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], ShrinkMult), _mm256_mul_ps(totalDelta, StepMult));
            avrgDelta1Ptr[x8] = avrgDelta1;
            matrPtr[x8] = val;
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        NewSum2 = _mm256_add_ps(NewSum2, rowSum2);
    }

    void AddBitRow(yint y, yint xSize, ui8 *bitDeltaPtr)
    {
        __m256 *avrgDelta1Ptr = (__m256 *) & AvrgDelta1[y][0];
        __m256 *matrPtr = (__m256 *) & Matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (int x8 = 0; x8 < xSize / 8; ++x8) {
            ui64 byteMask = ByteMaskToInt[bitDeltaPtr[x8]];
            __m256i mask = _mm256_cvtepi8_epi32(_mm_set_epi64x(0, byteMask));
            __m256 deltaSignBits = _mm256_and_ps(AllSignBits, _mm256_castsi256_ps(mask));
            __m256 oldAvrgDelta1 = avrgDelta1Ptr[x8];
            __m256 avrgDelta1 = _mm256_add_ps(_mm256_mul_ps(oldAvrgDelta1, Beta1), _mm256_xor_ps(deltaSignBits, BetaMult1));
            __m256 totalDelta12 = _mm256_mul_ps(avrgDelta1, Weight1);
            __m256 totalDelta = _mm256_add_ps(totalDelta12, _mm256_xor_ps(deltaSignBits, Weight0));
            __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], ShrinkMult), _mm256_mul_ps(totalDelta, StepMult));
            avrgDelta1Ptr[x8] = avrgDelta1;
            matrPtr[x8] = val;
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        NewSum2 = _mm256_add_ps(NewSum2, rowSum2);
    }
};


void THostModelMatrix::AddDelta(const TModelMatrixHalfDelta &delta, const TTrainingStep &step)
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    yint rowDispSize = YSize(RowDisp);
    yint rowDispStep = ySize / rowDispSize;
    float dispDecay = step.DispDecay;

    TMatrixDeltaAddCtx ctx(Matr.GetHostPtr(), AvrgDelta1.GetHostPtr(), step.Rate, Sparsity, step.L2Reg, step.Beta1, step.Weight0, step.Weight1);
    SumWeight = SumWeight * dispDecay + 1;
    float rowDispNorm = 1 / SumWeight;
    for (yint blockId = 0; blockId < rowDispSize; ++blockId) {
        yint blkStart = blockId * rowDispStep;

        float sum2 = 0;
        for (yint y = blkStart; y < blkStart + rowDispStep; ++y) {
            sum2 += delta.Rows[y].Sum2;
        }
        float disp = sum2 / (rowDispStep * xSize);
        
        RowDisp[blockId] = (RowDisp[blockId] * dispDecay) + disp;
        float normScale = 1 / sqrt(RowDisp[blockId] * rowDispNorm);
        for (yint y = blkStart; y < blkStart + rowDispStep; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            if (row.Sum2 > 0) {
                const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
                __m256 deltaScale = _mm256_set1_ps(row.Scale * normScale);
                ctx.AddRow(y, xSize, deltaPtr, deltaScale);
            } else {
                ctx.AddRow(y, xSize);
            }
        }
    }
    Sum2 = HorizontalSum(ctx.NewSum2);
}


bool THostModelMatrix::AddBitDelta(const TModelMatrixBitDelta &bitDelta, const TTrainingStep &step)
{
    if (bitDelta.IsEmpty()) {
        return false;
    }
    yint xSize = GetXSize();
    yint ySize = GetYSize();
    float dispDecay = step.DispDecay;

    yint rdCount = YSize(RowDisp);
    Y_VERIFY(rdCount == YSize(bitDelta.DeltaRowDisp));
    SumWeight = SumWeight * dispDecay + 1;
    for (yint y = 0; y < rdCount; ++y) {
        RowDisp[y] = (RowDisp[y] * dispDecay) + bitDelta.DeltaRowDisp[y];
    }

    TMatrixDeltaAddCtx ctx(
        Matr.GetHostPtr(), AvrgDelta1.GetHostPtr(), step.Rate, Sparsity, step.L2Reg, step.Beta1, step.Weight0, step.Weight1);
    for (yint y = 0; y < ySize; ++y) {
        ui8 *bitDeltaPtr = (ui8 *)&bitDelta.BitDelta[y * xSize / 64];
        ctx.AddBitRow(y, xSize, bitDeltaPtr);
    }
    Sum2 = HorizontalSum(ctx.NewSum2);
    return true;
}


inline void CompressLine(ui8 *resPtr, const __m128i *deltaPtr, __m256 deltaScale, __m256 *deltaTailPtr, yint xSize, __m256 allSignBits, __m256 basicStep)
{
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        __m256 deltaVal = _mm256_mul_ps(_mm256_cvtph_ps(deltaPtr[x8]), deltaScale);
        // val = tail + delta
        __m256 val = _mm256_add_ps(deltaTailPtr[x8], deltaVal);
        // signBit = val > 0
        __m256 signBit = _mm256_and_ps(allSignBits, val);
        // add = (val > 0) ? basicStep : -basicStep
        __m256 add = _mm256_or_ps(signBit, basicStep);
        // tail = val - add
        deltaTailPtr[x8] = _mm256_sub_ps(val, add);
        resPtr[x8] = _mm256_movemask_ps(signBit);
    }
}

void THostModelMatrix::CompressDelta(const TModelMatrixHalfDelta &delta, TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail)
{
    TArray2D<float> &deltaTail = *pDeltaTail;
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    yint rowDispSize = YSize(RowDisp);
    yint rowDispStep = ySize / rowDispSize;
    Y_ASSERT((xSize % 64) == 0);
    __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    pBitDelta->DeltaRowDisp.yresize(rowDispSize);
    pBitDelta->BitDelta.yresize(ySize * xSize / 64);
    for (yint blockId = 0; blockId < rowDispSize; ++blockId) {
        yint blkStart = blockId * rowDispStep;

        float sum2 = 0;
        for (yint y = blkStart; y < blkStart + rowDispStep; ++y) {
            sum2 += delta.Rows[y].Sum2;
        }
        float disp = sum2 / (rowDispStep * xSize);
        float dispEstimate = (RowDisp[blockId] + disp) / (SumWeight + 1);
        pBitDelta->DeltaRowDisp[blockId] = disp;

        __m256 basicStep = _mm256_set1_ps(sqrt(dispEstimate));
        for (yint y = blkStart; y < blkStart + rowDispStep; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
            __m256 deltaScale = _mm256_set1_ps(row.Scale);
            __m256 *deltaTailPtr = (__m256 *)deltaTail.GetRow(y);
            ui8 *resPtr = (ui8 *)&pBitDelta->BitDelta[y * xSize / 64];
            CompressLine(resPtr, deltaPtr, deltaScale, deltaTailPtr, xSize, allSignBits, basicStep);
        }
    }
}
