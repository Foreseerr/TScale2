#include "delta.h"
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr float HALF_SUM_SCALE = 1;

inline float HalfToFloat(fp16 x)
{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(x)));
}

inline fp16 FloatToHalf(float x)
{
    return _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(x), 0), 0);
}

inline float RoundFloatPow2(float x)
{
    int val = *(int *)&x;
    val &= 0xff800000;
    return *(float *)&val;
}


void TModelMatrixHalfDelta::GetAllData(TArray2D<float> *p) const
{
    yint xSize = SizeX;
    yint ySize = SizeY;
    p->SetSizes(xSize, ySize);
    p->FillZero();
    for (yint y = 0; y < ySize; ++y) {
        const TRow &row = Rows[y];
        if (row.Scale == 0) {
            continue;
        }
        const fp16 *deltaRowPtr = GetRow(y);
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = HalfToFloat(deltaRowPtr[x]) * row.Scale;
        }
    }
}


inline void CopyRow(TModelMatrixHalfDelta::TRow *pRow, yint xSize, fp16 *dstArg, const THostPackedDeltaPtr &srcData, yint y)
{
    float sum2 = 0;
    float maxScale = 0;
    for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
        maxScale = fmaxf(maxScale, srcData.TileScale[tile][y]);
    }
    float newRowScale = RoundFloatPow2(maxScale / HALF_SUM_SCALE);
    if (maxScale > 0) {
        const ui64 *src = (const ui64 *)srcData.Delta[y];
        __m128i *dst = (__m128i *)dstArg;
        __m256 rowSum2 = _mm256_setzero_ps();
        for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
            __m256 srcMult = _mm256_set1_ps(srcData.TileScale[tile][y] / newRowScale);
            for (yint x8 = 0; x8 < MODEL_INT8_DELTA_TILE / 8; ++x8) {
                // Load the 64-bit integer into a 128-bit register
                __m128i src8 = _mm_cvtsi64_si128(src[x8]);
                // Unpack the 8 int8 integers into 32-bit integers
                __m256i src32 = _mm256_cvtepi8_epi32(src8);
                // convert to float and scale
                __m256 newDstVal = _mm256_mul_ps(_mm256_cvtepi32_ps(src32), srcMult);
                // convert to fp16 and write
                dst[x8] = _mm256_cvtps_ph(newDstVal, 0);
                // collect rowsum
                rowSum2 = _mm256_add_ps(_mm256_mul_ps(newDstVal, newDstVal), rowSum2);
            }
            src += MODEL_INT8_DELTA_TILE / 8;
            dst += MODEL_INT8_DELTA_TILE / 8;
        }
        sum2 = HorizontalSum(rowSum2);
    }
    Y_ASSERT(!isnan(sum2) && isfinite(sum2));
    pRow->Scale = newRowScale;
    pRow->Sum2 = sum2 * Sqr(pRow->Scale);
}


inline void AddRow(TModelMatrixHalfDelta::TRow *pRow, yint xSize, fp16 *dstArg, const THostPackedDeltaPtr &srcData, yint y)
{
    float maxScale = 0;
    for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
        maxScale = fmaxf(maxScale, srcData.TileScale[tile][y]);
    }
    if (maxScale == 0) {
        return;
    }
    Y_ASSERT(pRow->Scale > 0 && maxScale > 0);
    float newRowScale = Max<float>(RoundFloatPow2(maxScale / HALF_SUM_SCALE), pRow->Scale);
    __m256 dstMult = _mm256_set1_ps(1 * pRow->Scale / newRowScale);

    const ui64 *src = (const ui64 *)srcData.Delta[y];
    __m128i *dst = (__m128i *)dstArg;
    __m256 rowSum2 = _mm256_setzero_ps();
    for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
        __m256 srcMult = _mm256_set1_ps(srcData.TileScale[tile][y] / newRowScale);
        for (yint x8 = 0; x8 < MODEL_INT8_DELTA_TILE / 8; ++x8) {
            __m256 dstVal = _mm256_cvtph_ps(dst[x8]);
            // Load the 64-bit integer into a 128-bit register
            __m128i src8 = _mm_cvtsi64_si128(src[x8]);
            // Unpack the 8 int8 integers into 32-bit integers
            __m256i src32 = _mm256_cvtepi8_epi32(src8);
            // convert to float and scale
            __m256 srcVal = _mm256_cvtepi32_ps(src32);
            // new val
            __m256 newDstVal = _mm256_add_ps(_mm256_mul_ps(srcVal, srcMult), _mm256_mul_ps(dstVal, dstMult));
            // convert to fp16 and write
            dst[x8] = _mm256_cvtps_ph(newDstVal, 0);
            // collect rowsum
            rowSum2 = _mm256_add_ps(_mm256_mul_ps(newDstVal, newDstVal), rowSum2);
        }
        src += MODEL_INT8_DELTA_TILE / 8;
        dst += MODEL_INT8_DELTA_TILE / 8;
    }
    float sum2 = HorizontalSum(rowSum2);
    Y_ASSERT(!isnan(sum2) && isfinite(sum2));
    pRow->Scale = newRowScale;
    pRow->Sum2 = sum2 * Sqr(pRow->Scale);
}


void Copy(TModelMatrixHalfDelta *p, const THostPackedDeltaPtr &delta)
{
    yint xSize = p->SizeX;
    yint ySize = p->SizeY;
    p->Delta.resize(xSize * ySize);
    p->Rows.resize(ySize);
    for (yint y = 0; y < ySize; ++y) {
        TModelMatrixHalfDelta::TRow &row = p->Rows[y];
        fp16 *dst = &p->Delta[y * xSize];
        CopyRow(&p->Rows[y], xSize, dst, delta, y);
    }
}


void Add(TModelMatrixHalfDelta *p, const THostPackedDeltaPtr &delta)
{
    yint xSize = p->SizeX;
    yint ySize = p->SizeY;
    for (yint y = 0; y < ySize; ++y) {
        TModelMatrixHalfDelta::TRow &row = p->Rows[y];
        fp16 *dst = &p->Delta[y * xSize];
        if (row.Sum2 == 0) {
            CopyRow(&row, xSize, dst, delta, y);
        } else {
            AddRow(&row, xSize, dst, delta, y);
        }
    }
}


void Compress(THostPackedDeltaPtr *p, const TArray2D<float> &data)
{
    Y_VERIFY((data.GetXSize() % MODEL_INT8_DELTA_TILE) == 0);
    yint xSize = data.GetXSize();
    yint ySize = data.GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        yint tileId = 0;
        for (yint xOffset = 0; xOffset < xSize; xOffset += MODEL_INT8_DELTA_TILE) {
            float maxVal = 0;
            for (yint x = 0; x < MODEL_INT8_DELTA_TILE; ++x) {
                maxVal = Max<float>(maxVal, fabs(data[y][xOffset + x]));
            }
            if (maxVal == 0) {
                p->TileScale[tileId][y] = 0;
            } else {
                float scale = maxVal / 127;
                float mult = 1 / scale;
                p->TileScale[tileId][y] = scale;
                i8 *dstPtr = p->Delta[y] + xOffset;
                ConvertArray(dstPtr, &data[y][xOffset], MODEL_INT8_DELTA_TILE, _mm256_set1_ps(mult));
            }
            ++tileId;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void Add1(yint sz, const ui64 *a, ui64 *tail)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        tail[k] = a1;
    }
}

static void Add2(yint sz, const ui64 *a, ui64 *tail, ui64 *res)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        ui64 a2 = tail[k];
        res[k] = a1 & a2;
        tail[k] = a1 ^ a2;
    }
}

static void Add3(yint sz, const ui64 *a, const ui64 *b, ui64 *tail, ui64 *res)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        ui64 a2 = b[k];
        ui64 a3 = tail[k];
        res[k] = (a1 & a2) | (a1 & a3) | (a2 & a3);
        tail[k] = a1 ^ a2 ^ a3;
    }
}


void SumBitDelta(const TModelMatrixBitDelta &a, const TModelMatrixBitDelta &b, TModelMatrixBitDeltaTail *pTail, TModelMatrixBitDelta *pRes)
{
    if (a.IsEmpty() && b.IsEmpty()) {
        // zero delta
        pRes->Clear();
        return;
    }
    // average row disp estimate
    Y_VERIFY(YSize(a.DeltaRowDisp) == YSize(b.DeltaRowDisp));
    yint rdCount = YSize(a.DeltaRowDisp);
    pRes->DeltaRowDisp.resize(rdCount);
    for (yint y = 0; y < rdCount; ++y) {
        pRes->DeltaRowDisp[y] = (a.DeltaRowDisp[y] + b.DeltaRowDisp[y]) * 0.5f;
    }
    // sum bit deltas
    yint sz = YSize(a.BitDelta);
    Y_VERIFY(YSize(b.BitDelta) == sz);
    Y_VERIFY(YSize(pTail->BitDelta) == sz);
    pRes->BitDelta.yresize(sz);
    Add3(sz, a.BitDelta.data(), b.BitDelta.data(), pTail->BitDelta.data(), pRes->BitDelta.data());
}
