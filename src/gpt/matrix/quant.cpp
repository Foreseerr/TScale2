#include "quant.h"
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
static i8 QBitRecode2bit[256];
static i8 QBitRecode4bit[256];
static struct TInitBitTable
{
    TInitBitTable()
    {
        for (yint a = 0; a < 256; ++a) {
            i8 x = a;
            if (x < -24) {
                QBitRecode2bit[a] = -36;
            } else if (x < 0) {
                QBitRecode2bit[a] = -12;
            } else if (x <= 24) {
                QBitRecode2bit[a] = 12;
            } else {
                QBitRecode2bit[a] = 36;
            }
        }
        for (yint a = 0; a < 256; ++a) {
            i8 x = a;
            yint xx = x;
            yint bitVal = ClampVal<yint>((xx + 4 + 9 * 8) / 9, 0, 15); // 4.88512
            QBitRecode4bit[a] = bitVal * 9 + 4 - 9 * 8;
        }
    }
} initBitTable;

void ConvertToFastMatrixFloat(i8 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    ConvertArray(dst, src, xSize, mult);

    // simulate quantization
    if (quant == MM_QUANT_158BIT) {
        // 1.58 bit
        for (yint x = 0; x < xSize; ++x) {
            if (dst[x] < -15) {
                dst[x] = -32;
            } else if (dst[x] > 15) {
                dst[x] = 32;
            } else {
                dst[x] = 0;
            }
        }
    } else if (quant == MM_QUANT_1BIT) {
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = (src[x] > 0) ? 32 : -32;
        }
    } else if (quant == MM_QUANT_2BIT) {
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = QBitRecode2bit[(ui8)dst[x]]; // can be speed up with SSE
        }
    } else if (quant == MM_QUANT_4BIT) {
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = QBitRecode4bit[(ui8)dst[x]]; // can be speed up with SSE
        }
    }
}


void ConvertToFastMatrixFloat(half *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    for (yint x = 0; x < xSize; x += 8) {
        // Load 8 floats from the input vector into a 256-bit register
        __m256 val = _mm256_mul_ps(_mm256_load_ps(src + x), mult);
        // Convert the 8 floats to 8 fp16 values and store them in a 128-bit register
        __m128i res = _mm256_cvtps_ph(val, 0);
        *(__m128i *)(dst + x) = res;
    }
    Y_VERIFY(quant == MM_QUANT_NONE);
}


void ConvertToFastMatrixFloat(e4m3 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    ConvertToFp8e4m3(dst, src, xSize, mult);
    Y_VERIFY(quant == MM_QUANT_NONE);
}
