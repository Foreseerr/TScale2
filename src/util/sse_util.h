#pragma once
#include <immintrin.h>
#include <xmmintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
inline void RepMovsb(ui8 *dst, const ui8 *src, size_t count)
{
#if defined(_MSC_VER)
    // MSVC provides a direct intrinsic for rep movsb
    __movsb(dst, src, count);
#elif defined(__GNUC__) || defined(__clang__)
    // GCC and Clang require inline assembly to guarantee this specific instruction
    // 'D' = edi/rdi, 'S' = esi/rsi, 'c' = ecx/rcx
    __asm__ __volatile__("rep movsb" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
#else
    for (size_t k = 0; k < count; ++k) {
        *dst++ = *src++;
    }
#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline yint Float2Int(float x)
{
    return _mm_cvtss_si32(_mm_set_ss(x));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
inline float HorizontalSum(__m256 x)
{
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
