#pragma once


inline float LogSech(float x)
{
    float ax = fabs(x);
    // Identity: log(sech(x)) = -|x| - log(1 + exp(-2|x|)) + log(2)
    // For large |x|, exp(-2|x|) approaches 0, and the result simplifies to -|x| + log(2).
    return -ax - logf(1 + expf(-2 * ax)) + logf(2.0);
}

// logarithm of hyperbolic secant distribution PDF
inline float LogHyperbolicSecant(float x, float lambda)
{
    return -logf(2 * lambda) + LogSech(PI / 2 * x / lambda);
}
