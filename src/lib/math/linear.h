#pragma once
#include <lib/math/eigen.h>
#include <util/sse_util.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TR2Loss
{
    float Sum2 = 0;
    float Rem2 = 0;

    void Init()
    {
        Sum2 = 0;
        Rem2 = 0;
    }
    void Add(float target, float approx, float w)
    {
        Sum2 += Sqr(target) * w;
        Rem2 += Sqr(target - approx) * w;
    }
    float GetR2() const
    {
        if (Sum2 > 0) {
            return (Sum2 - Rem2) / Sum2;
        } else {
            return 0;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2>
void Assign(TVector<T1> *p, const TVector<T2> &vec)
{
    yint sz = YSize(vec);
    p->resize(sz);
    for (yint k = 0; k < sz; ++k) {
        (*p)[k] = vec[k];
    }
}

template <class T1, class T2>
void Assign(TArray2D<T1> *p, const TArray2D<T2> &arr)
{
    yint xSize = arr.GetXSize();
    yint ySize = arr.GetYSize();
    p->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = arr[y][x];
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class T2>
void Scale(TVector<T> *p, T2 scale)
{
    yint sz = YSize(*p);
    for (yint k = 0; k < sz; ++k) {
        (*p)[k] *= scale;
    }
}

template <class T, class T2>
void Scale(TArray2D<T> *p, T2 scale)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] *= scale;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class T2, class T3>
static void AddScaled(TVector<T> *dst, const TVector<T2> &vec, T3 w)
{
    yint sz = YSize(*dst);
    Y_ASSERT(sz == YSize(vec));
    for (yint k = 0; k < sz; ++k) {
        (*dst)[k] += vec[k] * w;
    }
}

template <class T, class T2, class T3>
static void AddScaledProduct(TVector<T> *dst, const TVector<T2> &vec1, const TVector<T2> &vec2, T3 w)
{
    yint sz = YSize(*dst);
    Y_ASSERT(sz == YSize(vec1));
    Y_ASSERT(sz == YSize(vec2));
    for (yint k = 0; k < sz; ++k) {
        (*dst)[k] += vec1[k] * vec2[k] * w;
    }
}

template <class T, class T2, class T3>
void AddScaled(TArray2D<T> *dst, const TArray2D<T2> &a, T3 w)
{
    yint xSize = dst->GetXSize();
    yint ySize = dst->GetYSize();
    Y_ASSERT(xSize == a.GetXSize());
    Y_ASSERT(ySize == a.GetYSize());
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*dst)[y][x] += a[y][x] * w;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class T2, class T3>
static void ComputeEMA(TVector<T> *dst, const TVector<T2> &vec, T3 alpha)
{
    yint sz = YSize(*dst);
    Y_ASSERT(sz == YSize(vec));
    for (yint k = 0; k < sz; ++k) {
        (*dst)[k] = (*dst)[k] * alpha + vec[k] * (1 - alpha);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2, class T3>
void AddRankOne(TArray2D<T1> *dst, const TVector<T2> &vec, T3 w)
{
    yint sz = YSize(vec);
    Y_ASSERT(dst->GetXSize() == sz);
    Y_ASSERT(dst->GetYSize() == sz);
    const T2 *vecPtr = vec.data();
    for (yint y = 0; y < sz; ++y) {
        T1 *dstPtr = dst->GetRow(y);
        for (yint x = 0; x < sz; ++x) {
            dstPtr[x] += (vecPtr[y] * vecPtr[x]) * w;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
T CalcL1(const TVector <T> &a)
{
    T res = 0;
    for (T x : a) {
        res += fabs(x);
    }
    return res;
}

template <class T>
T CalcL2(const TVector <T> &a)
{
    T res = 0;
    for (T x : a) {
        res += x * x;
    }
    return sqrt(res);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
T CalcL2(const TArray2D<T> &a)
{
    double sum2 = 0;
    yint ySize = a.GetYSize();
    yint xSize = a.GetXSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            sum2 += Sqr(a[y][x]);
        }
    }
    return sum2;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2>
double Dot(const TVector<T1> &a, const TVector<T2> &b)
{
    yint sz = YSize(a);
    Y_ASSERT(YSize(b) == sz);
    double res = 0;
    for (yint k = 0; k < sz; ++k) {
        res += a[k] * b[k];
    }
    return res;
}

inline float DotFast(const TVector<float> &a, const TVector<float> &b)
{
    yint sz = YSize(a);
    Y_ASSERT(YSize(b) == sz);
    yint sz8 = sz / 8;
    const __m256 *aPtr8 = (const __m256 *)a.data();
    const __m256 *bPtr8 = (const __m256 *)b.data();
    const __m256 *aFin8 = aPtr8 + sz8;
    __m256 sum = _mm256_setzero_ps();
    while (aPtr8 < aFin8) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(*aPtr8++, *bPtr8++));
    }
    float res = HorizontalSum(sum);
    const float *aPtr = (const float *)aPtr8;
    const float *bPtr = (const float *)bPtr8;
    for (yint k = sz8 * 8; k < sz; ++k) {
        res += (*aPtr++) * (*bPtr++);
    }
    return res;
}

template <class T1, class T2, class T3>
static double Dot3(const TVector<T1> &a, const TVector<T2> &b, const TVector<T3> &c)
{
    yint sz = YSize(a);
    Y_ASSERT(YSize(b) == sz);
    Y_ASSERT(YSize(c) == sz);
    double res = 0;
    for (yint k = 0; k < sz; ++k) {
        res += a[k] * b[k] * c[k];
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2, class T3>
double BilinearProduct(const TVector<T1> &a, const TArray2D<T2> &matr, const TVector<T3> &b)
{
    yint xSize = YSize(b);
    yint ySize = YSize(a);
    Y_VERIFY(matr.GetXSize() == xSize);
    Y_VERIFY(matr.GetYSize() == ySize);
    double res = 0;
    for (yint y = 0; y < ySize; ++y) {
        double sum = 0;
        const T2 *matrRow = matr.GetRow(y);
        for (yint x = 0; x < xSize; ++x) {
            sum += matrRow[x] * b[x];
        }
        res += a[y] * sum;
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void Normalize(TVector<T> *p)
{
    T sum2 = Dot(*p, *p);
    if (sum2 > 0) {
        T scale = sqrt(1 / sum2);
        Scale(p, scale);
    }
}

template <class T>
void NormalizeL1(TVector<T> *p)
{
    T sum = 0;
    for (T x : *p) {
        sum += fabs(x);
    }
    if (sum > 0) {
        T scale = 1 / sum;
        Scale(p, scale);
    }
}

template <class T>
void Normalize(TArray2D<T> *p)
{
    T sum2 = CalcL2(*p);
    if (sum2 > 0) {
        T scale = sqrt(1 / sum2);
        Scale(p, scale);
    }
}

void BackpropNormalize(const TVector<float> &srcVec, const TVector<float> &gradVec, TVector<float> *pRes);


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2>
void MakeOrthogonal(TVector<T1> *p, const TVector<T2> &vec)
{
    T1 dp = Dot(*p, vec);
    T1 sum2 = Dot(vec, vec);
    if (sum2 > 0) {
        AddScaled(p, vec, -dp / sum2);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void Transpose(TArray2D<T> *p)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    Y_VERIFY(xSize == ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = y + 1; x < xSize; ++x) {
            DoSwap((*p)[y][x], (*p)[x][y]);
        }
    }
}

template <class T>
TArray2D<T> Transpose(const TArray2D<T> &matr)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    TArray2D<T> res;
    res.SetSizes(ySize, xSize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            res[x][y] = matr[y][x];
        }
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
TArray2D<T> MakeE(yint sz)
{
    TArray2D<T> rv;
    rv.SetSizes(sz, sz);
    rv.FillZero();
    for (yint f = 0; f < sz; ++f) {
        rv[f][f] = 1;
    }
    return rv;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TEigenDecomp
{
    TVector<double> EigenArr;
    TVector<TVector<double>> EigenVecs;
    SAVELOAD(EigenArr, EigenVecs);

    bool IsEmpty() const
    {
        return EigenArr.empty();
    }

    yint GetCount() const
    {
        return YSize(EigenArr);
    }

    TVector<float> GetVec(yint k) const
    {
        TVector<float> main;
        for (float x : EigenVecs[k]) {
            main.push_back(x);
        }
        return main;
    }

    void Compute(const TArray2D<double> &corr)
    {
        //DebugPrintf("calc eigen vecs\n");
        NEigen::CalcEigenVectors(&EigenArr, &EigenVecs, corr);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, class TDst>
inline void LinSystemGradientSolve(const TEigenDecomp &eigen, const TVector<T> &proj, double t, TVector<TDst> *p)
{
    yint sz = YSize(proj);
    ClearPodArray(p, sz);
    for (yint k = 0; k < sz; ++k) {
        const TVector<double> &vec = eigen.EigenVecs[k];
        double lambda = Max<double>(0, eigen.EigenArr[k]);
        double xInv = 0;
        if (lambda * t < 1e-5) {
            xInv = t;
        } else {
            xInv = (1 - exp(-lambda * t)) / lambda;
        }
        double dp = Dot(proj, vec);
        AddScaled(p, vec, dp * xInv);
    }
}


template <class T, class TDst>
inline void LinSystemRidgeSolve(const TEigenDecomp &eigen, const TVector<T> &proj, double reg, TVector<TDst> *p)
{
    yint sz = YSize(proj);
    ClearPodArray(p, sz);
    for (yint k = 0; k < sz; ++k) {
        const TVector<double> &vec = eigen.EigenVecs[k];
        double lambda = Max<double>(0, eigen.EigenArr[k]);
        double xInv = 1 / (lambda + reg);;
        double dp = Dot(proj, vec);
        AddScaled(p, vec, dp * xInv);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TFunc>
void TransformEigenVals(TArray2D<double> *p, TFunc func)
{
    yint sz = p->GetXSize();
    Y_ASSERT(sz == p->GetYSize());

    TEigenDecomp eigen;
    eigen.Compute(*p);
    p->SetSizes(sz, sz);
    p->FillZero();
    for (yint k = 0; k < sz; ++k) {
        const TVector<double> &vec = eigen.EigenVecs[k];
        double lambda = eigen.EigenArr[k];
        double newLambda = func(lambda);
        for (yint y = 0; y < sz; ++y) {
            const double *vPtr = vec.data();
            double *resPtr = p->GetRow(y);
            double scale = vPtr[y] * newLambda;
            for (yint x = 0; x < sz; ++x) {
                resPtr[x] += vPtr[x] * scale;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T1, class T2, class T3>
void MatrixMult(const TArray2D<T1> &a, const TArray2D<T2> &b, TArray2D<T3> *res)
{
    yint xSize = b.GetXSize();
    yint ySize = a.GetYSize();
    yint sz = a.GetXSize();
    Y_ASSERT(sz == (yint)b.GetYSize());
    TArray2D<T3> rv;
    rv.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            T3 val = 0;
            for (yint k = 0; k < sz; ++k) {
                val += a[y][k] * b[k][x];
            }
            rv[y][x] = val;
        }
    }
    res->Swap(rv);
}


// multiply A by B^T, this way memory access is sequential
template <class T1, class T2, class T3>
void MatrixMultABT(const TArray2D<T1> &a, const TArray2D<T2> &b, TArray2D<T3> *res)
{
    yint xSize = b.GetYSize();
    yint ySize = a.GetYSize();
    yint sz = a.GetXSize();
    Y_ASSERT(sz == (yint)b.GetXSize());
    TArray2D<T3> rv;
    rv.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            T3 val = 0;
            const T1 *aRow = a.GetRow(y);
            const T2 *bRow = b.GetRow(x);
            const T1 *aRowEnd = aRow + sz;
            for (; aRow < aRowEnd; ++aRow, ++bRow) {
                val += *aRow * *bRow;
            }
            rv[y][x] = val;
        }
    }
    res->Swap(rv);
}


template<class TRes, class T1, class T2>
void Mul(vector<TRes> *pRes, const TArray2D<T1> &m, const vector<T2> &s)
{
    yint xSize = m.GetXSize();
    yint ySize = m.GetYSize();
    vector<TRes> &res = *pRes;
    res.resize(ySize);
    ASSERT(m.GetXSize() == s.size());
    for (int y = 0; y < ySize; ++y) {
        TRes fRes = 0;
        const T1 *mRowPtr = m.GetRow(y);
        const T2 *sPtr = s.data();
        const T2 *sPtrEnd = sPtr + xSize;
        while (sPtr < sPtrEnd) {
            fRes += *mRowPtr++ * *sPtr++;
        }
        res[y] = fRes;
    }
}


template<class T, class T1>
void MulLeft(TVector<T> *pRes, const TArray2D<T1> &m, const TVector<T> &s)
{
    TVector<T> &r = *pRes;
    yint xSize = m.GetXSize();
    yint ySize = m.GetYSize();
    ClearPodArray(&r, xSize);
    Y_ASSERT(ySize == s.size());
    for (int y = 0; y < ySize; ++y) {
        double mult = s[y];
        for (int x = 0; x < xSize; ++x) {
            r[x] += m[y][x] * mult;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// legacy
inline void InvertMatrix(TArray2D<double> *pMatrix)
{
    TransformEigenVals(pMatrix, [](double lambda) { return 1 / lambda; });
}

template<class T>
bool InvertRobust(TArray2D<T> *pMatrix)
{
    const T about0((T)1.0e-20);
    const T zero(0);
    ASSERT(pMatrix->GetXSize() == pMatrix->GetYSize());
    if (pMatrix->GetXSize() != pMatrix->GetYSize())
        return false;
    int nSize = pMatrix->GetXSize();
    TArray2D<T> right, left(*pMatrix);
    right.SetSizes(nSize, nSize);
    MakeE(&right);
    for (int i = 0; i < nSize; i++) {
        T diag = left[i][i], diag1 = left[i][i];
        int maxi = i;
        for (int k = i + 1; k < nSize; k++) {
            if (fabs(left[k][i]) > diag1) {
                diag1 = left[k][i];
                maxi = k;
            }
        }
        if (maxi != i && fabs(diag) * ((T)10) < fabs(diag1)) {
            for (int u = 0; u < nSize; u++) {
                left[i][u] += left[maxi][u];
                right[i][u] += right[maxi][u];
            }
            diag = left[i][i];
        }
        if (fabs(diag) < about0) {
            int h = i;
            while ((h < nSize - 1) && (fabs(diag) < about0)) {
                h++;
                if (fabs(left[h][i]) > about0) {
                    for (int u = 0; u < nSize; u++) {
                        left[i][u] += left[h][u];
                        right[i][u] += right[h][u];
                    }
                    diag = left[i][i];
                }
            }
            if (fabs(diag) < about0) {
                return false;
            }
        }
        T invdiag;
        invdiag = ((T)1) / diag;
        for (int j = 0; j < nSize; j++) {
            left[i][j] *= invdiag;
            right[i][j] *= invdiag;
        }
        for (int k = i + 1; k < nSize; k++) {
            T  koef = left[k][i];
            T *le = &left[k][0], *lei = &left[i][0];
            T *ri = &right[k][0], *rii = &right[i][0];
            T *lefin = le + nSize;
            if (koef != zero) {
                //	for(s=0; s<size; s++){ left[k][s]-=koef*left[i][s]; right[k][s]-=koef*right[i][s]; }
                while (le < lefin)
                {
                    le[0] -= koef * lei[0]; le++; lei++;
                    ri[0] -= koef * rii[0]; ri++; rii++;
                }
            }
        }
    }
    for (int i = nSize - 1; i >= 0; i--) {
        for (int k = 0; k < i; k++) {
            T  koef = left[k][i];
            T *le = &left[k][0], *lei = &left[i][0];
            T *ri = &right[k][0], *rii = &right[i][0];
            T *lefin = le + nSize;
            if (koef != zero) {
                //	for(s=0; s<size; s++){ left[k][s]-=koef*left[i][s]; right[k][s]-=koef*right[i][s];}
                while (le < lefin) {
                    le[0] -= koef * lei[0]; le++; lei++;
                    ri[0] -= koef * rii[0]; ri++; rii++;
                }
            }
        }
    }
    *pMatrix = right;
    return true;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
void FindSomeLinearSolution(const TArray2D<double> &_m, vector<double> &proj, vector<double> *pRes);
