#pragma once
#include "host_matrix.h"
#include <gpt/model_params/model_dim.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace NCuda
{
template <class TMatrixFloat>
class ICudaModelMatrixBase;
class TGraph;
class TStream;
class TMultiDeviceBuffers;
class TCudaMemoryAllocator;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
enum EAddToModel {
    GRADIENT_ACCUMULATE,
    GRADIENT_APPLY,
};

enum EModelMatrixMemory {
    MM_MEM_HOST,
    MM_MEM_DEVICE,
};

enum EModelMatrixDelayGradient {
    MM_SYNC_GRADIENT,
    MM_DELAY_GRADIENT,
};

enum ETensorParallelSplit {
    TPS_COPY,
    TPS_ROW,
    TPS_COLUMN,
    TPS_ROW_MOE,
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// backprop options
enum EBackpropMode {
    BM_NONE = 0,
    BM_GRAD_MOV = 0,
    BM_GRAD_ADD = 1,
    BM_GRAD_ACCUMULATE = 0,
    BM_GRAD_APPLY = 2,
    BM_COUNT = 4,
};

inline EBackpropMode SetFlag(EBackpropMode m1, EBackpropMode m2) {
    return EBackpropMode(m1 | m2);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IModelMatrixHostCompute
{
    virtual void GetFastFloatData(TArray2D<float> *p, bool noScale) const = 0;
    virtual void HostApplyDelta(const TArray2D<float> &data) = 0;
    virtual void HostAllowDelayedUpdates() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
class IModelMatrixBase : virtual public TThrRefBase
{
protected:
    THostModelMatrix Matr;
 
public:
    void Create(const TModelMatrix &data)
    {
        Matr.Create(data.GetXSize(), data.GetYSize(), data.GetSparsity());
        Matr.SetData(data);
    }

    // dimensions
    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    yint GetRowDispSize() const { return Matr.GetRowDispSize(); }

    // data
    void GetData(TModelMatrix *p) { Matr.GetData(p); }
    void SetData(const TModelMatrix &data) { Matr.SetData(data); }
    THostModelMatrix &GetHostMatrix() { return Matr; }

    // for cpu transformer
    virtual IModelMatrixHostCompute *GetHostCompute() = 0;

    // create cuda matrix
    virtual TIntrusivePtr<NCuda::ICudaModelMatrixBase<TMatrixFloat>> CreateCudaMatrix(yint deviceId) = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelVector : public TThrRefBase
{
    TVector<float> Vec;

    TModelVector() {}
    TModelVector(const TVector<float> &v) : Vec(v) {}
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// get quantized matrix data for reference cpu compute
template <class TMatrixFloat>
inline TArray2D<float> GetData(TIntrusivePtr<IModelMatrixBase<TMatrixFloat>> p)
{
    TArray2D<float> res;
    p->GetHostCompute()->GetFastFloatData(&res, false);
    return res;
}


template <class TMatrixFloat>
inline TArray2D<float> GetDataNoScale(TIntrusivePtr<IModelMatrixBase<TMatrixFloat>> p)
{
    TArray2D<float> res;
    p->GetHostCompute()->GetFastFloatData(&res, true);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IModelOps : public TThrRefBase
{
    virtual TIntrusivePtr<IModelMatrixBase<i8>> CreateModelMatrix(i8 *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) = 0;
    virtual TIntrusivePtr<IModelMatrixBase<e4m3>> CreateModelMatrix(e4m3 *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) = 0;
    virtual TIntrusivePtr<IModelMatrixBase<half>> CreateModelMatrix(half *, const TModelMatrix &data, EModelMatrixQuant quant,
        EModelMatrixDelayGradient delayGrad, EModelMatrixMemory mmm, ETensorParallelSplit tps) = 0;
    virtual void LaunchWorkers() = 0;
    virtual void InitDevice(yint deviceId, NCuda::TStream &stream, TPtrArg<NCuda::TCudaMemoryAllocator> cudaMemllocator) = 0;
    virtual void InitFwdPass(yint deviceId, TPtrArg<NCuda::TGraph> c, bool copyModelToDevice) = 0;
    virtual void InitBwdPass(yint deviceId, TPtrArg<NCuda::TGraph> c) = 0;
    virtual TPtrArg<NCuda::TMultiDeviceBuffers> GetMultiBuffers(yint deviceGroupId) = 0;

    virtual yint GetDeviceCount() = 0;
    virtual yint GetDeviceGroupCount() = 0;
    virtual bool IsHostMasterModel() = 0;
    virtual ETensorParallelSplit GetTensorSplit(ELayerType layerType, int matrId) { return TPS_COPY; }
    virtual ETensorParallelSplit GetTensorSplit(int matrId) { return TPS_COPY; }
    virtual EBackpropMode StartIteration(const TTrainingStep &step, EAddToModel addToModel) = 0;
    virtual yint GetBackpropModeCount() = 0;
    virtual void WaitActiveCompute() = 0;
    virtual void WaitDelayedCompute() = 0;
    virtual void ConvertMatrices() = 0; // for internal use, performs conversion
    virtual void CopyModelParamsToHost() = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// cpu gradient processing hooks
///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelMatrixBitDelta;

struct IBitDelta : virtual public TThrRefBase
{
    virtual yint GetBitDeltaXSize() const = 0;
    virtual yint GetBitDeltaYSize() const = 0;
    virtual TModelMatrixBitDelta &GetBitDelta() = 0;
    virtual void ExtractDelta(TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail) = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EOperation {
    MOP_NONE,
    MOP_APPLY_DELAYED_DELTA_WAIT,
    MOP_NEW_DELTA,
    MOP_WAIT,
    MOP_ADD_DELTA,
    MOP_ADD_BIT_DELTA,
    MOP_DELAY_DELTA,
    MOP_CONVERT,
};

struct TMatrixOpTracker : public TThrRefBase
{
    TVector<int> MatrixOpArr;

public:
    EOperation GetOp(yint matrixId) { return EOperation(MatrixOpArr[matrixId]); }
    void SetOp(yint matrixId, EOperation op) { MatrixOpArr[matrixId] = op; }

    yint AddMatrix()
    {
        MatrixOpArr.push_back(MOP_NONE);
        return YSize(MatrixOpArr);
    }

    bool StartDelayedDelta(int newOp)
    {
        bool rv = false;
        for (int &op : MatrixOpArr) {
            if (op == MOP_DELAY_DELTA) {
                op = newOp;
                rv = true;
            }
        }
        return rv;
    }

    void ConvertMatrices()
    {
        for (int &op : MatrixOpArr) {
            Y_VERIFY(op == MOP_NONE);
            op = MOP_CONVERT;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IMMDeltaHook : public TThrRefBase
{
    virtual void OnDelta() = 0;
};

struct IMMDeltaHookGen : public TThrRefBase
{
    virtual IMMDeltaHook *CreateDeltaHook(yint matrixId, TPtrArg<IBitDelta> p, TPtrArg<TMatrixOpTracker> matrixOps) = 0;
    virtual void OnIterationStart() = 0;
};
