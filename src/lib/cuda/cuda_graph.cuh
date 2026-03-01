#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_arrays.h"
#include "cuda_util.cuh"

namespace NCuda
{
class TGraph;

enum EOpDep
{
    DEP_NONE = 0,
    DEP_READ = 1, // op reads data
    DEP_READWRITE = 3, // op modifies or rewrites data, other writes should be complete before this op
    DEP_OVERWRITE = DEP_READWRITE, // no concurrent writes are allowed

    DEP_IS_READ = 1,
    DEP_IS_WRITE = 2,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline void EraseElement(TVector<T> *p, const T &elem)
{
    yint dst = 0;
    for (yint k = 0, sz = YSize(*p); k < sz; ++k) {
        if ((*p)[k] != elem) {
            (*p)[dst++] = (*p)[k];
        }
    }
    p->resize(dst);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TGraphOp : public TThrRefBase
{
    bool NeedSetParams = false;
    bool IsEnabled = true;
    bool TargetIsEnabled = true;
    TVector<const void *> ReadSet;
    TVector<const void *> WriteSet;
    TVector<TIntrusivePtr<TGraphOp>> DepArr;
    TVector<TIntrusivePtr<TCudaMemoryPool>> MemPools;

protected:
    cudaGraph_t Graph = 0;
    cudaGraphNode_t Node = 0;

    virtual void SetParams(cudaGraphExec_t) {}
    virtual void CreateNode() = 0;
    virtual TIntrusivePtr<TCudaMemoryPool> GetNewMemPool()
    {
        return nullptr;
    }
    virtual void ClearNewMemPool(bool doClear) {}
    void ParamsUpdated()
    {
        NeedSetParams = false;
    }

public:
    TGraphOp(cudaGraph_t graph) : Graph(graph) {}
    void OnParamsChange()
    {
        NeedSetParams = true;
    }
    void UpdateParams(cudaGraphExec_t execGraph)
    {
        if (TargetIsEnabled != IsEnabled) {
            IsEnabled = TargetIsEnabled;
            Y_VERIFY(cudaGraphNodeSetEnabled(execGraph, Node, IsEnabled ? 1 : 0) == cudaSuccess);
        }
        if (NeedSetParams) {
            SetParams(execGraph);
        }
    }
    void SetEnabled(bool b)
    {
        TargetIsEnabled = b;
    }

    // deps
    void AddDep(TGraphOp *op)
    {
        DepArr.push_back(op);
    }
    void AddDep(EOpDep dep, const void *p)
    {
        if (dep & DEP_IS_READ) {
            ReadSet.push_back(p);
        }
        if (dep & DEP_IS_WRITE) {
            WriteSet.push_back(p);
        }
    }
    void AddDepOverwrite(const void *p)
    {
        // we want to wait other writes to complete, can do it by adding to ReadSet
        ReadSet.push_back(p);
        WriteSet.push_back(p);
    }
    void AddMemPool(TPtrArg<TCudaMemoryPool> pool)
    {
        if (pool.Get()) {
            MemPools.push_back(pool.Get());
        }
    }

    friend class TGraph;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TKernelEnable 
{
    TVector<TIntrusivePtr<TGraphOp>> OpArr;

public:
    void Link(TGraphOp &op) { OpArr.push_back(&op); }
    void SetEnabled(bool b)
    {
        for (auto &x : OpArr) {
            x->SetEnabled(b);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class TKernelParameterStorage : public TThrRefBase
{
    struct TRef
    {
        T *Data;
        TIntrusivePtr<TGraphOp> Op;
        yint Arg = 0;
        yint Arg2 = 0;

        TRef(T *p, TGraphOp *op, yint arg, yint arg2) : Data(p), Op(op), Arg(arg), Arg2(arg2) {}
    };
    T Val = T();
    TVector<TRef> ValArr;
    TVector<TRef> DivCeilArr;
    TVector<TRef> RoundUpArr;
public:
    void Set(const T &newValue)
    {
        if (Val != newValue) {
            Val = newValue;
            for (const TRef &x : ValArr) {
                *x.Data = newValue;
                x.Op->OnParamsChange();
            }
            for (const TRef &x : DivCeilArr) {
                yint tileCount = DivCeil(newValue, x.Arg);
                *x.Data = tileCount * x.Arg2;
                x.Op->OnParamsChange();
            }
            for (const TRef &x : RoundUpArr) {
                *x.Data = RoundUp(newValue, x.Arg);
                x.Op->OnParamsChange();
            }
        }
    }
    const T &Get() const
    {
        return Val;
    }
    void AddRef(T *p, TGraphOp *op)
    {
        ValArr.push_back(TRef(p, op, 0, 0));
        *p = 0;
    }
    void AddDivCeilRef(T *p, TGraphOp *op, yint tile, yint largeTile)
    {
        DivCeilArr.push_back(TRef(p, op, largeTile, largeTile / tile));
        *p = 0;
    }
    void AddRoundUpRef(T *p, TGraphOp *op, yint tile)
    {
        RoundUpArr.push_back(TRef(p, op, tile, 0));
        *p = 0;
    }
};


template <class T>
struct IKernelParameter
{
    virtual void AddRef(T *p, TGraphOp *op) const = 0;
};


template <class T>
struct TKernelParameter : public IKernelParameter<T>
{
    TIntrusivePtr<TKernelParameterStorage<T>> Param;
public:
    TKernelParameter() { Param = new TKernelParameterStorage<T>(); }
    void Set(const T &newValue) { Param->Set(newValue); }
    const T &Get() const { return Param->Get(); }
    void AddRef(T *p, TGraphOp *op) const override { Param->AddRef(p, op); }
};


template <class T>
class TKernelParameterDivCeil : public IKernelParameter<T>
{
    TIntrusivePtr<TKernelParameterStorage<T>> Param;
    yint Tile = 0;
    yint LargeTile = 0;
public:
    TKernelParameterDivCeil(const TKernelParameter<T> &param, yint tile, yint largeTile)
        : Param(param.Param), Tile(tile), LargeTile(largeTile)
    {
    }
    void AddRef(T *p, TGraphOp *op) const override { Param->AddDivCeilRef(p, op, Tile, LargeTile); }
};


template <class T>
class TKernelParameterRoundUp : public IKernelParameter<T>
{
    TIntrusivePtr<TKernelParameterStorage<T>> Param;
    yint Tile = 0;
public:
    TKernelParameterRoundUp(const TKernelParameter<T> &param, yint tile) : Param(param.Param), Tile(tile) {}
    void AddRef(T *p, TGraphOp *op) const override { Param->AddRoundUpRef(p, op, Tile); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Kernel call op
class TKernelOp : public TGraphOp
{
    enum {
        PARAM_BUF_SIZE = 2048
    };
    cudaKernelNodeParams CudaParams;
    char KParamBuf[PARAM_BUF_SIZE];
    TVector<void *> KParamList;
    yint KParamPtr = 0;
    bool IsStructParaam = false;
    yint StructOffset = 0;

    void CreateNode() override
    {
        if (CudaParams.sharedMemBytes > 48 * 1024) {
            Y_VERIFY(cudaSuccess == cudaFuncSetAttribute(CudaParams.func, cudaFuncAttributeMaxDynamicSharedMemorySize, CudaParams.sharedMemBytes));
        }
        CudaParams.kernelParams = KParamList.data();
        Y_ASSERT(Node == 0);
        cudaError_t err = cudaGraphAddKernelNode(&Node, Graph, 0, 0, &CudaParams);
        Y_VERIFY(err == cudaSuccess);
        ParamsUpdated();
    }

    void SetParams(cudaGraphExec_t execGraph)
    {
        cudaError_t err = cudaGraphExecKernelNodeSetParams(execGraph, Node, &CudaParams);
        Y_VERIFY(err == cudaSuccess);
        ParamsUpdated();
    }

    template <class T>
    struct TAlignmentComputer
    {
        char C;
        T Data;
    };

    template <class T>
    void AddParam(const T &val, EOpDep dep)
    {
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // allow write to arrays only atm
        if (IsStructParaam) {
            // alignment
            TAlignmentComputer<T> *pComputer = nullptr;
            yint alignSize = (char*) &pComputer->Data - (char*)nullptr;
            int alignedOffset = (StructOffset + (alignSize - 1)) & ~(alignSize - 1);
            KParamPtr += alignedOffset - StructOffset;
            StructOffset = alignedOffset + sizeof(T);
        } else {
            KParamList.push_back(KParamBuf + KParamPtr);
        }
        Y_VERIFY(KParamPtr + sizeof(T) <= PARAM_BUF_SIZE);
        T *pParamPlace = (T *)(KParamBuf + KParamPtr);
        memcpy(pParamPlace, &val, sizeof(T));
        KParamPtr += sizeof(T);
    }

    template <class T>
    void AddParam(const TCudaPOD<T> &param, EOpDep dep)
    {
        void *pDeviceData = param.GetDevicePtr();
        const void *owner = param.GetOwner();
        Y_ASSERT(owner);
        AddParam(pDeviceData, DEP_NONE);
        AddDep(dep, owner); // generalize dependency to whole owner object
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(const TCudaVectorFragment<T> &param, EOpDep dep)
    {
        const void *owner = param.GetOwner();
        Y_ASSERT(owner);
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDep(dep, owner);
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(const TCudaVector<T> &param, EOpDep dep)
    {
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDep(dep, param.GetOwner());
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(const TCuda2DArrayFragment<T> &param, EOpDep dep)
    {
        const void *owner = param.GetOwner();
        Y_ASSERT(owner);
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDep(dep, owner);
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(const TCuda2DArray<T> &param, EOpDep dep)
    {
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDep(dep, param.GetOwner());
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddKernelParam(const IKernelParameter<T> &param, EOpDep dep)
    {
        T *pParamPlace = (T *)(KParamBuf + KParamPtr);
        AddParam(T(), DEP_NONE);
        param.AddRef(pParamPlace, this);
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // writing to parameters is prohibited
    }
    template <class T>
    void AddParam(const TKernelParameter<T> &param, EOpDep dep) { AddKernelParam(param, dep); }
    template <class T>
    void AddParam(const TKernelParameterDivCeil<T> &param, EOpDep dep) { AddKernelParam(param, dep); }
    template <class T>
    void AddParam(const TKernelParameterRoundUp<T> &param, EOpDep dep) { AddKernelParam(param, dep); }

public:
    TKernelOp(cudaGraph_t graph, void *kernel) : TGraphOp(graph)
    {
        Zero(CudaParams);
        CudaParams.func = kernel;
        CudaParams.blockDim = dim3(WARP_SIZE);
        CudaParams.gridDim = dim3(1);
    }

    // Shmem
    TKernelOp &Shmem(int sz)
    {
        CudaParams.sharedMemBytes = sz;
        return *this;
    }

    // Grid
    TKernelOp &Grid(int x, int y = 1, int z = 1)
    {
        CudaParams.gridDim = dim3(x, y, z);
        return *this;
    }
    TKernelOp &Grid(const IKernelParameter<int> &x, int y = 1, int z = 1)
    {
        CudaParams.gridDim = dim3(0, y, z);
        Y_ASSERT(sizeof(CudaParams.gridDim.x) == sizeof(int));
        x.AddRef((int*)&CudaParams.gridDim.x, this);
        return *this;
    }
    TKernelOp &Grid(int x, const IKernelParameter<int> &y, int z = 1)
    {
        CudaParams.gridDim = dim3(x, 0, z);
        Y_ASSERT(sizeof(CudaParams.gridDim.y) == sizeof(int));
        y.AddRef((int *)&CudaParams.gridDim.y, this);
        return *this;
    }
    TKernelOp &Grid(int x, int y, const IKernelParameter<int> &z)
    {
        CudaParams.gridDim = dim3(x, y, 0);
        Y_ASSERT(sizeof(CudaParams.gridDim.z) == sizeof(int));
        z.AddRef((int *)&CudaParams.gridDim.z, this);
        return *this;
    }
    TKernelOp &FullGrid(TPtrArg<TGraph> c, int y = 1);
    TKernelOp &GridZ(int z)
    {
        CudaParams.gridDim.z = z;
        return *this;
    }

    // Block
    TKernelOp &Block(int x, int y = 1, int z = 1)
    {
        CudaParams.blockDim = dim3(x, y, z);
        return *this;
    }
    TKernelOp &Block(int x, const IKernelParameter<int> &y, int z = 1)
    {
        CudaParams.blockDim = dim3(x, 0, z);
        Y_ASSERT(sizeof(CudaParams.blockDim.y) == sizeof(int));
        y.AddRef((int *)&CudaParams.blockDim.y, this);
        return *this;
    }

    // Struct
    TKernelOp &Struct()
    {
        IsStructParaam = true;
        char *pParamPlace = (KParamBuf + KParamPtr);
        KParamList.push_back(pParamPlace);
        StructOffset = 0;
        return *this;
    }
    TKernelOp &Params()
    {
        IsStructParaam = false;
        return *this;
    }

    // read dependency
    template <typename T>
    TKernelOp &DepRead(const T &param)
    {
        AddDep(DEP_READ, param.GetOwner());
        AddMemPool(param.GetMemPool());
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &DepRead(const T &param, TRest &&...x)
    {
        AddDep(DEP_READ, param.GetOwner());
        AddMemPool(param.GetMemPool());
        return DepRead(x...);
    }

    // write dependency
    template <typename T>
    TKernelOp &DepWrite(const T &param)
    {
        AddDep(DEP_READWRITE, param.GetOwner());
        AddMemPool(param.GetMemPool());
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &DepWrite(const T &param, TRest &&...x)
    {
        AddDep(DEP_READWRITE, param.GetOwner());
        AddMemPool(param.GetMemPool());
        return DepWrite(x...);
    }

    // op dep
    TKernelOp &Dep(TGraphOp &op)
    {
        AddDep(&op);
        return *this;
    }
    TKernelOp &Dep(TPtrArg<TGraphOp> op)
    {
        if (op) {
            AddDep(op);
        }
        return *this;
    }
    TKernelOp &Chain(yint k, TPtrArg<TGraph> c);
    TKernelOp &Link(TKernelEnable &ee)
    {
        ee.Link(*this);
        return *this;
    }

    // pass read only kernel parameters
    template <typename T>
    TKernelOp &Read(const T &param)
    {
        AddParam(param, DEP_READ);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &Read(const T &param, TRest&&... x)
    {
        AddParam(param, DEP_READ);
        return Read(x...);
    }

    // pass kernel target params 
    template <typename T>
    TKernelOp &Write(T *param)
    {
        AddParam(*param, DEP_READWRITE);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &Write(T *param, TRest... x)
    {
        AddParam(*param, DEP_READWRITE);
        return Write(x...);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TSetMemPoolOp : public TGraphOp
{
    TIntrusivePtr<TCudaMemoryPool> Pool;
    cudaMemsetParams Params;
    bool NeedClearMem = false;

    void CreateNode() override
    {
#ifdef NDEBUG
        NeedClearMem = false;
#endif
        cudaError_t err = cudaSuccess;
        if (NeedClearMem && Pool->GetMemSize() > 0) {
            // garbage fill memory to try to catch using unitialized RAM
            Zero(Params);
            Params.dst = Pool->GetDevicePtr(0);
            Params.elementSize = 1;
            Params.height = 1;
            Params.value = 0xff; // results in memory filled with NaNs
            Params.width = Pool->GetMemSize();
            err = cudaGraphAddMemsetNode(&Node, Graph, 0, 0, &Params);
        } else {
            err = cudaGraphAddEmptyNode(&Node, Graph, 0, 0);
        }
        Y_VERIFY(err == cudaSuccess);
    }
    TIntrusivePtr<TCudaMemoryPool> GetNewMemPool() override
    {
        return Pool;
    }
    void ClearNewMemPool(bool doClear)
    {
        NeedClearMem = doClear;
    }

public:
    TSetMemPoolOp(cudaGraph_t graph, TIntrusivePtr<TCudaMemoryPool> pool) : TGraphOp(graph), Pool(pool)
    {
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
const int KERNEL_COPY_BLOCK = 32;
__global__ void KernelMemsetImpl(int4 *dst, int lineWidthInBytes, int dstYSize, int ySize);
__global__ void KernelCopyImpl(int4 *dst, int4 *src, int lineWidthInBytes, int srcYSize, int dstYSize, int ySize);
__global__ void KernelCopy2DImpl(int4 *dst, int4 *src, int dstStride, int srcStride, int rowWidthInBytes, TCuda1DPtr<int> rowRename, int rowCount);


///////////////////////////////////////////////////////////////////////////////////////////////////
// default kernel block size
THashMap<TString, dim3> &GetKernelBlockSize();
#define KERNEL_BLOCK_SIZE(a, ...) namespace { struct TKernelBlock##a { TKernelBlock##a() {\
    Y_ASSERT(GetKernelBlockSize().find(KERNEL_UNIT #a) == GetKernelBlockSize().end());\
    GetKernelBlockSize()[KERNEL_UNIT #a] = dim3(__VA_ARGS__);\
} } setKernelBlockSize##a; }


///////////////////////////////////////////////////////////////////////////////////////////////////
enum {
    CHAIN_OP_PCI_TO_DEVICE,
    CHAIN_OP_PCI_TO_HOST,
    CHAIN_OP_COUNT,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Graph
struct TCudaOpDependencies;
class TGraph : public TThrRefBase
{
    cudaGraph_t Graph;
    cudaGraphExec_t ExecGraph = 0;
    TVector<TIntrusivePtr<TGraphOp>> OpArr;
    TVector<yint> DepChainBreaks;
    TIntrusivePtr<TGraphOp> ChainOpPrev[CHAIN_OP_COUNT];
    yint FullGridSize = 0;
    yint LinDepFragmentFlag = 0;
    TIntrusivePtr<TKernelOp> LinDepPrevOp;

    void AddMemPoolDeps(TCudaOpDependencies *pDep);
    void AddLinearDeps(TCudaOpDependencies *pDep);
    void AddDeps(TCudaOpDependencies *pDep);
    void CreateExecGraph();

    ~TGraph()
    {
        if (ExecGraph != 0) {
            cudaGraphExecDestroy(ExecGraph);
        }
        cudaGraphDestroy(Graph);
    }

    template <class F>
    void ForEachDepChain(F func) 
    {
        yint beg = 0;
        for (yint fin : DepChainBreaks) {
            if (beg != fin) {
                func(beg, fin);
            }
            beg = fin;
        }
    }

public:
    TGraph();

    void FinishDependencyChain()
    {
        Y_VERIFY(LinDepFragmentFlag == 0);
        DepChainBreaks.push_back(YSize(OpArr));
    }
    void StartLinDepFragment()
    {
        if (LinDepFragmentFlag++ == 0) {
            LinDepPrevOp = 0;
        }
    }
    void FinishLinDepFragment() { --LinDepFragmentFlag; }

    void ReserveSM(yint n)
    {
        Y_VERIFY(FullGridSize > n);
        FullGridSize -= n;
    }

    yint GetFullGridSize() const { return FullGridSize; }

    bool HasOp(TGraphOp *op) { return IsInSet(OpArr, op); }

    TKernelOp &CudaCallImplementation(const TString &kernelUnit, const TString &kernelName, void *kernel)
    {
        TKernelOp *p = new TKernelOp(Graph, kernel);
        TString kernelFuncName = kernelName.substr(0, kernelName.find('<'));
        const THashMap<TString, dim3> &kernelBlockSize = GetKernelBlockSize();
        auto it = kernelBlockSize.find(kernelUnit + kernelFuncName);
        if (it != kernelBlockSize.end()) {
            p->Block(it->second.x, it->second.y, it->second.z);
        }
        if (LinDepFragmentFlag > 0) {
            p->Dep(PtrArg(LinDepPrevOp));
            LinDepPrevOp = p;
        }
        OpArr.push_back(p);
        return *p;
    }

    void ChainOp(yint cg, TGraphOp *op)
    {
        Y_VERIFY(cg >= 0 && cg < ARRAY_SIZE(ChainOpPrev));
        Y_ASSERT(HasOp(op));
        if (ChainOpPrev[cg].Get()) {
            op->AddDep(ChainOpPrev[cg].Get());
        }
        ChainOpPrev[cg] = op;
    }

    // memory ops
    void SetMemPool(TIntrusivePtr<TCudaMemoryPool> pool)
    {
        OpArr.push_back(new TSetMemPoolOp(Graph, pool));
    }

    // memset
    template <class T, class TYSize>
    TKernelOp &ClearMem(T *dst, TYSize &&ySize)
    {
        TMemoryBlob dstBlob = dst->GetDeviceMem();
        TIntrusivePtr<TKernelOp> p = new TKernelOp(Graph, (void *)KernelMemsetImpl);
        (*p).Read(dstBlob.Ptr, dstBlob.Stride, (int)dstBlob.YSize, ySize);
        (*p).Block(WARP_SIZE, KERNEL_COPY_BLOCK);
        // deps
        p->AddDepOverwrite(dst->GetOwner());
        p->AddMemPool(dst->GetMemPool());
        OpArr.push_back(p.Get());
        return *p.Get();
    }
    template <class T>
    TKernelOp &ClearMem(T *dst)
    {
        TMemoryBlob dstBlob = dst->GetDeviceMem();
        return ClearMem(dst, (int)dstBlob.YSize);
    }

    // memcpy, use kernel to copy arrays (avoid WDDM induced delays on Windows)
    template <class TDst, class TSrc, class TYSize>
    TKernelOp &KernelCopy(TDst *dst, const TSrc &src, TYSize &&ySize)
    {
        TMemoryBlob srcBlob = src.GetDeviceMem();
        TMemoryBlob dstBlob = dst->GetDeviceMem();
        Y_VERIFY(srcBlob.Stride == dstBlob.Stride);
        TIntrusivePtr<TKernelOp> p = new TKernelOp(Graph, (void*)KernelCopyImpl);
        (*p).Read(dstBlob.Ptr, srcBlob.Ptr, srcBlob.Stride, (int)srcBlob.YSize, (int)dstBlob.YSize, ySize);
        (*p).Block(WARP_SIZE, KERNEL_COPY_BLOCK);
        // deps
        p->AddDep(DEP_READ, src.GetOwner());
        p->AddDepOverwrite(dst->GetOwner());
        p->AddMemPool(src.GetMemPool());
        p->AddMemPool(dst->GetMemPool());
        OpArr.push_back(p.Get());
        return *p.Get();
    }
    template <class TDst, class TSrc>
    TKernelOp &KernelCopy(TDst *dst, const TSrc &src)
    {
        TMemoryBlob srcBlob = src.GetDeviceMem();
        TMemoryBlob dstBlob = dst->GetDeviceMem();
        Y_VERIFY(srcBlob.IsSameSize(dstBlob));
        return KernelCopy(dst, src, (int)srcBlob.YSize);
    }
    template <class TDst, class TSrc>
    void KernelCopy2D(TDst *dst, const TSrc &src, TCudaVector<int> &rowRename)
    {
        typedef typename TDst::TElem TElem;
        TCuda2DPtr<TElem> srcPtr = src.GetDevicePtr();
        TCuda2DPtr<TElem> dstPtr = dst->GetDevicePtr();
        Y_VERIFY(dst->GetXSize() >= src.GetXSize() && dst->GetYSize() >= rowRename.GetSize());
        TIntrusivePtr<TKernelOp> p = new TKernelOp(Graph, (void *)KernelCopy2DImpl);
        (*p).Read(dstPtr.Data, srcPtr.Data, dstPtr.StrideInBytes, srcPtr.StrideInBytes, (int)(sizeof(TElem) * src.GetXSize()), rowRename,
            (int)rowRename.GetSize());
        (*p).Block(WARP_SIZE, KERNEL_COPY_BLOCK);
        // deps
        p->AddDep(DEP_READ, src.GetOwner());
        p->AddDepOverwrite(dst->GetOwner());
        p->AddMemPool(src.GetMemPool());
        p->AddMemPool(dst->GetMemPool());
        OpArr.push_back(p.Get());
    }

    // run
    void Run(TStream &stream);
};

#define CudaCall(c, kernel, ...) c->CudaCallImplementation(KERNEL_UNIT, #kernel, (void*)kernel, ##__VA_ARGS__)

}

///////////////////////////////////////////////////////////////////////////////////////////////////
inline NCuda::TKernelParameterDivCeil<int> DivCeil(const NCuda::TKernelParameter<int> &param, yint tile)
{
    return NCuda::TKernelParameterDivCeil<int>(param, tile, tile);
}

inline NCuda::TKernelParameterDivCeil<int> DivCeil(const NCuda::TKernelParameter<int> &param, yint tile, yint largeTile)
{
    return NCuda::TKernelParameterDivCeil<int>(param, tile, largeTile);
}

inline NCuda::TKernelParameterRoundUp<int> RoundUp(const NCuda::TKernelParameter<int> &param, yint tile)
{
    return NCuda::TKernelParameterRoundUp<int>(param, tile);
}
