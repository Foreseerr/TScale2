#include <util/pch.h>
#include "cuda_graph.cuh"
#include "cuda_init.h"


namespace NCuda
{
__global__ void KernelMemsetImpl(int4 *dst, int lineWidthInBytes, int dstYSize, int ySize)
{
    if (ySize > dstYSize) {
        CUDA_ASSERT(0 && "kernel memset buffer overrun");
        return;
    }
    int thrOffset = blockIdx.x * WARP_SIZE * blockDim.y + threadIdx.y * WARP_SIZE + threadIdx.x;
    int lenBytes = lineWidthInBytes * ySize;
    int len = lenBytes / sizeof(*dst);
    int4 zero = make_int4(0, 0, 0, 0);
    for (int blkOffset = 0; blkOffset < len; blkOffset += WARP_SIZE * blockDim.y * gridDim.x) {
        int offset = blkOffset + thrOffset;
        if (offset < len) {
            dst[offset] = zero;
        }
    }
    int tailStart = len * sizeof(*dst);
    int h = threadIdx.x;
    if (blockIdx.x == 0 && threadIdx.y == 0 && tailStart + h < lenBytes) {
        ((char *)dst)[tailStart + h] = 0;
    }
    __threadfence_system(); // flush cache
}


__global__ void KernelCopyImpl(int4 *dst, int4 *src, int lineWidthInBytes, int srcYSize, int dstYSize, int ySize)
{
    if (ySize > srcYSize || ySize > dstYSize) {
        CUDA_ASSERT(0 && "kernel copy buffer overrun");
        return;
    }
    int thrOffset = blockIdx.x * WARP_SIZE * blockDim.y + threadIdx.y * WARP_SIZE + threadIdx.x;
    int lenBytes = lineWidthInBytes * ySize;
    int len = lenBytes / sizeof(*src);
    for (int blkOffset = 0; blkOffset < len; blkOffset += WARP_SIZE * blockDim.y * gridDim.x) {
        int offset = blkOffset + thrOffset;
        if (offset < len) {
            dst[offset] = src[offset];
        }
    }
    int tailStart = len * sizeof(*src);
    int h = threadIdx.x;
    if (blockIdx.x == 0 && threadIdx.y == 0 && tailStart + h < lenBytes) {
        ((char *)dst)[tailStart + h] = ((const char *)src)[tailStart + h];
    }
    __threadfence_system(); // flush cache
}


__global__ void KernelCopy2DImpl(int4 *dst, int4 *src, int dstStride, int srcStride, int rowWidthInBytes, TCuda1DPtr<int> rowRename,
    int rowCount)
{
    int rowWidth16 = rowWidthInBytes / 16;
    int rowTail = rowWidthInBytes - rowWidth16 * 16;
    int h = threadIdx.x;
    for (int y = threadIdx.y; y < rowCount; y += blockDim.y) {
        int ySrc = rowRename[y];
        int4 *rowDst = AdvancePtr(dst, dstStride * y);
        int4 *rowSrc = AdvancePtr(src, srcStride * ySrc);
        for (int x = h; x < rowWidth16; x += WARP_SIZE) {
            rowDst[x] = rowSrc[x];
        }
        if (h < rowTail) {
            int offset = rowWidth16 * 16 + h;
            *AdvancePtr((ui8 *)rowDst, offset) = *AdvancePtr((ui8 *)rowSrc, offset);
        }
    }
    __threadfence_system(); // flush cache
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// TKernelOp
TKernelOp &TKernelOp::FullGrid(TPtrArg<TGraph> c, int y)
{
    return Grid(c->GetFullGridSize(), y);
}


TKernelOp &TKernelOp::Chain(yint k, TPtrArg<TGraph> c)
{
    c->ChainOp(k, this);
    return *this;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
THashMap<TString, dim3> &GetKernelBlockSize()
{
    static THashMap<TString, dim3> kernelBlockSize;
    return kernelBlockSize;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// TGraph
TGraph::TGraph()
{
    cudaGraphCreate(&Graph, 0);
    FullGridSize = GetCudaSMCount();
}


struct TCudaOpDependencies
{
    struct TLink
    {
        int From = 0;
        int To = 0;

        TLink() {}
        TLink(int from, int to) : From(from), To(to) {}
        bool operator==(const TLink &x) const { return From == x.From && To == x.To; }
    };
    struct TLinkHash
    {
        yint operator()(const TLink &a) const
        {
            return a.From * 0xbf1765d37f0cf121ll + a.To * 0x49260d554d12eb93ll;
        }
    };

public:
    TVector<TLink> Links;

public:
    void AddDep(int from, int to)
    {
        if (from != to) {
            Links.push_back(TLink(from, to));
        }
    }

    void EraseDuplicates()
    {
        THashMap<TLink, bool, TLinkHash> allLinks;
        for (const TLink &lnk : Links) {
            allLinks[lnk];
        }
        Links.resize(0);
        for (auto it = allLinks.begin(); it != allLinks.end(); ++it) {
            Links.push_back(it->first);
        }
    }

    void TransitiveReduction(const TVector<yint> &depChainBreaks)
    {
        TVector<TLink> crossLinks;
        TVector<TLink> newLinks;
        yint beg = 0;
        for (yint fin : depChainBreaks) {
            yint sz = fin - beg;
            
            // collect inbound links
            TVector<TVector<int>> linksTo;
            linksTo.resize(sz);
            for (const TLink &lnk : Links) {
                bool hasFrom = (lnk.From >= beg && lnk.From < fin);
                bool hasTo = (lnk.To >= beg && lnk.To < fin);
                if (hasFrom && hasTo) {
                    linksTo[lnk.To - beg].push_back(lnk.From - beg);
                } else if (hasFrom || hasTo) {
                    // cross links
                    crossLinks.push_back(lnk);
                }
            }
            
            // reach matrix
            TArray2D<bool> reach;
            reach.SetSizes(sz, sz);
            reach.FillZero();
            for (yint k = 0; k < sz; ++k) {
                reach[k][k] = true;
            }
            
            // add links
            for (yint to = 0; to < sz; ++to) {
                Sort(linksTo[to].begin(), linksTo[to].end(), [](int a, int b) { return a > b; });
                for (int from : linksTo[to]) {
                    if (!reach[to][from]) {
                        newLinks.push_back(TLink(from + beg, to + beg));
                        for (yint z = 0; z <= from; ++z) {
                            reach[to][z] |= reach[from][z];
                        }
                    }
                }
            }

            // next chain
            beg = fin;
        }
        Links = crossLinks;
        EraseDuplicates();
        Links.insert(Links.end(), newLinks.begin(), newLinks.end());
    }
};


struct TGraphMemPoolInfo
{
    int ActivateOp = -1;
    TVector<int> UseOps;
};

void TGraph::AddMemPoolDeps(TCudaOpDependencies *pDep)
{
    ForEachDepChain([&](yint beg, yint fin) {
        TCudaActivePoolTracker activeTracker;
        THashMap<TCudaMemoryPool *, TGraphMemPoolInfo, TPtrHash> activePools;
        for (yint k = beg; k < fin; ++k) {
            TGraphOp *op = OpArr[k].Get();
            for (auto &memPool : op->MemPools) {
                auto it = activePools.find(memPool.Get());
                if (it == activePools.end()) {
                    Y_VERIFY(0 && "should use only active mem pools");
                }
                TGraphMemPoolInfo &info = it->second;
                pDep->AddDep(info.ActivateOp, k);
                info.UseOps.push_back(k);
            }
            TIntrusivePtr<TCudaMemoryPool> newPool = op->GetNewMemPool();
            if (newPool.Get()) {
                TCudaActivePoolTracker::TPools addPools, delPools, requiredPools;
                activeTracker.ActivatePool(newPool.Get(), &addPools, &delPools, &requiredPools);

                Y_VERIFY(addPools.empty() || addPools.find(newPool.Get()) != addPools.end());
                op->ClearNewMemPool(!newPool->IsUsedForAsyncIO() && !addPools.empty());

                for (auto it = addPools.begin(); it != addPools.end(); ++it) {
                    TGraphMemPoolInfo &info = activePools[it->first];
                    info.ActivateOp = k;
                    info.UseOps.push_back(k);
                }

                for (auto it = delPools.begin(); it != delPools.end(); ++it) {
                    auto dd = activePools.find(it->first);
                    Y_ASSERT(dd != activePools.end());
                    // pool is no longer active, all operations using it should be complete
                    for (int useOp : dd->second.UseOps) {
                        pDep->AddDep(useOp, k);
                    }
                    activePools.erase(dd);
                }

                for (auto it = requiredPools.begin(); it != requiredPools.end(); ++it) {
                    Y_ASSERT(activePools.find(it->first) != activePools.end());
                    TGraphMemPoolInfo &info = activePools[it->first];
                    pDep->AddDep(info.ActivateOp, k);
                    info.UseOps.push_back(k);
                }
            }
        }
    });
}


void TGraph::AddLinearDeps(TCudaOpDependencies *pDep)
{
    ForEachDepChain([&](yint beg, yint fin) {
        for (yint k = beg + 1; k < fin; ++k) {
            pDep->AddDep(k - 1, k);
        }
    });
}


void TGraph::AddDeps(TCudaOpDependencies *pDep)
{
    THashMap<const void *, int, TPtrHash> opIndex;
    for (yint k = 0; k < YSize(OpArr); ++k) {
        TGraphOp *op = OpArr[k].Get();
        opIndex[op] = k;
    }
    // add explicit deps
    for (yint k = 0; k < YSize(OpArr); ++k) {
        TGraphOp *op = OpArr[k].Get();
        for (auto &dep : op->DepArr) {
            auto it = opIndex.find(dep.Get());
            if (it != opIndex.end()) {
                pDep->AddDep(it->second, k);
            }
        }
    }

    ForEachDepChain([&](yint beg, yint fin) {
        // read after write
        THashMap<const void *, TVector<int>, TPtrHash> writeOp;
        for (yint k = beg; k < fin; ++k) {
            TGraphOp *op = OpArr[k].Get();
            // read should start after write is complete
            for (const void *data : op->ReadSet) {
                auto it = writeOp.find(data);
                if (it != writeOp.end()) {
                    for (int writeOpIndex : it->second) {
                        pDep->AddDep(writeOpIndex, k);
                    }
                }
            }
            for (const void *data : op->WriteSet) {
                writeOp[data].push_back(k);
            }
        }

        // write should start after read is complete
        writeOp.clear();
        for (yint k = fin  - 1; k >= beg; --k) {
            TGraphOp *op = OpArr[k].Get();
            for (const void *data : op->ReadSet) {
                auto it = writeOp.find(data);
                if (it != writeOp.end()) {
                    for (int writeOpIndex : it->second) {
                        pDep->AddDep(k, writeOpIndex);
                    }
                }
            }
            for (const void *data : op->WriteSet) {
                writeOp[data].push_back(k);
            }
        }
    });
}


void TGraph::CreateExecGraph()
{
    Y_ASSERT(ExecGraph == 0);

    TCudaOpDependencies dep;
    FinishDependencyChain();
    AddMemPoolDeps(&dep);
    //AddLinearDeps(&dep);
    AddDeps(&dep);
    dep.TransitiveReduction(DepChainBreaks);

    // create nodes
    TVector<cudaGraphNode_t> allNodes;
    for (const TIntrusivePtr<TGraphOp> &op : OpArr) {
        op->CreateNode();
        allNodes.push_back(op->Node);
    }

    // add dependencies
    if (!dep.Links.empty()) {
        TVector<cudaGraphNode_t> fromArr;
        TVector<cudaGraphNode_t> toArr;
        for (const TCudaOpDependencies::TLink &lnk : dep.Links) {
            fromArr.push_back(allNodes[lnk.From]);
            toArr.push_back(allNodes[lnk.To]);
        }
#if (CUDART_VERSION >= 13000)
        Y_VERIFY(cudaGraphAddDependencies(Graph, fromArr.data(), toArr.data(), NULL, YSize(fromArr)) == cudaSuccess);
#else
        Y_VERIFY(cudaGraphAddDependencies(Graph, fromArr.data(), toArr.data(), YSize(fromArr)) == cudaSuccess);
#endif
    }

    //cudaGraphDebugDotPrint(Graph, "d:/g.dot", cudaGraphDebugDotFlagsVerbose);
    cudaError_t err = cudaGraphInstantiateWithFlags(&ExecGraph, Graph, 0);
    Y_VERIFY(err == cudaSuccess);
}


void TGraph::Run(TStream &stream)
{
    if (ExecGraph == 0) {
        CreateExecGraph();
    }
    for (TIntrusivePtr<TGraphOp> &op : OpArr) {
        op->UpdateParams(ExecGraph);
    }
    cudaError_t err = cudaGraphLaunch(ExecGraph, stream);
    Y_VERIFY(err == cudaSuccess);
}

}
