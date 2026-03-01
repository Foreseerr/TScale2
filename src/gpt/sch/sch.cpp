#include "sch.h"


// scheduler
//   target: produce op order and mem allocation minimizing execution time
//   limits: memory
//
// layout options:
//   assign blocks to device/host
//   checkpointing - keep intermediate results (like attention lookup or wide/gate) or recompute
//   tensor parallel (each layer separately?)
//   place 2+ model copies
//
// questions:
//   just use greedy scheduler (execute first possible), no smart reorders?
//
// plan
//   no mem, single device, learn to schedule pcie read/write and sort op optimally (int8 ffn backprop)
//   backprop graphs
//   memory allocation, checkpointing options
//   cuda/host gradients
//     add nvl resource
//     add ib resource
//   cross device/host layouts
//     copy across devices/hosts occupies only one link of communication resource - can be overlapped with other copies
//   launch multiple batches concurrently
//   tensor parallel support
// 

enum EResourceType
{
    RT_NVL,
    RT_IB,
    // RT_PCIE_READ,
    // RT_PCIE_WRITE,
    RT_MEM,
    RT_BG,
    RT_COUNT,
};
const char *RTName[] = { "NVL", "IB", "MEM", "BG" };


enum EGenMask {
    GM_FWD = 1,
    GM_BWD = 2,
};


struct TExecPlace
{
    yint HostId = 0;
    yint DeviceId = 0;
};

struct TExecAttrs
{
    TExecPlace Place;
    yint GroupId = 0;
    yint GroupType = 0;
};


struct TOperation
{
    int Id = 0;
    TExecAttrs Attrs;
    EResourceType RT = RT_MEM;
    TString Name;
    float Duration = 0;
    TVector<int> DepArr;

    TOperation *Dep(TOperation *p)
    {
        DepArr.push_back(p->Id);
        return this;
    }
};


struct TBatch
{
    yint OpCount = 0;
    TVector<TOperation> OpArr;

    TBatch() { OpArr.reserve(10000); }
    TOperation *AddOp(const TExecAttrs &attrs, EResourceType rt, float dur, const TString &name)
    {
        OpArr.resize(OpCount + 1);
        TOperation &op = OpArr[OpCount];
        op.Attrs = attrs;
        op.Id = OpCount;
        op.RT = rt;
        op.Name = name;
        op.Duration = dur;
        ++OpCount;
        return &op;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const float MAX_TIME = 1e38f;
struct TBatchTrack
{
    TBatch Batch;
    TVector<float> CompleteTime;
    TVector<yint> RTptr;

    float GetMinStartTime(yint opId)
    {
        if (opId >= YSize(CompleteTime)) {
            return MAX_TIME;
        }
        float rv = 0;
        for (int dp : Batch.OpArr[opId].DepArr) {
            rv = Max(rv, CompleteTime[dp]);
        }
        return rv;
    }

    yint GetNextOp(yint rt, yint from)
    {
        for (; from < YSize(Batch.OpArr); ++from) {
            if (Batch.OpArr[from].RT == rt) {
                return from;
            }
        }
        return from;
    }

    void Init(const TBatch &bb)
    {
        Batch = bb;
        yint opCount = bb.OpCount;
        CompleteTime.resize(opCount, MAX_TIME);
        RTptr.resize(RT_COUNT);
        for (yint rt = 0; rt < RT_COUNT; ++rt) {
            RTptr[rt] = GetNextOp(rt, 0);
        }
    }

    void StartOperation(yint opId, float startTime)
    {
        TOperation &op = Batch.OpArr[opId];
        yint rt = op.RT;
        RTptr[rt] = GetNextOp(rt, opId + 1);
        CompleteTime[opId] = startTime + op.Duration;
    }
};


inline yint GetGroupHash(yint groupType, yint groupId) {
    return groupType * 0x10000 + groupId;
}


struct TDeviceTracker
{
    struct TMem
    {
        yint BatchId = 0;
        yint GroupType = 0;
        yint GroupId = 0;
        yint OpCount = 0;
        float Finish = 0;
    };
    TVector<TMem> Allocated;
    float Finish[RT_COUNT] = {};

    float GetMemStart(yint maxBatchesPerGroupType, yint b, yint groupType, yint groupId) const
    {
        if (groupType < 0) {
            return 0;
        }
        yint count = 0;
        float minTime = MAX_TIME;
        for (const TMem &mem : Allocated) {
            if (mem.GroupType == groupType) {
                if (mem.BatchId == b && mem.GroupId == groupId) {
                    return 0;
                }
                count += 1;
                if (mem.OpCount == 0) {
                    minTime = Min(minTime, mem.Finish);
                }
            }
        }
        Y_ASSERT(count <= maxBatchesPerGroupType);
        if (count < maxBatchesPerGroupType) {
            return 0;
        }
        return minTime;
    }

    void StartOperation(yint b, yint groupType, yint groupId, yint rt, THashMap<yint,yint> &groupOpCount, float startTime, float duration) 
    {
        Finish[rt] = startTime + duration;
        if (groupType < 0) {
            return;
        }
        for (;;) {
            for (yint k = 0; k < YSize(Allocated); ++k) {
                TMem &mem = Allocated[k];
                if (mem.BatchId == b && mem.GroupType == groupType && mem.GroupId == groupId) {
                    --mem.OpCount;
                    mem.Finish = Max(mem.Finish, startTime + duration);
                    return;
                }
            }
            // free mem
            for (yint k = 0; k < YSize(Allocated); ++k) {
                TMem &mem = Allocated[k];
                if (mem.OpCount == 0 && startTime >= mem.Finish) {
                    Allocated.erase(Allocated.begin() + k);
                    --k;
                }
            }
            TMem mem;
            mem.BatchId = b;
            mem.GroupType = groupType;
            mem.GroupId = groupId;
            mem.OpCount = groupOpCount[GetGroupHash(groupType, groupId)];
            mem.Finish = 0;
            Allocated.push_back(mem);
        }
    }
};


static void Schedule(
    yint maxBatchesPerGroupType, yint gmMask, const TVector<TBatch> &bbArr, float paramCount, yint len, yint hostCount, yint devicesPerHost, yint tp)
{
    THashMap<yint, yint> groupOpCount;
    for (const TOperation &op : bbArr[0].OpArr) {
        groupOpCount[GetGroupHash(op.Attrs.GroupType, op.Attrs.GroupId)] += 1;
    }

    TVector<TVector<TDeviceTracker>> hostArr;
    hostArr.resize(hostCount);
    for (yint hostId = 0; hostId < hostCount; ++hostId) {
        hostArr[hostId].resize(devicesPerHost);
    }
    yint batchCount = YSize(bbArr);
   
    TVector<TBatchTrack> btArr;
    btArr.resize(batchCount);
    for (yint b = 0; b < batchCount; ++b) {
        btArr[b].Init(bbArr[b]);
    }
   
    for (;;) {
        float minStartTime = MAX_TIME;
        float minB = 0;
        yint minOpId = 0;
        for (yint b = 0; b < batchCount; ++b) {
            TBatchTrack &bt = btArr[b];
            for (yint rt = 0; rt < RT_COUNT; ++rt) {
                yint opId = bt.RTptr[rt];
                float depStart = bt.GetMinStartTime(opId);
                if (depStart == MAX_TIME) {
                    continue;
                }
                TOperation &op = bt.Batch.OpArr[opId];
                TExecPlace &place = op.Attrs.Place;
                TDeviceTracker &dev = hostArr[place.HostId][place.DeviceId];
                float devStart = dev.Finish[rt];
                float memStart = dev.GetMemStart(maxBatchesPerGroupType, b, op.Attrs.GroupType, op.Attrs.GroupId);
                float start = Max(depStart, Max(devStart, memStart));
                if (start < minStartTime) {
                    minStartTime = start;
                    minB = b;
                    minOpId = opId;
                }
            }
        }
        if (minStartTime == MAX_TIME) {
            // complete
            break;
        }
        {
            const TOperation &op = btArr[minB].Batch.OpArr[minOpId];
            const TExecPlace &place = op.Attrs.Place;
            btArr[minB].StartOperation(minOpId, minStartTime);
            hostArr[place.HostId][place.DeviceId].StartOperation(minB, op.Attrs.GroupType, op.Attrs.GroupId, op.RT, groupOpCount, minStartTime, op.Duration);
            DebugPrintf("t = %g, batch %g, op %g, run %s at %g:%g %s\n", minStartTime, minB * 1., minOpId * 1., op.Name.c_str(), place.HostId * 1., place.DeviceId * 1., RTName[op.RT]);
        }
    }
    float completeTime = 0;
    for (yint hostId = 0; hostId < hostCount; ++hostId) {
        for (yint deviceId = 0; deviceId < devicesPerHost; ++deviceId) {
            for (yint rt = 0; rt < RT_COUNT; ++rt) {
                completeTime = Max(completeTime, hostArr[hostId][deviceId].Finish[rt]);
            }
        }
    }
    float usefulOps = 0;
    if (gmMask & GM_FWD) {
        usefulOps += 2;
    }
    if (gmMask & GM_BWD) {
        usefulOps += 4;
    }
    float peak = 1900 / usefulOps;
    yint gpuCount = hostCount * devicesPerHost * tp;
    float tpps = YSize(bbArr) * len * paramCount / completeTime / 1e9 / gpuCount;
    DebugPrintf("\n%g ms, %g peak, %g Tparams/sec, %g%% MFU\n\n", completeTime, peak, tpps, tpps / peak * 100);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDims
{
    yint Dim = 1024;
    yint Xp = 512;
    yint XpCount = 24;
    yint XpSel = 2;
    yint HeadCount = 8;
    yint Ffn = 8;
    yint Depth = 42;

    //yint GetParamCount() const { return Depth * (128 * HeadCount * Dim * 5 + Dim * 128 * Ffn * 3); }
    yint GetParamCount() const { return Depth * (128 * HeadCount * Dim * 5 + Dim * Xp * XpCount * 3); }
    yint GetActiveParamCount() const { return Depth * (128 * HeadCount * Dim * 5 + Dim * Xp * XpSel * 3); }
};


static void AddPlace(yint k, yint devicesPerHost, TVector<TExecPlace> *p)
{
    TExecPlace lp;
    lp.DeviceId = k % devicesPerHost;
    lp.HostId = k / devicesPerHost;
    p->push_back(lp);
}

static void PlaceLayers(const TModelDims &dims, yint split, yint devicesPerHost, TVector<TExecPlace> *p0, TVector<TExecPlace> *p1)
{
    yint layersPerGpu = DivCeil(dims.Depth, split);
    for (yint d = 0; d < dims.Depth; ++d) {
        AddPlace(d / layersPerGpu, devicesPerHost, p0);
        AddPlace((dims.Depth - d - 1) / layersPerGpu, devicesPerHost, p1);
    }
}


static float GetMatmulDuration16(yint mnk, yint k)
{
    float rate = 780e12 * (k / (k + 450.));
    return 2 * mnk / rate * 1e3f;
}


static float GetMatmulDuration8(yint mnk, yint k)
{
    float rate = 1600e12 * (k / (k + 1000.));
    return 2 * mnk / rate * 1e3f;
}


static float GetMemDuration(yint sz)
{
    float rate = 3300e9;
    return sz / rate * 1e3f;
}


static float GetAttDuration(yint len, yint att)
{
    float rate = 250e12;
    constexpr float FRAGMENT_LEN = 1024;
    return 4 * len * att * FRAGMENT_LEN / rate * 1e3f; // QK and weighted sum of V
}


static float GetNVLDuration(yint sz)
{
    float rate = 370e9;
    return sz / rate * 1e3f;
}


static float GetNVLDuration(yint sz, yint tp)
{
    if (tp == 1) {
        return 0;
    }
    return GetNVLDuration(sz) * (tp - 1) / tp;
}


static float GetIBDuration(yint sz)
{
    float rate = 45e9;
    return sz / rate * 1e3f;
}


static void GenerateBatchGraph(TBatch *b, yint gmMask, const TModelDims &dims, const TVector<TExecPlace> &placeArr, yint len, yint tp)
{
    yint dim = dims.Dim;
    yint ffn = dims.Ffn * 128;
    yint att = dims.HeadCount * 128;
    yint xp = dims.Xp;
    yint xpSel = dims.XpSel;
    yint depth = dims.Depth;
    yint xps = xp * xpSel;
    Y_VERIFY(depth == YSize(placeArr));
    TExecAttrs attrs;
    attrs.Place = placeArr[0];
    attrs.GroupId = 0;
    attrs.GroupType = -1;
    TOperation *state = b->AddOp(attrs, RT_MEM, GetMemDuration(len * dim * 4), "embed");
    //
    constexpr int FWD_GROUP_TYPE = 0;
    // constexpr int BWD_GROUP_TYPE = 2;
    constexpr int BWD_GROUP_TYPE = 0;
    // forward
    TVector<TOperation *> stateAtt;
    TVector<TOperation *> stateFfn;
    TVector<TOperation *> stateMoe;
    TExecAttrs prevAttrs;
    for (yint d = 0; d < depth; ++d) {
        if (gmMask & GM_FWD) {
            attrs.GroupType = -1;
            attrs.Place = placeArr[d];
            if (d > 0) {
                const TExecPlace &place = attrs.Place;
                const TExecPlace &prevPlace = prevAttrs.Place;
                if (place.HostId != prevPlace.HostId) {
                    state = b->AddOp(attrs, RT_IB, GetIBDuration(len * dim * 4 / tp), "fwd_ib_send")->Dep(state);
                } else if (place.DeviceId != prevPlace.DeviceId) {
                    state = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim * 4 / tp), "fwd_nvl_send")->Dep(state);
                }
            }
            prevAttrs = attrs;

            // att
            attrs.GroupType = FWD_GROUP_TYPE;
            stateAtt.push_back(state);
            if (tp > 1) {
                state = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim, tp), "state_gather")->Dep(state);
            }
            TOperation *q = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "q")->Dep(state);
            TOperation *k = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "k")->Dep(state);
            TOperation *v = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "v")->Dep(state);
            TOperation *att_gate = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "att_gate")->Dep(state);
            TOperation *attV = b->AddOp(attrs, RT_MEM, GetAttDuration(len, att / tp), "att")->Dep(q)->Dep(k)->Dep(v);
            TOperation *attLolu = b->AddOp(attrs, RT_MEM, GetMemDuration(len * att * 4 / tp), "att_lolu")->Dep(attV)->Dep(att_gate);
            if (tp > 1) {
                attLolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * att, tp), "att_gather")->Dep(attLolu);
            }
            state = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, att), "att_compact")->Dep(attLolu);
            ++attrs.GroupId;

            // // ffn
            // stateFfn.push_back(state);
            // if (tp > 1) {
            //     state = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim, tp), "state_gather")->Dep(state);
            // }
            // TOperation *wide = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, dim), "wide")->Dep(state);
            // TOperation *gate = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, dim), "ffn_gate")->Dep(state);
            // TOperation *lolu = b->AddOp(attrs, RT_MEM, GetMemDuration(len * ffn * 4 / tp), "ffn_lolu")->Dep(wide)->Dep(gate);
            // if (tp > 1) {
            //     lolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * ffn, tp), "ffn_gather")->Dep(lolu);
            // }
            // state = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, ffn), "ffn_compact")->Dep(lolu);
            //++attrs.GroupId;

            // moe
            attrs.GroupType = FWD_GROUP_TYPE;
            stateMoe.push_back(state);
            if (tp > 1) {
                state = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim, tp), "state_gather")->Dep(state);
            }
            TOperation *moeSelect = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * 128, dim), "moe_select")->Dep(state);
            TOperation *moeExpList = b->AddOp(attrs, RT_MEM, GetMemDuration(len * 1024), "moe_sampleExp")->Dep(moeSelect);
            TOperation *moeExpGroup = b->AddOp(attrs, RT_BG, GetMemDuration(len * 1024 * 4), "moe_groupExp")->Dep(moeExpList);
            TOperation *moeGroupSamples = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps / tp), "moe_groupSamples")->Dep(moeExpList);
            TOperation *wide = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, dim), "moe_wide")->Dep(moeGroupSamples);
            TOperation *gate = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, dim), "moe_gate")->Dep(moeGroupSamples);
            TOperation *lolu = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps * 4 / tp), "moe_lolu")->Dep(wide)->Dep(gate);
            if (tp > 1) {
                lolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * xps, tp), "moe_gather")->Dep(lolu);
            }
            state = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, xp), "moe_compact")->Dep(lolu);
            state = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps * 2 / tp), "moe_add")->Dep(state);
            ++attrs.GroupId;
        } else {
            stateAtt.push_back(b->AddOp(attrs, RT_MEM, 0, "nop"));
            stateMoe.push_back(b->AddOp(attrs, RT_MEM, 0, "nop"));
        }
    }

    // backward
    TOperation *grad = state;
    for (yint d = depth - 1; d >= 0; --d) {
        if (gmMask & GM_BWD) {
            attrs.GroupType = -1;
            attrs.Place = placeArr[d];
            {
                const TExecPlace &place = attrs.Place;
                const TExecPlace &prevPlace = prevAttrs.Place;
                if (place.HostId != prevPlace.HostId) {
                    grad = b->AddOp(attrs, RT_IB, GetIBDuration(len * dim * 4 / tp), "bwd_ib_send")->Dep(grad);
                } else if (place.DeviceId != prevPlace.DeviceId) {
                    grad = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim * 4 / tp), "bwd_nvl_send")->Dep(grad);
                }
            }
            prevAttrs = attrs;

            // att backprop
            attrs.GroupType = BWD_GROUP_TYPE;
            if (tp > 1) {
                state = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim, tp), "state_gather")->Dep(stateAtt[d]);
            }
            TOperation *q = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "q")->Dep(state);
            TOperation *k = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "k")->Dep(state);
            TOperation *v = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "v")->Dep(state);
            TOperation *att_gate = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "att_gate")->Dep(state);
            TOperation *attV = b->AddOp(attrs, RT_MEM, GetAttDuration(len, att / tp), "att")->Dep(q)->Dep(k)->Dep(v);
            TOperation *attLolu = b->AddOp(attrs, RT_MEM, GetMemDuration(len * att * 4 / tp), "att_lolu")->Dep(attV)->Dep(att_gate);
            if (tp > 1) {
                attLolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * att, tp), "att_gather")->Dep(attLolu);
            }
            TOperation *dLolu = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, dim), "att_dLolu")->Dep(grad);
            TOperation *gradT = b->AddOp(attrs, RT_MEM, GetMemDuration(len * att / tp), "att_gradT")->Dep(grad);
            if (tp > 1) {
                dLolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * att, tp), "att_gather_dLolu")->Dep(dLolu)->Dep(attLolu);
            }
            TOperation *dAtt = b->AddOp(attrs, RT_MEM, GetMemDuration(len * att * 4 / tp), "att_lolu")->Dep(dLolu)->Dep(attV)->Dep(attLolu);
            TOperation *dAttGradQ = b->AddOp(attrs, RT_MEM, GetAttDuration(len, att / tp), "att_gradQ")->Dep(q)->Dep(k)->Dep(v)->Dep(dAtt);
            TOperation *dAttGradKV = b->AddOp(attrs, RT_MEM, GetAttDuration(len, att / tp) * 1.5, "att_gradKV")->Dep(q)->Dep(k)->Dep(v)->Dep(dAtt);
            grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, att / tp), "dq")->Dep(dAttGradQ)->Dep(grad);
            grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, att / tp), "dk")->Dep(dAttGradKV)->Dep(grad);
            grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, att / tp), "dv")->Dep(dAttGradKV)->Dep(grad);
            grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, att / tp), "datt_gate")->Dep(dAtt)->Dep(grad);
            if (tp > 1) {
                grad = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * att * 4, tp), "att_reduce_dNormState")->Dep(grad);
                grad = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * att * 4, tp), "reduce_grad")->Dep(grad);
            }

            // low priority
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, len), "att_delta_compact")->Dep(gradT)->Dep(attLolu);
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, len), "att_delta_q")->Dep(state)->Dep(dAttGradQ);
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, len), "att_delta_q")->Dep(state)->Dep(dAttGradKV);
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, len), "att_delta_q")->Dep(state)->Dep(dAttGradKV);
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * att / tp, len), "att_delta_q")->Dep(state)->Dep(dAtt);
            ++attrs.GroupId;


            // // ffn backprop
            // if (tp > 1) {
            //     state = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim, tp), "state_gather")->Dep(stateFfn[d]);
            // }
            // TOperation *wide = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, dim), "wide")->Dep(state);
            // TOperation *gate = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, dim), "ffn_gate")->Dep(state);
            // TOperation *lolu = b->AddOp(attrs, RT_MEM, GetMemDuration(len * ffn * 4 / tp), "ffn_lolu")->Dep(wide)->Dep(gate);
            // if (tp > 1) {
            //     lolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * ffn, tp), "ffn_gather")->Dep(lolu);
            // }
            // dLolu = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, dim), "ffn_dLolu")->Dep(grad)->Dep(lolu);
            // TOperation *dWG = b->AddOp(attrs, RT_MEM, GetMemDuration(len * ffn * 4 / tp), "ffn_lolu")->Dep(wide)->Dep(gate)->Dep(dLolu);
            // gradT = b->AddOp(attrs, RT_MEM, GetMemDuration(len * ffn / tp), "ffn_gradT")->Dep(grad);
            // grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, ffn / tp), "dwide")->Dep(dWG)->Dep(grad);
            // grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, ffn / tp), "dgate")->Dep(dWG)->Dep(grad);
            // if (tp > 1) {
            //     grad = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * ffn * 4, tp), "ffn_reduce_dNormState")->Dep(grad);
            //     grad = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * ffn * 4, tp), "reduce_grad")->Dep(grad);
            // }

            // // low priority
            // b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, len), "ffn_delta_compact")->Dep(gradT)->Dep(lolu);
            // b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, len), "ffn_delta_wide")->Dep(state)->Dep(dWG);
            // b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * ffn / tp, len), "ffn_delta_gate")->Dep(state)->Dep(dWG);
            // ++attrs.GroupId;

            // moe backprop
            attrs.GroupType = BWD_GROUP_TYPE;
            if (tp > 1) {
                state = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * dim, tp), "state_gather")->Dep(stateMoe[d]);
            }
            TOperation *moeSelect = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * 128, dim), "moe_select")->Dep(state);
            TOperation *moeExpList = b->AddOp(attrs, RT_MEM, GetMemDuration(len * 1024), "moe_sampleExp")->Dep(moeSelect);
            TOperation *moeExpGroup = b->AddOp(attrs, RT_BG, GetMemDuration(len * 1024 * 4), "moe_groupExp")->Dep(moeExpList);
            TOperation *moeGroupSamples = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps / tp), "moe_groupSamples")->Dep(moeExpList);
            TOperation *wide = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, dim), "moe_wide")->Dep(moeGroupSamples);
            TOperation *gate = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, dim), "moe_gate")->Dep(moeGroupSamples);
            TOperation *lolu = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps * 4 / tp), "moe_lolu")->Dep(wide)->Dep(gate);
            if (tp > 1) {
                lolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * xps, tp), "moe_gather")->Dep(lolu);
            }
            grad = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps / tp), "moe_group_grad")->Dep(grad);
            dLolu = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, xp), "moe_dLolu")->Dep(grad)->Dep(lolu);
            if (tp > 1) {
                dLolu = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * xps * 2, tp), "moe_dLolu_gather")->Dep(dLolu);
            }
            TOperation *dWG = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps * 4 / tp), "moe_dLolu")->Dep(wide)->Dep(gate)->Dep(dLolu);
            TOperation *dMoeSelect = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xpSel * 1024), "moe_BackpropMoe")->Dep(grad)->Dep(lolu);
            gradT = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps / tp), "ffn_gradT")->Dep(grad);
            grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, xp), "moe_dwide")->Dep(dWG)->Dep(grad);
            grad = b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, xp), "moe_dgate")->Dep(dWG)->Dep(grad);
            grad = b->AddOp(attrs, RT_MEM, GetMemDuration(len * xps * 2 / tp), "moe_combineExperts")->Dep(dWG)->Dep(grad);
            if (tp > 1) {
                grad = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * xps * 4, tp), "moe_reduce_dNormState")->Dep(grad);
                grad = b->AddOp(attrs, RT_NVL, GetNVLDuration(len * xps * 4, tp), "reduce_grad")->Dep(grad);
            }

            // low priority
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, len), "moe_delta_compact")->Dep(gradT)->Dep(lolu);
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, len), "moe_delta_wide")->Dep(state)->Dep(dWG);
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * xps / tp, len), "moe_delta_gate")->Dep(state)->Dep(dWG);
            b->AddOp(attrs, RT_MEM, GetMatmulDuration8(len * dim * 128 / tp, len), "moe_delta_moe_select")->Dep(state)->Dep(dMoeSelect);
            ++attrs.GroupId;
        }
    }
}


int main()
{
    yint tp = 1;
    yint len = 8192;
    yint split = 1;
    yint batchCount = 1;
    yint maxBatchesPerGroupType = 1;

    TModelDims dims;
    // 1.5B of 10B
    tp = 1;
    split = 4;
    batchCount = 16;
    dims.Dim = 2048;
    dims.HeadCount = 16;
    dims.Ffn = 24;
    dims.Depth = 48;
    dims.Xp = 512;
    dims.XpSel = 4;
    dims.XpCount = 64;

    // // concurrent batches research
    // tp = 2;
    // split = 2;
    // batchCount = 2;
    // dims.Dim = 4096;
    // dims.HeadCount = 32;
    // dims.Ffn = 48;
    // dims.Depth = 4;
    // dims.Xp = 512;
    // dims.XpSel = 8;
    // dims.XpCount = 128;

    // // 8B of 56B
    // tp = 2;
    // split = 16;
    // batchCount = 32;
    // dims.Dim = 4096;
    // dims.HeadCount = 32;
    // dims.Ffn = 48;
    // dims.Depth = 64;
    // dims.Xp = 512;
    // dims.XpSel = 8;
    // dims.XpCount = 128;

    // // 25B of 325B, batch 128 28% MFU (for split 12 = H200 = 38% MFU)
    // tp = 4;
    // split = 24;
    // batchCount = 64 * 2;
    // dims.Dim = 8192;
    // dims.HeadCount = 64;
    // dims.Ffn = 96;
    // dims.Depth = 48;
    // dims.Xp = 1024;
    // dims.XpSel = 8;
    // dims.XpCount = 256;

    // // 35B of 634B
    // tp = 4;
    // split = 48;
    // batchCount = 128;
    // dims.Dim = 8192;
    // dims.HeadCount = 64;
    // dims.Ffn = 96;
    // dims.Depth = 48;
    // dims.Xp = 2048;
    // dims.XpSel = 8;
    // dims.XpCount = 256;

    TVector<TExecPlace> lpArr0, lpArr1;
    yint devicesPerHost = Min<yint>(split, 8 / tp);
    PlaceLayers(dims, split, devicesPerHost, &lpArr0, &lpArr1);
    yint hostCount = 1;
    for (TExecPlace &place : lpArr0) {
        hostCount = Max(hostCount, place.HostId + 1);
    }

    //yint gmMask = GM_FWD;
    //yint gmMask = GM_BWD;
    yint gmMask = GM_FWD | GM_BWD;
    TBatch bb0;
    GenerateBatchGraph(&bb0, gmMask, dims, lpArr0, len, tp);
    TBatch bb1;
    GenerateBatchGraph(&bb1, gmMask, dims, lpArr1, len, tp);
    TVector<TBatch> bbArr;
    for (yint k = 0; k < batchCount; ++k) {
        bbArr.push_back(bb0);
    }
    // for (yint k = 0; k < batchCount / 2; ++k) {
    //     bbArr.push_back(bb0);
    //     bbArr.push_back(bb1);
    // }
    Schedule(maxBatchesPerGroupType, gmMask, bbArr, dims.GetActiveParamCount(), len, hostCount, devicesPerHost, tp);
    DebugPrintf("model size(B) %g of %g, %g per device\n\n", dims.GetActiveParamCount() / 1e9, dims.GetParamCount() / 1e9, dims.GetParamCount() / split / tp / 1e9);
    return 0;
}

