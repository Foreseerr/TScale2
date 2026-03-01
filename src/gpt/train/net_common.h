#pragma once
#include <gpt/data/data.h>
#include <gpt/train_ctx/backprop.h>
#include <gpt/train_ctx/train_ctx.h>
#include <lib/net/p2p.h>


enum ECommandResult {
    CMD_OK,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TBase>
class TCalcModelError : public TBase
{
    TVector<TFragment> FragArr;
    SAVELOAD_OVERRIDE(FragArr);

public:
    TCalcModelError() {}
    TCalcModelError(const TVector<TFragment> &fragArr) : FragArr(fragArr) {}
    void Exec(typename TBase::TExecCtx *p) override
    {
        double res = CalcModelErr(FragArr, p->Model->GetInstance(0));
        p->Master.Send(res);
    }
};

template <class TBase>
double DistributedCalcModelErr(NNet::TMasterNetTempl<TBase> &masterNet, const TVector<TVector<TFragment>> &batches)
{
    if (batches.empty()) {
        return 0;
    }
    TVector<TIntrusivePtr<TBase>> cmdArr;
    for (const TVector<TFragment> &b : batches) {
        cmdArr.push_back(new TCalcModelError<TBase>(b));
    }
    TVector<double> resArr;
    masterNet.DistributedExec(cmdArr, &resArr);
    double sum = 0;
    for (double x : resArr) {
        sum += x;
    }
    return sum / YSize(resArr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// model fetch
template <class TBase>
class TWaitDelayedUpdates : public TBase
{
public:
    void Exec(typename TBase::TExecCtx *p) override
    {
        p->Model->WaitAllCompute();
        p->Master.SendCopy(CMD_OK);
    }
};


template <class TBase>
class TMakeParamsSnapshot : public TBase
{
public:
    void Exec(typename TBase::TExecCtx *p) override
    {
        TModelParams params;
        p->Model->GetParams(&params);
        SerializeMem(IO_WRITE, &p->ModelSnapshot, params);
        yint sz = YSize(p->ModelSnapshot);
        p->Master.Send(sz);
    }
};


template <class TBase>
class TGetParamsSnapshotFragment : public TBase
{
    yint Offset = 0;
    yint Size = 0;
    SAVELOAD_OVERRIDE(Offset, Size);

public:
    TGetParamsSnapshotFragment() {}
    TGetParamsSnapshotFragment(yint offset, yint size) : Offset(offset), Size(size) {}
    void Exec(typename TBase::TExecCtx *p) override
    {
        TVector<ui8> frag;
        frag.resize(Size);
        memcpy(frag.data(), p->ModelSnapshot.data() + Offset, Size);
        p->Master.Send(frag);
    }
};


template <class TBase>
class TModelParamsFetcher
{
    enum {
        FRAG_COUNT = 1000
    };
    bool IsFetchingFlag = false;
    yint TotalSize = 0;
    yint Offset = 0;
    yint FragSize = 0;
    TVector<ui8> Buf;
    TString ResFilename;

public:
    bool IsFetching() const { return IsFetchingFlag; }
    void StartFetch(yint sz, const TString &resFilename)
    {
        Y_VERIFY(!IsFetchingFlag);
        IsFetchingFlag = true;
        TotalSize = sz;
        Offset = 0;
        FragSize = sz / FRAG_COUNT + 1;
        Buf.resize(sz);
        ResFilename = resFilename;
    }
    TGetParamsSnapshotFragment<TBase> *MakeDownloadCommand()
    {
        Y_VERIFY(IsFetchingFlag);
        yint sz = Min<yint>(FragSize, TotalSize - Offset);
        return new TGetParamsSnapshotFragment<TBase>(Offset, sz);
    }
    void GotDownloadCommandResult(const TVector<ui8> &result)
    {
        yint sz = YSize(result);
        memcpy(Buf.data() + Offset, result.data(), sz);
        Offset += sz;
        Y_VERIFY(Offset <= TotalSize);
        if (Offset == TotalSize) {
            TFileStream f(IO_WRITE, ResFilename.c_str());
            f.Write(Buf.data(), YSize(Buf));
            IsFetchingFlag = false;
        }
    }
};


template <class TBase>
void FetchModelFragment(NNet::TMasterNetTempl<TBase> &masterNet, TModelParamsFetcher<TBase> *p, TPtrArg<NNet::ITcpConnection> modelFetchConn)
{
    TModelParamsFetcher<TBase> &modelFetch = *p;
    masterNet.SendCommand(modelFetchConn, modelFetch.MakeDownloadCommand());
    TVector<ui8> result;
    WaitData(masterNet.Queue, modelFetchConn, &result);
    modelFetch.GotDownloadCommandResult(result);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TBase>
class TBackprop : public TBase
{
    yint Iter = 0;
    yint MaxIters = 0;
    TDescentConfig DescentConfig;
    THostBatchConfig HostBatchConfig;
    TVector<TFragment> FragArr;
    SAVELOAD_OVERRIDE(Iter, MaxIters, DescentConfig, HostBatchConfig, FragArr);

public:
    TBackprop() {}
    TBackprop(yint iter, yint maxIters, const TDescentConfig &dc, const THostBatchConfig &bc, const TVector<TFragment> &fragArr)
        : Iter(iter), MaxIters(maxIters), DescentConfig(dc), HostBatchConfig(bc), FragArr(fragArr)
    {
    }
    void Exec(typename TBase::TExecCtx *p) override
    {
        TXRng iterRng(Iter);
        TTrainingStep step = DescentConfig.GetStep(Iter, MaxIters);
        BackpropBatch(iterRng, DescentConfig, HostBatchConfig, step, FragArr, p->Model);
        p->Master.SendCopy(CMD_OK);
    }
};


template <class TBase>
void NetTrainModel(NNet::TMasterNetTempl<TBase> &masterNet, const TTrainContext &trainCtx, yint startIteration)
{
    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    TModelParamsFetcher<TBase> modelFetch;
    TIntrusivePtr<NNet::ITcpConnection> modelFetchConn = masterNet.WorkerSet.begin()->first;
    yint maxIters = trainCtx.GetMaxIters();
    for (yint iter = startIteration; iter <= maxIters; ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            // wait delayed updates on all hosts (otherwise no network exchange will happen)
            TVector<ECommandResult> cmdResults;
            masterNet.BroadcastCommand(new TWaitDelayedUpdates<TBase>(), &cmdResults);
            if (trainCtx.IsSaveModel() && !modelFetch.IsFetching()) {
                // make model params snapshot on first host
                masterNet.SendCommand(modelFetchConn, new TMakeParamsSnapshot<TBase>());
                yint sz;
                WaitData(masterNet.Queue, modelFetchConn, &sz);
                modelFetch.StartFetch(sz, GetHomeDir() + Sprintf("eden_gpt_%.8gk.bin", iter / 1000.));
            }
            float trainErr = DistributedCalcModelErr(masterNet, trainCtx.GetScoreTrainBatches()) * trainCtx.GetCompression();
            float testErr = DistributedCalcModelErr(masterNet, trainCtx.GetScoreTestBatches()) * trainCtx.GetCompression();
            if (testErr != 0) {
                DebugPrintf(
                    "iter %.8gk, %g sec, train err %g, test err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr, testErr);
                fflush(0);
            } else {
                DebugPrintf("iter %.8gk, %g sec, train err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr);
                fflush(0);
            }
        }

        // fetch model snapshot one fragment per iteration
        if (modelFetch.IsFetching()) {
            FetchModelFragment(masterNet, &modelFetch, modelFetchConn);
        }

        // backprop
        for (auto it = masterNet.WorkerSet.begin(); it != masterNet.WorkerSet.end(); ++it) {
            ui64 rank = it->second;
            ui64 rngSeed = (iter + 0xbadf00d) * 0x3148efull + rank * 0x0b08d424991cf81eull;
            TVector<TFragment> fragArr;
            trainCtx.SampleTrainBatches(rngSeed, &fragArr);

            const TDescentConfig &dc = trainCtx.GetDescentConfig();
            const THostBatchConfig &bc = trainCtx.GetBatchConfig();
            masterNet.SendCommand(it->first, new TBackprop<TBase>(iter, maxIters, dc, bc, fragArr));
        }
        TVector<ECommandResult> cmdResults;
        masterNet.CollectCommandResults(&cmdResults);
    }

    DebugPrintf("Fetch last iteration model\n");
    while (modelFetch.IsFetching()) {
        FetchModelFragment(masterNet, &modelFetch, modelFetchConn);
    }
}