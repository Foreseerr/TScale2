#pragma once
#include "thread.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
class TParallelExec : public TNonCopyable
{
    struct IJob : public TThrRefBase
    {
        yint Count = 0;

        IJob(yint count) : Count(count) {}
        virtual void Run(yint k) = 0;
    };

    template <class TFunc>
    struct TJob : public IJob
    {
        TFunc F;

        TJob(yint count, TFunc f) : IJob(count), F(f) {}
        void Run(yint k) override { F(k); }
    };

    TIntrusivePtr<IJob> Job;
    std::atomic<yint> JobId;
    std::atomic<yint> JobK;
    std::atomic<yint> CompleteCount;
    TVector<TIntrusivePtr<TThreadHolder>> ThrArr;
    volatile bool Exit = false;

    void RunJob(TPtrArg<IJob> job);

public:
    TParallelExec(yint threadCount);
    ~TParallelExec();
    void WorkerThread();

    template <class F>
    void Run(yint count, F f)
    {
        Job = new TJob<F>(count, f);
        CompleteCount = 0;
        JobK = 0;
        JobId.fetch_add(1);
        RunJob(Job);
        while (CompleteCount < YSize(ThrArr)) {
            SchedYield();
        }
        Job = 0;
    }
};
