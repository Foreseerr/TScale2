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

    IJob *volatile Job = 0;
    std::atomic<yint> JobId;
    std::atomic<yint> JobK;
    std::atomic<yint> CompleteCount;
    TVector<TIntrusivePtr<TThreadHolder>> ThrArr;
    volatile bool Exit = false;

    void RunCurrentJob();

public:
    TParallelExec(yint threadCount);
    ~TParallelExec();
    void WorkerThread();

    template <class F>
    void Run(yint start, yint count, F f)
    {
        TIntrusivePtr<IJob> job = new TJob<F>(count, f);
        Job = job.Get();
        CompleteCount = 0;
        JobK = start;
        JobId.fetch_add(1);
        RunCurrentJob();
        while (CompleteCount < YSize(ThrArr)) {
            SchedYield();
        }
        Job = 0;
    }
    
    template <class F>
    void Run(yint count, F f)
    {
        Run(0, count, f);
    }
};
