#include "thr_pool.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
void TParallelExec::RunJob(TPtrArg<IJob> job)
{
    for (;;) {
        yint k = JobK.fetch_add(1);
        if (k >= job->Count) {
            return;
        }
        Job->Run(k);
    }
}


void TParallelExec::WorkerThread()
{
    yint prevJobId = 0;
    while (!Exit) {
        yint curJobId = JobId;
        if (curJobId != prevJobId) {
            RunJob(Job);
            CompleteCount.fetch_add(1);
            prevJobId = curJobId;
        } else {
            SchedYield();
        }
    }
}


TParallelExec::TParallelExec(yint threadCount)
{
    for (yint thrId = 0; thrId < threadCount; ++thrId) {
        ThrArr.push_back(new TThreadHolder(this));
    }
}


TParallelExec::~TParallelExec()
{
    Exit = true;
}
