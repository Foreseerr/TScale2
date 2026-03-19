#include "thr_pool.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
void TParallelExec::RunCurrentJob()
{
    for (;;) {
        yint k = JobK.fetch_add(1);
        if (k >= Job->Count) {
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
            RunCurrentJob();
            CompleteCount.fetch_add(1);
            prevJobId = curJobId;
        } else {
            SchedYield();
        }
    }
}


TParallelExec::TParallelExec(yint threadCount) : JobId(0), JobK(0), CompleteCount(0)
{
    for (yint thrId = 0; thrId < threadCount; ++thrId) {
        ThrArr.push_back(new TThreadHolder(this));
    }
}


TParallelExec::~TParallelExec()
{
    Exit = true;
}
