#include "acpu_process.h"
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
yint MatrixAddWorkerThreadCount = 0;


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
void TCPUMatrixAdd::NewJob(int k, EJob op)
{
    JobCount.fetch_add(1);
    JobQueue.Add(TJob(k, op));
}


bool TCPUMatrixAdd::GenerateJobs()
{
    bool hasFinished = true;
    yint deviceCount = YSize(DeviceArr);
    yint matrixCount = MatrixCount;
    for (yint matrixId = 0; matrixId < matrixCount; ++matrixId) {
        // perform ops until we run into waiting state
        for (bool keepProcess = true; keepProcess;) {
            int op = GetOp(matrixId);
            keepProcess = false;
            if (op == MOP_NONE) {
                // check if we got new delta
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    TDeviceData &dev = *DeviceArr[deviceId];
                    if (dev.DeltaFlag.CheckFlag(matrixId)) {
                        if (deviceCount > 1) {
                            if (++DeltaReadyDeviceCount[matrixId] < deviceCount) {
                                continue;
                            }
                            DeltaReadyDeviceCount[matrixId] = 0;
                        }
                        SetOp(matrixId, MOP_WAIT);
                        NewJob(matrixId, MJ_SUM_DEVICE_DELTA);
                        hasFinished = false;
                    }
                }
            } else if (op == MOP_APPLY_DELAYED_DELTA_WAIT) {
                // wait signal that matrix contents were used and delta can be applied, we are not finished
                hasFinished = false;
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    TDeviceData &dev = *DeviceArr[deviceId];
                    if (dev.AllowDelayedFlag.CheckFlag(matrixId)) {
                        if (deviceCount > 1) {
                            if (++AllowDelayedReadyDeviceCount[matrixId] < deviceCount) {
                                continue;
                            }
                            AllowDelayedReadyDeviceCount[matrixId] = 0;
                        }
                        SetOp(matrixId, MOP_NEW_DELTA);
                        keepProcess = true;
                    }
                }
            } else if (op == MOP_NEW_DELTA) {
                // start delta processing, continue processing
                if (DeltaHookArr[matrixId].Get()) {
                    SetOp(matrixId, MOP_WAIT);
                    NewJob(matrixId, MJ_ON_DELTA_HOOK);
                    hasFinished = false;
                } else {
                    SetOp(matrixId, MOP_ADD_DELTA);
                    keepProcess = true;
                }
            } else if (op == MOP_WAIT) {
                // we are waiting for network or something, iteration is incomplete
                hasFinished = false;
            } else if (op == MOP_ADD_DELTA) {
                SetOp(matrixId, MOP_WAIT);
                NewJob(matrixId, MJ_ADD_DELTA);
                hasFinished = false;
            } else if (op == MOP_ADD_BIT_DELTA) {
                SetOp(matrixId, MOP_WAIT);
                NewJob(matrixId, MJ_ADD_BIT_DELTA);
                hasFinished = false;
            } else if (op == MOP_DELAY_DELTA) {
                // do nothing, we have delta which will be applied on next iteration
            } else if (op == MOP_CONVERT) {
                SetOp(matrixId, MOP_WAIT);
                NewJob(matrixId, MJ_CONVERT);
                hasFinished = false;
            } else {
                Y_VERIFY(0 && "unknown add matrix state");
            }
        }
    }
    return hasFinished;
}


void TCPUMatrixAdd::Process(const TJob &job)
{
    yint deviceCount = YSize(DeviceArr);
    yint matrixId = job.MatrixId;
    TIntrusivePtr<TMatrixBaseOps> matrix = MatrixArr[matrixId];
    switch (job.Op) {

    case MJ_SUM_DEVICE_DELTA:
        for (yint srcDeviceId = 0; srcDeviceId < deviceCount; ++srcDeviceId) {
            matrix->AddDeviceToSumDelta(srcDeviceId);
        }
        if (AddToModel == GRADIENT_ACCUMULATE) {
            Y_ASSERT(GetOp(matrixId) == MOP_WAIT);
            SetOp(matrixId, MOP_NONE);
            break;
        }
        // got new delta
        if (matrix->IsDelayGradient()) {
            SetOp(matrixId, MOP_DELAY_DELTA);
        } else {
            if (DeltaHookArr[job.MatrixId].Get()) {
                DeltaHookArr[job.MatrixId]->OnDelta();
            } else {
                Y_ASSERT(GetOp(matrixId) == MOP_WAIT);
                matrix->AddDelta(Step);
                SetOp(matrixId, MOP_NONE);
            }
        }
        break;

    case MJ_ON_DELTA_HOOK:
        DeltaHookArr[job.MatrixId]->OnDelta();
        break;

    case MJ_ADD_DELTA:
        Y_ASSERT(GetOp(matrixId) == MOP_WAIT);
        matrix->AddDelta(Step);
        SetOp(matrixId, MOP_NONE);
        break;

    case MJ_ADD_BIT_DELTA:
        Y_ASSERT(GetOp(matrixId) == MOP_WAIT);
        if (matrix->AddBitDelta(Step)) {
            matrix->Convert();
        }
        SetOp(matrixId, MOP_NONE);
        break;

    case MJ_CONVERT:
        Y_ASSERT(GetOp(matrixId) == MOP_WAIT);
        matrix->Convert();
        SetOp(matrixId, MOP_NONE);
        break;

    default:
        Y_VERIFY(0);
    }
    JobCount.fetch_add(-1);
}


void TCPUMatrixAdd::WorkerThread()
{
    yint workerId = WorkerCount.fetch_add(1);
    TWorkerData *data = WorkerArr[workerId].Get();
    while (!Exit) {
        TJob job;
        if (!IsIdle) {
            yint freeJobGenerator = 0;
            if (JobGeneratorFlag.compare_exchange_strong(freeJobGenerator, 1)) {
                // acquired job generator role
                for (;;) {
                    bool wasEnterIdle = EnterIdle;
                    int oldJobCount = JobCount.load();
                    bool hasFinished = GenerateJobs();
                    if (JobQueue.Get(&job)) {
                        // release job generator role, perform work
                        JobGeneratorFlag = 0;
                        Process(job);
                        break;
                    } else if (hasFinished && wasEnterIdle && oldJobCount == 0) {
                        EnterIdle = false;
                        IsIdle = true;
                        JobGeneratorFlag = 0;
                        break;
                    } else if (Exit || IsIdle) {
                        JobGeneratorFlag = 0;
                        break;
                    }
                }
            } else if (JobQueue.Get(&job)) {
                // has job to do
                Process(job);
            } else {
                _mm_pause();
            }
        } else {
            _mm_pause();
        }
    }
}


void TCPUMatrixAdd::ResetAllowDelayedFlag()
{
    for (auto &dev : DeviceArr) {
        for (yint k = 0; k < MatrixCount; ++k) {
            dev->AllowDelayedFlag.CheckFlag(k);
        }
    }
}


TCPUMatrixAdd::~TCPUMatrixAdd()
{
    Exit = true;
}


TCPUMatrixAdd::TCPUMatrixAdd(yint deviceCount, yint maxDeltaMatrices, IMMDeltaHookGen *deltaHookGen)
    : DeltaHookGen(deltaHookGen), WorkerCount(0), JobCount(0), JobGeneratorFlag(0), Exit(false), IsIdle(true), EnterIdle(false)
{
    MaxDeltaMatrices = maxDeltaMatrices;
    MatrixArr.resize(maxDeltaMatrices);
    MatrixOps = new TMatrixOpTracker();
    DeltaHookArr.resize(maxDeltaMatrices);
    DeviceArr.resize(deviceCount);
    for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
        DeviceArr[deviceId] = new TDeviceData;
        TDeviceData &dev = *DeviceArr[deviceId];
        dev.DeltaFlag.Init(maxDeltaMatrices);
        dev.AllowDelayedFlag.Init(maxDeltaMatrices);
    }
    ClearPodArray(&DeltaReadyDeviceCount, maxDeltaMatrices);
    ClearPodArray(&AllowDelayedReadyDeviceCount, maxDeltaMatrices);
    yint workerCount = BASE_WORKER_COUNT;
    if (MatrixAddWorkerThreadCount > 0) {
        workerCount = MatrixAddWorkerThreadCount;
    }
    WorkerArr.resize(workerCount);
    for (yint workerId = 0; workerId < workerCount; ++workerId) {
        WorkerArr[workerId] = new TWorkerData();
    }
    CudaIterCount.AllocateHost(1);
    CudaIterCount.ClearHostMem();
}


void TCPUMatrixAdd::AddMatrix(TPtrArg<TMatrixBaseOps> p)
{
    yint matrixId = MatrixCount++;
    Y_VERIFY(matrixId < YSize(MatrixArr));
    yint matrixOpCount = MatrixOps->AddMatrix();
    Y_VERIFY(matrixOpCount == MatrixCount);
    MatrixArr[matrixId] = p.Get();
    TVector<TCudaPOD<int>> cudaDeltaFlags;
    TVector<TCudaPOD<int>> cudaAllowDelayedFlags;
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        cudaDeltaFlags.push_back(DeviceArr[deviceId]->DeltaFlag.CudaFlag.GetElement(matrixId));
        cudaAllowDelayedFlags.push_back(DeviceArr[deviceId]->AllowDelayedFlag.CudaFlag.GetElement(matrixId));
    }
    p->AttachOp(GetCurrentIteration(), cudaDeltaFlags, cudaAllowDelayedFlags);
    if (DeltaHookGen.Get()) {
        DeltaHookArr[matrixId] = DeltaHookGen->CreateDeltaHook(matrixId, p, MatrixOps);
    }
}


void TCPUMatrixAdd::LaunchWorkers()
{
    // launch workers
    for (TIntrusivePtr<TWorkerData> &w : WorkerArr) {
        w->Thr.Create(this);
    }
}


void TCPUMatrixAdd::StartIteration(const TTrainingStep &step, EAddToModel addToModel)
{
    Y_VERIFY(IsIdle);
    Y_ASSERT(JobCount.load() == 0);
    CudaIterCount.GetHostPtr()[0] += 1;
    Step = step;
    AddToModel = addToModel;
    if (DeltaHookGen.Get()) {
        DeltaHookGen->OnIterationStart();
    }
    ResetAllowDelayedFlag();
    MatrixOps->StartDelayedDelta(MOP_APPLY_DELAYED_DELTA_WAIT);
    IsIdle = false;
}


void TCPUMatrixAdd::WaitActiveCompute()
{
    if (IsIdle) {
        return;
    }
    EnterIdle = true;
    // wait entering Idle state
    while (!IsIdle) {
        _mm_pause();
    }
    // wait pending jobs completion
    for (;;) {
        if (JobCount.load() == 0) {
            break;
        }
        _mm_pause();
    }
}


void TCPUMatrixAdd::WaitDelayedCompute()
{
    WaitActiveCompute();
    if (MatrixOps->StartDelayedDelta(MOP_NEW_DELTA)) {
        IsIdle = false;
        WaitActiveCompute();
    }
}


void TCPUMatrixAdd::ConvertMatrices()
{
    Y_VERIFY(IsIdle);
    MatrixOps->ConvertMatrices();
    IsIdle = false;
    WaitActiveCompute();
}
}