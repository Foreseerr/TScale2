#pragma once
#include "acpu_base.h"
#include <gpt/train_config/train_step.h>
#include <lib/cuda/cuda_arrays.h>
#include <util/thread.h>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
class TCPUMatrixAdd : public TThrRefBase
{
    enum {
        BASE_WORKER_COUNT = 4,
    };

    enum EJob {
        MJ_SUM_DEVICE_DELTA,
        MJ_ON_DELTA_HOOK,
        MJ_ADD_DELTA,
        MJ_ADD_BIT_DELTA,
        MJ_CONVERT,
        MJ_NONE,
    };

    struct TJob
    {
        int MatrixId = -1;
        EJob Op = MJ_NONE;

        TJob() {}
        TJob(int id, EJob op) : MatrixId(id), Op(op) {}
    };

    struct TCudaLaunchFlags
    {
        TCudaVector<int> CudaFlag;
        TVector<int> PrevCudaFlag;

        void Init(yint sz)
        {
            CudaFlag.AllocateHost(sz);
            CudaFlag.ClearHostMem();
            ClearPodArray(&PrevCudaFlag, sz);
        }
        bool CheckFlag(yint k)
        {
            volatile int *cudaBuf = CudaFlag.GetHostPtr();
            int newCudaFlag = cudaBuf[k];
            if (newCudaFlag != PrevCudaFlag[k]) {
                // avoid modifying cudaAddDeltaFlag from cpu & gpu concurrently
                PrevCudaFlag[k] = newCudaFlag;
                return true;
            }
            return false;
        }
    };


    struct TDeviceData : public TThrRefBase
    {
        TCudaLaunchFlags DeltaFlag;
        TCudaLaunchFlags AllowDelayedFlag;
    };


    struct TWorkerData : public TThrRefBase
    {
        TThread Thr;
    };

private:
    yint MaxDeltaMatrices = 0;
    TVector<TIntrusivePtr<TDeviceData>> DeviceArr;
    TIntrusivePtr<TMatrixOpTracker> MatrixOps;
    TVector<int> DeltaReadyDeviceCount;
    TVector<int> AllowDelayedReadyDeviceCount;
    TVector<TIntrusivePtr<TMatrixBaseOps>> MatrixArr;
    TIntrusivePtr<IMMDeltaHookGen> DeltaHookGen;
    TVector<TIntrusivePtr<IMMDeltaHook>> DeltaHookArr;
    TVector<TIntrusivePtr<TWorkerData>> WorkerArr;
    TSingleProducerJobCircleBuffer<TJob, 8192> JobQueue;
    std::atomic<yint> WorkerCount;
    std::atomic<yint> JobCount;
    std::atomic<yint> JobGeneratorFlag;
    std::atomic<bool> Exit;
    std::atomic<bool> IsIdle;
    std::atomic<bool> EnterIdle;
    yint MatrixCount = 0;
    TTrainingStep Step;
    EAddToModel AddToModel = GRADIENT_APPLY;
    TCudaVector<int> CudaIterCount;

    EOperation GetOp(yint matrixId) { return MatrixOps->GetOp(matrixId); }
    void SetOp(yint matrixId, EOperation op) { MatrixOps->SetOp(matrixId, op); }
    void NewJob(int k, EJob op);
    bool GenerateJobs();
    void Process(const TJob &job);
    bool StartDelayedOps(int newOp);
    void ResetAllowDelayedFlag();
    ~TCPUMatrixAdd();

public:
    TCPUMatrixAdd(yint deviceCount, yint maxDeltaMatrices, IMMDeltaHookGen *deltaHookGen);
    void AddMatrix(TPtrArg<TMatrixBaseOps> p);
    void LaunchWorkers();
    void StartIteration(const TTrainingStep &step, EAddToModel addToModel); // assume no pending ops at this moment
    void WaitActiveCompute();
    void WaitDelayedCompute();
    void ConvertMatrices();
    TCudaPOD<int> GetCurrentIteration() { return CudaIterCount.GetElement(0); }
    yint GetDeviceCount() const { return YSize(DeviceArr); }

public:
    void WorkerThread();
};


}