#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
struct THostBatchConfig
{
    yint BatchSize = 0;
    yint ModelInstanceCount = 0;
    yint MicroBatchCount = 0;
    yint AccumulateSteps = 0;
    yint InstanceMicroBatchSize = 0;
    yint FragLen = 0;

public:
    THostBatchConfig() {}
    THostBatchConfig(yint modelInstanceCount, yint microBatchCount, yint limitNodeCount, yint batchSize, yint fragLen);
    yint GetInstanceMaxNodeCount() const;
};
