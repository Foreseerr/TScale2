#pragma once

namespace NHPTimer
{
typedef i64 STime;
// get current time
void GetTime( STime *pTime );
// count time passed since *pTime, current time will be stored to *pTime
double GetTimePassed( STime *pTime );
// get CPU frequency
double GetClockRate();
};
