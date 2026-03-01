#include "time.h"


time_t GetTimeUTC(int year, int month, int day)
{
    struct tm tmval;
    Zero(tmval);
    tmval.tm_year = year - 1900;
    tmval.tm_mon = month - 1;
    tmval.tm_mday = day;
#ifdef _win_
    return _mkgmtime(&tmval);
#else
    return timegm(&tmval);
#endif
}
