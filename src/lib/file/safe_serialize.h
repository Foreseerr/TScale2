#pragma once
#include "dir.h"


inline bool CanSafeRead(const TString &fname)
{
    TString tmpFname = fname + ".tmp";
    return DoesFileExist(fname) || DoesFileExist(tmpFname);
}

template <class T>
void SafeWrite(const TString &fname, T &data)
{
    TString tmpFname = fname + ".tmp";
    Serialize(IO_WRITE, tmpFname, data);
    EraseFile(fname);
    RenameFile(tmpFname, fname);
}

template <class T>
bool SafeRead(const TString &fname, T &data)
{
    TString tmpFname = fname + ".tmp";
    if (DoesFileExist(fname)) {
        if (DoesFileExist(tmpFname)) {
            EraseFile(tmpFname);
        }
    } else if (DoesFileExist(tmpFname)) {
        RenameFile(tmpFname, fname);
    } else {
        data = T();
        return false;
    }
    Serialize(IO_READ, fname, data);
    return true;
}
