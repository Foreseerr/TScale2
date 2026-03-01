#include "fast_io.h"


TBufferedStream::~TBufferedStream()
{
    if (MemStream) {
        MemStream->Swap(&Buf);
        MemStream->Seek(Pos);
        if (!IsReadingFlag) {
            MemStream->Truncate();
        }
    } else {
        if (IsReadingFlag) {
            // reading position in Stream is corrupted due to prefetch
        } else {
            Stream->Write(Buf.data(), Pos);
            Y_VERIFY(!Stream->IsFailed());
        }
    }
}


void TBufferedStream::ReadLarge(void *userBuffer, yint size)
{
    if (IsEof) {
        memset(userBuffer, 0, size);
        return;
    }
    ui8 *dst = (ui8*)userBuffer;
    yint leftBytes = BufSize - Pos;
    memcpy(dst, Buf.data() + Pos, leftBytes);
    dst += leftBytes;
    size -= leftBytes;
    // fill buffer (or fulfil request)
    if (MemStream) {
        IsEof = true;
    } else {
        Pos = 0;
        BufSize = 0;
        if (size > PREFETCH_SIZE) {
            yint n = Stream->Read(dst, size);
            if (n == size) {
                return;
            }
            dst += n;
            size -= n;
            IsEof = true;
        } else {
            BufSize = Stream->Read(Buf.data(), PREFETCH_SIZE);
            if (BufSize == 0) {
                IsEof = true;
            }
        }
    }
    Read(dst, size);
}


void TBufferedStream::WriteLarge(const void *userBuffer, yint size)
{
    if (MemStream) {
        Buf.yresize(BufSize + size + PREFETCH_SIZE);
        BufSize = YSize(Buf);
    } else {
        Stream->Write(Buf.data(), Pos);
        Pos = 0;
        if (size >= PREFETCH_SIZE) {
            Stream->Write(userBuffer, size);
            return;
        }
    }
    Write(userBuffer, size);
}


void TBufferedStream::Flush()
{
    Y_VERIFY(!IsReadingFlag);
    if (!MemStream && Pos > 0) {
        Stream->Write(Buf.data(), Pos);
        Pos = 0;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
TOFStream &TOFStream::operator<<(yint k)
{
    char buf[128];
    char *ptr = buf + ARRAY_SIZE(buf) - 1;
    *ptr-- = 0;
    if (k == 0) {
        *ptr-- = '0';
    } else {
        // value
        yint val = abs(k);
        while (val > 0) {
            yint x = val / 10;
            *ptr-- = (val - x * 10) + '0';
            val = x;
        }
        // sign
        if (k < 0) {
            *ptr-- = '-';
        }
    }
    return (*this) << ptr + 1;
}


TOFStream &TOFStream::operator<<(double f)
{
    char buf[128];
    snprintf(buf, ARRAY_SIZE(buf), "%g", f);
    return (*this) << buf;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
bool ReadWholeFile(const TString &szFileName, TVector<char> *res)
{
    res->resize(0);
    TFileStream fs(IO_READ, szFileName);
    if (fs.IsValid()) {
        yint sz = fs.GetLength();
        res->yresize(sz);
        yint readCount = fs.Read(&(*res)[0], sz);
        if (readCount == sz) {
            return true;
        }
    }
    res->resize(0);
    return false;
}


void ReadNonEmptyLines(TVector<TString> *pRes, const TString &fName)
{
    TSeqReader fr(fName);
    Y_VERIFY(fr.IsValid() && "file not found");
    while (!fr.IsEof()) {
        TString sz = fr.ReadLine();
        if (!sz.empty()) {
            pRes->push_back(sz);
        }
    }
}
