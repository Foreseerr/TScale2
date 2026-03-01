#pragma once


//////////////////////////////////////////////////////////////////////////////////////////////////
// utf utils
extern ui8 Utf8CodeLength[256];

enum EWordCase
{
    WORD_LOWER_CASE,
    WORD_CAPITAL_START,
    WORD_MIXED_CASE,
};

EWordCase GetWordCase(const TString &str);
TString ToLower(const TString &str);
TString UpcaseFirstLetter(const TString &str);


//////////////////////////////////////////////////////////////////////////////////////////////////
// cp1251 encoding
TString Utf2Win(const TString &utf8);
TString Win2Utf(const TString &cp1251);
char Unicode2Win(yint key);

template <class TDst>
void Unicode2Utf(ui32 code, TDst *pBuf)
{
    if (code < 128) {
        pBuf->push_back(code);
    } else {
        if (code < 0x800) {
            pBuf->push_back(0xc0 + (code >> 6));
            pBuf->push_back(0x80 + (code & 0x3f));
        } else if (code < 0x10000) {
            pBuf->push_back(0xe0 + (code >> 12));
            pBuf->push_back(0x80 + ((code >> 6) & 0x3f));
            pBuf->push_back(0x80 + (code & 0x3f));
        } else {
            pBuf->push_back(0xf0 + (code >> 18));
            pBuf->push_back(0x80 + ((code >> 12) & 0x3f));
            pBuf->push_back(0x80 + ((code >> 6) & 0x3f));
            pBuf->push_back(0x80 + (code & 0x3f));
        }
    }
}

template <class TFunc>
bool IterateUtf8Chars(const TString &utf8, TFunc func)
{
    bool broken = false;
    yint sz = YSize(utf8);
    for (yint i = 0; i < sz; ++i) {
        ui8 f = utf8[i];
        ui32 code = 0;

        if (f < 128) {
            code = f;

        } else if ((f & 0xe0) == 0xc0) {
            if (i + 1 >= sz) {
                return false;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            code = (c1 & 0x3f) + ((c0 & 0x1f) << 6);

        } else if ((f & 0xf0) == 0xe0) {
            if (i + 2 >= sz) {
                return false;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            ui32 c2 = utf8[i + 2];
            code = (c2 & 0x3f) + ((c1 & 0x3f) << 6) + ((c0 & 0xf) << 12);

        } else if ((f & 0xf8) == 0xf0) {
            if (i + 3 >= sz) {
                return false;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            ui32 c2 = utf8[i + 2];
            ui32 c3 = utf8[i + 3];
            code = (c3 & 0x3f) + ((c2 & 0x3f) << 6) + ((c1 & 0x3f) << 12) + ((c0 & 0x7) << 18);

        } else {
            broken = true;
            continue; // not first utf encoding char somehow
        }

        func(code);
    }
    return true;
}
