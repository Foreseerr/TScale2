#include "gzip.h"
#include <util/fast_io.h>
#include <util/sse_util.h>


namespace NDeflate
{
struct TGzipReader
{
    TVector<char> Packed;
    yint Size = 0;
    yint Ptr = 0;
    ui64 Bits = 0;
    yint BitCount = 0;

    TGzipReader(const TString &fname)
    {
        Y_VERIFY(ReadWholeFile(fname, &Packed));
        Size = YSize(Packed);
    }

    // byte reading
    ui8 Read8()
    {
        Y_VERIFY(Ptr < Size);
        return (ui8)Packed[Ptr++];
    }

    ui8 Read16()
    {
        return Read8() + Read8() * 256;
    }

    void Skip(yint k) { Ptr += k; }

    void SkipString()
    {
        while (Read8()) {
        }
    }

    // Bit stream reading
    ui64 ReadBit()
    {
        Y_ASSERT(BitCount > 0);
        yint rv = Bits & 1;
        Bits >>= 1;
        --BitCount;
        return rv;
    }

    ui64 ReadBits(yint sz)
    {
        Y_ASSERT(BitCount >= sz);
        ui64 rv = Bits & ((1 << sz) - 1);
        Bits >>= sz;
        BitCount -= sz;
        return rv;
    }

    ui64 LookAheadBits(yint sz) { return Bits & ((1 << sz) - 1); }

    void SkipBits(yint sz)
    {
        Bits >>= sz;
        BitCount -= sz;
        Y_ASSERT(BitCount >= 0);
    }

    void PrefetchBits()
    {
        ui64 newBits = *(ui64 *)(Packed.data() + Ptr);
        Bits |= newBits << BitCount;
        yint byteCount = (64 - BitCount) / 8;
        BitCount += byteCount * 8;
        Ptr += byteCount;
        Y_VERIFY(Ptr <= Size);
    }

    void StopBitsReader()
    {
        Ptr -= BitCount / 8;
        BitCount = 0;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TDeflateTree
{
    enum {
        PRECOMP_BITS = 8
    };

    struct TPrecomputeEntry
    {
        int Len;
        int Sum;
        int Cur;
        int Result;
    };

    yint CodeLengthCount[16];  // indexed by code length
    ui16 Trans[288]; // code -> symbol translation table
    TPrecomputeEntry PrecomputeArr[1 << PRECOMP_BITS];

    void Build(const ui8 *lengths, yint count) 
    {
        Zero(CodeLengthCount);
        for (yint i = 0; i < count; ++i) {
            CodeLengthCount[lengths[i]]++;
        }
        CodeLengthCount[0] = 0; // do not count unused codes

        yint offset[16];
        yint sum = 0;
        for (yint i = 0; i < 16; ++i) {
            offset[i] = sum;
            sum += CodeLengthCount[i];
        }
        Y_VERIFY(sum <= ARRAY_SIZE(Trans));

        for (yint i = 0; i < count; ++i) {
            if (lengths[i]) {
                Trans[offset[lengths[i]]++] = i;
            }
        }
    }

    void DecodeBit(yint len, yint &cur, yint &sum, ui64 bit)
    {
        cur = 2 * cur + bit;
        yint count = CodeLengthCount[len];
        sum += count;
        cur -= count;
    }

    void Precompute()
    {
        for (yint k = 0; k < (1 << PRECOMP_BITS); ++k) {
            yint cur = 0;
            yint sum = 0;
            ui64 bits = k;
            TPrecomputeEntry &pe = PrecomputeArr[k];
            bool completeCode = false;
            for (yint len = 1; len <= PRECOMP_BITS; ++len) {
                DecodeBit(len, cur, sum, bits & 1);
                bits >>= 1;
                if (cur < 0) {
                    // complete code fits into PRECOMP_BITS
                    pe.Len = len;
                    pe.Sum = 0;
                    pe.Cur = 0;
                    pe.Result = Trans[sum + cur];
                    completeCode = true;
                    break;
                }
            }
            if (!completeCode) {
                // incomplete code
                pe.Len = PRECOMP_BITS;
                pe.Sum = sum;
                pe.Cur = cur;
                pe.Result = -1;
            }
            PrecomputeArr[k] = pe;
        }
    }

    yint Read(TGzipReader &gzip)
    {
        yint cur = 0;
        yint sum = 0;
        for (yint len = 1; len < ARRAY_SIZE(CodeLengthCount); ++len) {
            DecodeBit(len, cur, sum, gzip.ReadBit());
            if (cur < 0) {
                return Trans[sum + cur];
            }
        }
        Y_VERIFY(0);
        return 0;
    }

    // uses Precompute() constructed table
    yint ReadFast(TGzipReader &gzip)
    {
        const TPrecomputeEntry &pe = PrecomputeArr[gzip.LookAheadBits(PRECOMP_BITS)];
        if (pe.Result < 0) {
            gzip.SkipBits(PRECOMP_BITS);
            yint cur = pe.Cur;
            yint sum = pe.Sum;
            for (yint len = PRECOMP_BITS + 1; len < ARRAY_SIZE(CodeLengthCount); ++len) {
                DecodeBit(len, cur, sum, gzip.ReadBit());
                if (cur < 0) {
                    return Trans[sum + cur];
                }
            }
            Y_VERIFY(0);
            return 0;
        } else {
            gzip.SkipBits(pe.Len);
            return pe.Result;
        }
    }
};


// special ordering
const unsigned char CodeLengthIdx[] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

// read dynamic trees
static void DecodeTrees(TGzipReader &gzip, TDeflateTree *letterTree, TDeflateTree *distTree)
{
    // code lengths for 288 literal/len symbols and 32 dist symbols
    ui8 lengths[288 + 32];

    gzip.PrefetchBits();
    yint hLetter = gzip.ReadBits(5) + 257;
    yint hDist = gzip.ReadBits(5) + 1;
    yint hCodeLength = gzip.ReadBits(4) + 4;

    // code length
    for (yint i = 0; i < 19; ++i) {
        lengths[i] = 0;
    }
    for (yint i = 0; i < hCodeLength; ++i) {
        lengths[CodeLengthIdx[i]] = gzip.ReadBits(3);
        gzip.PrefetchBits();
    }
    TDeflateTree clTree;
    clTree.Build(lengths, 19);

    // decode code lengths for the dynamic trees
    for (yint num = 0; num < hLetter + hDist;) {
        yint sym = clTree.Read(gzip);
        if (sym < 16) {
            lengths[num++] = sym;
        } else {
            // blocks
            ui8 fillValue = 0;
            yint fillLen = 0;
            if (sym == 16) {
                // copy previous
                Y_VERIFY(num > 0);
                fillValue = lengths[num - 1];
                fillLen = gzip.ReadBits(2) + 3;
            } else if (sym == 17) {
                fillLen = gzip.ReadBits(3) + 3;
            } else if (sym == 18) {
                fillLen = gzip.ReadBits(7) + 11;
            }
            while (fillLen-- > 0) {
                lengths[num++] = fillValue;
            }
        }
        gzip.PrefetchBits();
    }
    letterTree->Build(lengths, hLetter);
    distTree->Build(lengths + hLetter, hDist);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TResultBuf
{
    TVector<char> Buf;
    yint BufSize = 0;
    yint Size = 0;

    void Alloc(yint sz)
    {
        while (BufSize < sz) {
            BufSize *= 2;
        }
        Buf.yresize(BufSize);
    }

public:
    TResultBuf()
    {
        BufSize = 8192;
        Buf.yresize(BufSize);
    }

    void Write(ui8 c)
    {
        if (Size == BufSize) {
            Alloc(Size + 1);
        }
        Buf[Size++] = c;
    }

    void WriteBlock(yint blockDist, yint blockLen)
    {
        if (Size + blockLen > BufSize) {
            Alloc(Size + blockLen);
        }
        ui8 *dst = (ui8*)Buf.data() + Size;
        ui8 *src = dst - blockDist;
        Y_VERIFY(src >= (ui8*)Buf.data());
        RepMovsb(dst, src, blockLen);
        Size += blockLen;
    }

    void Extract(TVector<char> *p)
    {
        Buf.resize(Size);
        p->swap(Buf);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
static ui8 DefaultLetterLengths[] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8};

static ui8 DefaultDistLengths[] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

static ui8 BlockLenBits[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5};
static yint BlockLenBase[] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

static ui8 DistBits[] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
static yint DistBase[] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193, 12289, 16385, 24577};

}
using namespace NDeflate;


static bool SkipHeader(TGzipReader &gzip)
{
    if (gzip.Read8() != 0x1f || gzip.Read8() != 0x8b || gzip.Read8() != 0x8) {
        return false;
    }
    yint bitMask = gzip.Read8();
    gzip.Skip(6);
    // extra
    if (bitMask & 4) {
        gzip.Skip(gzip.Read16());
    }
    // file name
    if (bitMask & 8) {
        gzip.SkipString();
    }
    // comment
    if (bitMask & 16) {
        gzip.SkipString();
    }
    // header crc
    if (bitMask & 2) {
        gzip.Read16();
    }
    return true;
}


void ReadGzip(const TString &fname, TVector<char> *p)
{
    p->resize(0);

    TResultBuf res;
    TGzipReader gzip(fname);
    Y_VERIFY(SkipHeader(gzip));

    TDeflateTree letterTree, distTree;
    gzip.PrefetchBits();
    for (bool isFinal = false; !isFinal;) {
        isFinal = gzip.ReadBit();
        yint blockType = gzip.ReadBits(2);

        if (blockType == 0) {
            // uncompressed block
            gzip.StopBitsReader();
            ui16 len = gzip.Read16();
            ui16 invLen = gzip.Read16();
            Y_VERIFY(len == ~invLen);
            for (yint k = 0; k < len; ++k) {
                res.Write(gzip.Read8());
            }
            gzip.PrefetchBits();

        } else if (blockType == 1 || blockType == 2) {
            // compressed block
            if (blockType == 1) {
                letterTree.Build(DefaultLetterLengths, ARRAY_SIZE(DefaultLetterLengths));
                distTree.Build(DefaultDistLengths, ARRAY_SIZE(DefaultDistLengths));
            } else {
                DecodeTrees(gzip, &letterTree, &distTree);
            }
            letterTree.Precompute();
            distTree.Precompute();

            for (;;) {
                yint letter = letterTree.ReadFast(gzip);
                if (letter < 256) {
                    // single char
                    res.Write(letter);
                } else if (letter == 256) {
                    // eob
                    break;
                } else {
                    // rep block
                    yint blk = letter - 257;
                    Y_VERIFY(blk < ARRAY_SIZE(BlockLenBase));
                    yint blockLen = gzip.ReadBits(BlockLenBits[blk]) + BlockLenBase[blk];

                    yint dist = distTree.ReadFast(gzip);
                    Y_VERIFY(dist < ARRAY_SIZE(DistBase));
                    yint blockDist = gzip.ReadBits(DistBits[dist]) + DistBase[dist];
                    
                    res.WriteBlock(blockDist, blockLen);
                }
                gzip.PrefetchBits();
            }

        } else {
            Y_VERIFY(0 && "incorrect block");
        }
    }
    gzip.StopBitsReader();
    res.Extract(p);
}
