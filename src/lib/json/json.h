#pragma once
#include <util/string.h>


void PrintLog(const TString &str);

namespace NJson
{

///////////////////////////////////////////////////////////////////////////////////////////////////
enum EElementType {
    ELEM_NONE,
    ELEM_ARRAY,
    ELEM_OBJECT,
    ELEM_VALUE,
    ELEM_STRING
};

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TJsonStringPtr
{
    int Block = -1;
    int Ptr = 0;

    bool IsEmpty() const { return Block < 0; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TElement
{
    // name -> value, name can be empty (for root, arrays)
    TJsonStringPtr NamePos;
    TJsonStringPtr ValuePos;
    EElementType Type = ELEM_NONE;
    yint Next = -1;

    bool CanExpand() { return Type != ELEM_VALUE; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TJson : public TThrRefBase
{
    TVector<TVector<char>> StringBlocks;
    yint StringBlockPtr = 0;
    TVector<TElement> Elements;

public:
    TJson() { Clear(); }

    void Clear()
    {
        StringBlocks.reserve(1000);
        StringBlocks.resize(1);
        StringBlocks[0].resize(1 << 20);
        StringBlockPtr = 0;
        Elements.resize(0);
    }

    const char *GetString(const TJsonStringPtr &pos) const
    {
        if (pos.Block < 0) {
            return "";
        }
        return StringBlocks[pos.Block].data() + pos.Ptr;
    }

    TJsonStringPtr AddString(const TString &x)
    {
        yint sz = YSize(x);
        if (sz == 0) {
            return TJsonStringPtr();
        }
        if (StringBlockPtr + sz + 1 > YSize(StringBlocks.back())) {
            yint blkSize = Min<yint>(1 << 30, YSize(StringBlocks.back()) * 2);
            StringBlocks.push_back();
            StringBlocks.back().resize(blkSize);
            StringBlockPtr = 0;
        }
        TJsonStringPtr res;
        res.Block = YSize(StringBlocks) - 1;
        res.Ptr = StringBlockPtr;
        char *bufPtr = StringBlocks.back().data() + StringBlockPtr;
        memcpy(bufPtr, x.data(), sz);
        bufPtr[sz] = 0;
        StringBlockPtr += sz + 1;
        return res;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<TJson> ParseJson(const TVector<char> &str);
TVector<char> Render(TPtrArg<TJson> json);
TString RenderString(TPtrArg<TJson> json);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TJsonIterator
{
    TIntrusivePtr<TJson> Json;
    yint ElementId = 0;

public:
    TJsonIterator(TIntrusivePtr<TJson> p, yint id = 0) : Json(p), ElementId(id)
    {
        Y_VERIFY(p.Get());
        if (Json->Elements[id].Type == ELEM_NONE) {
            // empty object has no fields
            ElementId = -1;
        }
    }

    bool IsValid() const
    {
        return ElementId >= 0;
    }

    void Next()
    {
        ElementId = Json->Elements[ElementId].Next;
    }

    bool IsArray() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return elem.Type == ELEM_ARRAY;
    }

    bool IsObject() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return elem.Type == ELEM_OBJECT;
    }

    TJsonIterator Expand() const
    {
        Y_VERIFY(Json->Elements[ElementId].CanExpand());
        return TJsonIterator(Json, ElementId + 1);
    }

    const char *GetName() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return Json->GetString(elem.NamePos);
    }

    bool IsValue() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return !elem.ValuePos.IsEmpty();
    }

    const char *GetValue() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return Json->GetString(elem.ValuePos);
    }

    bool GetBoolValue() const
    {
        return ToLower(GetValue()) == "true";
    }

    double GetFloatValue() const
    {
        return atof(GetValue());
    }
};


template <class TFunc>
inline void Enum(const TJsonIterator &root, TFunc func)
{
    for (TJsonIterator f(root.Expand()); f.IsValid(); f.Next()) {
        func(f);
    }
}

template <class TFunc>
inline void EnumObjects(const TJsonIterator &root, TFunc func)
{
    for (TJsonIterator f(root.Expand()); f.IsValid(); f.Next()) {
        Y_VERIFY(f.IsObject());
        func(f);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// parse query
#define PARSE_CHK(f) if (!(f)) { PrintLog(#f); return false; }

template <class T>
inline bool ReadFloat(TJsonIterator &w, T *p)
{
    if (!w.IsValid()) {
        PrintLog("float parse error");
        return false;
    }
    *p = w.GetFloatValue();
    w.Next();
    return true;
}

inline void ReadField(TJsonIterator &w, const char *fieldName, TString *p)
{
    if (w.IsValid() && w.IsValue() && strcmp(w.GetName(), fieldName) == 0) {
        *p = w.GetValue();
    }
}

inline void ReadField(TJsonIterator &w, const char *fieldName, double *p)
{
    if (w.IsValid() && w.IsValue() && strcmp(w.GetName(), fieldName) == 0) {
        *p = w.GetFloatValue();
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TJsonWriter : public TNonCopyable
{
    TIntrusivePtr<TJson> Json;
    TVector<yint> ElementStack;

    yint AddNode(bool doPush)
    {
        yint elemId = YSize(Json->Elements);
        if (!ElementStack.empty()) {
            if (ElementStack.back() >= 0) {
                TElement &prev = Json->Elements[ElementStack.back()];
                prev.Next = elemId;
            }
            ElementStack.back() = elemId;
        }
        Json->Elements.push_back();
        if (doPush) {
            ElementStack.push_back(-1);
        }
        return elemId;
    }

    TElement &AddElement(bool doPush, const TString &name, EElementType tt)
    {
        TElement &add = Json->Elements[AddNode(doPush)];
        add.NamePos = Json->AddString(name);
        add.Type = tt;
        return add;
    }
public:
    TJsonWriter(TPtrArg<TJson> json) : Json(json.Get())
    {
        Json->Clear();
    }

    void AddValue(const TString &name, const TString &value)
    {
        TElement &add = AddElement(false, name, ELEM_VALUE);
        add.ValuePos = Json->AddString(value);
    }

    void AddString(const TString &name, const TString &value)
    {
        TElement &add = AddElement(false, name, ELEM_STRING);
        add.ValuePos = Json->AddString(value);
    }

    void AddBool(const TString &name, bool x)
    {
        TElement &add = AddElement(false, name, ELEM_VALUE);
        add.ValuePos = Json->AddString(x ? "True" : "False");
    }

    void AddFloat(const TString &name, double x)
    {
        TElement &add = AddElement(false, name, ELEM_VALUE);
        add.ValuePos = Json->AddString(Sprintf("%g", x));
    }

    void AddArray(const TString &name) { AddElement(true, name, ELEM_ARRAY); }

    void AddObject(const TString &name) { AddElement(true, name, ELEM_OBJECT); }

    void Finish()
    {
        Y_VERIFY(!ElementStack.empty());
        ElementStack.pop_back();
    }
};


void Test();
}

