#pragma once


struct TKVcacheTracker
{
    yint MaxWidth = 0;
    TVector<int> FreeList;

    void Init(yint maxWidth, yint kvCacheSize)
    {
        MaxWidth = maxWidth;
        FreeList.resize(kvCacheSize);
        for (yint i = 0; i < kvCacheSize; ++i) {
            FreeList[i] = i;
        }
    }

    int Alloc()
    {
        Y_VERIFY(!FreeList.empty());
        int res = FreeList.back();
        FreeList.pop_back();
        return res;
    }

    void Free(int k)
    {
        Y_ASSERT(!IsInSet(FreeList, k));
        FreeList.push_back(k);
    }
};


struct TKVcacheReference
{
    yint Time = -1;
    yint KVwrite = -1;
    TVector<int> KVrefs;

public:
    bool IsEmpty() const { return KVrefs.empty(); }
    
    void Next(TKVcacheTracker *pCache)
    {
        // add previous entry to reference list
        if (KVwrite >= 0) {
            KVrefs.push_back(KVwrite);
        }
        // limit 
        if (pCache->MaxWidth > 0 && YSize(KVrefs) > pCache->MaxWidth) {
            pCache->Free(KVrefs[0]);
            KVrefs.erase(KVrefs.begin());
        }
        ++Time;
        KVwrite = pCache->Alloc();
    }

    void Free(TKVcacheTracker *pCache)
    {
        for (yint ptr : KVrefs) {
            pCache->Free(ptr);
        }
        if (KVwrite >= 0) {
            pCache->Free(KVwrite);
        }
        *this = TKVcacheReference();
    }
};
