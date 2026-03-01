#pragma once
#include <regex>


class TGpt2Tokenizer
{
    struct TMerge
    {
        int Left = 0;
        int Right = 0;

        bool operator==(const TMerge &x) const { return Left == x.Left && Right == x.Right; }
    };
    struct TMergeHash
    {
        yint operator()(const TMerge &x) const { return x.Left * 124567 + x.Right; }
    };
    struct TMergeResult
    {
        int Rank = 0;
        int ResToken = 0;
    };

private:
    TVector<TString> Tokens;
    THashMap<TString, yint> TokenId;
    THashMap<TMerge, TMergeResult, TMergeHash> Merges;
    TVector<int> ByteToken;
    std::regex Splitter;

private:
    yint GetToken(const TString &sz) const
    {
        auto it = TokenId.find(sz);
        if (it == TokenId.end()) {
            return -1;
        } else {
            return it->second;
        }
    }

public:
    SAVELOAD(Tokens, TokenId, Merges, ByteToken);

    TGpt2Tokenizer();
    void Encode(const TString &str, TVector<int> *p);
    TString Decode(const TVector<int> &tokenArr);
    TString GetToken(int tokenId) const { return Tokens[tokenId]; }
    void Import(const TString &vocabJson, const TString &mergesFname);
};