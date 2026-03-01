#include "gpt2_tokenizer.h"
#include <lib/file/fmt_reader.h>
#include <lib/json/json.h>


using namespace NJson;

TGpt2Tokenizer::TGpt2Tokenizer() : Splitter("'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+") {}


void TGpt2Tokenizer::Encode(const TString &str, TVector<int> *p)
{
    std::vector<int> tokens;

    // Use regex to try and split text into smaller chunks
    std::string stdStr = str.c_str();
    std::sregex_iterator iter(stdStr.begin(), stdStr.end(), Splitter);
    // A default-constructed std::sregex_iterator represents the past-the-end iterator
    std::sregex_iterator end_iter;

    // while there are chunks left, tokenize them
    for (; iter != end_iter; ++iter) {
        std::string part = iter->str();

        TVector<int> frag;
        for (ui8 c : part) {
            int token = ByteToken[c];
            Y_VERIFY(token >= 0);
            frag.push_back(token);
        }

        for (;;) {
            yint bestRank = MAX_INT64;
            yint bestPos = 0;
            yint bestToken = 0;
            // quadratic, but who cares
            for (int i = 1; i < YSize(frag); ++i) {
                TMerge mrg;
                mrg.Left = frag[i - 1];
                mrg.Right = frag[i];
                auto it = Merges.find(mrg);
                if (it != Merges.end()) {
                    const TMergeResult &mres = it->second;
                    if (mres.Rank < bestRank) {
                        bestRank = mres.Rank;
                        bestPos = i;
                        bestToken = mres.ResToken;
                    }
                }
            }
            if (bestRank < MAX_INT64) {
                frag[bestPos - 1] = bestToken;
                frag.erase(frag.begin() + bestPos);
            } else {
                break;
            }
        }

        for (int token : frag) {
            p->push_back(token);
        }
    }
}


TString TGpt2Tokenizer::Decode(const TVector<int> &tokenArr)
{
    TString res;
    for (int token : tokenArr) {
        res += Tokens[token];
    }
    return res;
}


static void UnicodeToBytes(THashMap<ui32, ui32> *p)
{
    // Purpose: Create a specific bijective mapping between byte values (0-255) and Unicode code points.
    // This mapping is designed to be consistent with GPT-2's original tokenization scheme.

    TVector<ui32> bs;
    // Step 1: Add printable ASCII characters (33 to 126, i.e., '!' to '~')
    // Note: We will handl 0-32 (and the other missing values) later
    for (int i = 33; i <= 126; ++i)
        bs.push_back(i);
    // Step 2: Add extended ASCII characters (161 - '¡' to 172 - '¬' and 174 - '®'to 255 - 'ÿ')
    for (int i = 161; i <= 172; ++i)
        bs.push_back(i);
    for (int i = 174; i <= 255; ++i)
        bs.push_back(i);

    // Create a copy of bs to store the Unicode mappings
    TVector<ui32> cs = bs;
    int n = 0;
    // Step 3: Map remaining byte values (0-32, 127-160, 173) to Unicode points starting at 256
    // This includes control characters, space, delete, and some extended ASCII characters
    // Mapping these to 256+ ensures:
    // 1. Consistency with GPT-2's original tokenization scheme
    // 2. Clear visual distinction of special characters during debugging
    // 3. Avoidance of potential issues with the way text editors handle control characters

    for (int b = 0; b < 256; ++b) {

        // if we have already added this byte, skip it
        if (std::find(bs.begin(), bs.end(), b) != bs.end())
            continue;

        bs.push_back(b);
        // Map to Unicode characters starting from 256
        // Note: we add 256 to avoid conflicts with the ASCII range
        cs.push_back(256 + n);
        ++n;
    }

    // Create the final mapping
    // Note: We need to use char32_t rather than char to handle Unicode code points over 255
    //p->resize(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        //(*p)[bs[i]] = cs[i];
        (*p)[cs[i]] = bs[i];
    }
}


static TString DecodeUB(const TString &mstr, const THashMap<ui32, ui32> &ub)
{
    TString res;
    IterateUtf8Chars(mstr, [&](ui32 code) {
        auto it = ub.find(code);
        if (it != ub.end()) {
            res += (char)it->second;
        } else {
            Y_VERIFY(0);
        }
    });
    return res;
}


void TGpt2Tokenizer::Import(const TString &vocabJson, const TString &mergesFname)
{
    THashMap<ui32, ui32> ub;
    UnicodeToBytes(&ub);
    {
        TVector<char> vocabJsonText;
        ReadWholeFile(vocabJson, &vocabJsonText);
        TIntrusivePtr<TJson> vocab = ParseJson(vocabJsonText);
        TJsonIterator root(vocab);
        yint maxId = 0;
        ByteToken.resize(256, -1);
        if (root.IsObject()) {
            for (TJsonIterator it(root.Expand()); it.IsValid(); it.Next()) {
                if (it.IsValue()) {
                    TString token = DecodeUB(it.GetName(), ub);
                    yint id = it.GetFloatValue();
                    TokenId[token] = id;
                    maxId = Max<yint>(maxId, id);
                    if (YSize(token) == 1) {
                        ByteToken[(ui8)token[0]] = id;
                    }
                }
            }
        }
        Tokens.resize(maxId + 1);
        for (auto it = TokenId.begin(); it != TokenId.end(); ++it) {
            Tokens[it->second] = it->first;
        }
    }
    {
        TSeqReader fPair(mergesFname);
        Y_VERIFY(fPair.IsValid());
        yint mergeId = 0;
        while (!fPair.IsEof()) {
            TString line = fPair.ReadLine();
            if (line.empty() || line[0] == '#') {
                continue;
            }
            TVector<const char *> words;
            Split(&line[0], &words, ' ');
            if (YSize(words) == 2) {
                TMerge mrg;
                mrg.Left = GetToken(DecodeUB(words[0], ub));
                mrg.Right = GetToken(DecodeUB(words[1], ub));
                Y_VERIFY(mrg.Left >= 0 && mrg.Right >= 0);
                TMergeResult mres;
                mres.Rank = mergeId++;
                mres.ResToken = GetToken(Tokens[mrg.Left] + Tokens[mrg.Right]);
                Merges[mrg] = mres;
            }
        }
    }
}
