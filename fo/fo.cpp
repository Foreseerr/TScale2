#include <string>
#include <vector>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <algorithm>
#include <unordered_map>

// support just my code - unclear how to disable in VS code
// #set (CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/CompilerOptions.cmake")

using namespace std;

typedef string TString;
typedef long long yint;
typedef unsigned char ui8;
typedef unsigned short ui16;
typedef unsigned int ui32;
typedef unsigned long long ui64;
typedef ofstream TOFStream;

#define TVector vector
#define THashMap unordered_map

#define Y_ASSERT assert

template <class T>
yint YSize(const T &col)
{
    return (yint)col.size();
}

template <class T, class TElem>
inline bool IsInSet(const T &c, const TElem &e) { return find(c.begin(), c.end(), e) != c.end(); }

inline void Out(TOFStream &f, const TString &s) { f << s.c_str(); }

//////////////////////////////////////////////////////////////////////////////////////////////
struct TFindFileResult
{
    TString Name;
    bool IsDir = false;

    TFindFileResult() {}
    TFindFileResult(const TString &n, bool isDir) : Name(n), IsDir(isDir) {}
};

#ifdef _MSC_VER
#include <windows.h> // for findfirst

static void FindAllFiles(const TString &prefix, TVector<TFindFileResult> *res)
{
    res->resize(0);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((prefix + "/*.*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    if (fd.cFileName[0] != '.') {
        res->push_back(TFindFileResult(fd.cFileName, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)));
    }
    while (FindNextFileA(h, &fd)) {
        if (fd.cFileName[0] != '.') {
            res->push_back(TFindFileResult(fd.cFileName, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)));
        }
    }
    FindClose(h);
}

void MakeDirectory(const TString &dir)
{
    CreateDirectoryA(dir.c_str(), 0);
}

#else
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
//#include <unistd.h>
//#include <libgen.h>

static void FindAllFiles(const TString &prefix, TVector<TFindFileResult> *res)
{
    res->resize(0);
    DIR *dir = opendir(prefix.c_str());
    if (dir == NULL) {
        Y_ASSERT(0); // directory does not exist?
        return;
    }
    for (;;) {
        struct dirent *dp = readdir(dir);
        if (dp == 0) {
            break;
        }
        if (dp->d_name[0] == '.') {
            continue;
        }
        struct stat fprop;
        TString fname(dp->d_name);
        int rv = stat((prefix + "/" + fname).c_str(), &fprop);
        if (S_ISDIR(fprop.st_mode)) {
            res->push_back(TFindFileResult(fname, true));
        } else {
            res->push_back(TFindFileResult(fname, false));
        }
    }
    closedir(dir);
}

void MakeDirectory(const TString &dir)
{
    mkdir(dir.c_str(), 0777);
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
enum EFolderType
{
    FT_PROGRAM,
    FT_LIBRARY,
};

enum ESystemLibs
{
    LIB_CUDA = 1,
    LIB_IBVERBS = 2,
};

enum ESourceType
{
    ST_H,
    ST_CPP,
    ST_CUH,
    ST_CU,
};

enum EPrecompiledHeaders
{
    PCH_NONE,
    PCH_GEN,
    PCH_REUSE,
};

struct TSourceFile
{
    TString Name;
    ESourceType Type = ST_CPP;
};

struct TFolderPath
{
    TVector<TString> PathArr;
};

struct TSourceFolder
{
    TString Folder;
    EFolderType FType = FT_LIBRARY;
    EPrecompiledHeaders PCH = PCH_NONE;
    TFolderPath Path;
    yint UsingLibs = 0;
    TVector<TSourceFile> Files;
    THashMap<TString,bool> AllIncludeFolders;
    TVector<TFolderPath> DepArr;
};


//////////////////////////////////////////////////////////////////////////////////////////////
TString Sprintf(const char *pszFormat, ...)
{
    TString res;

    va_list va;
    va_start(va, pszFormat);
    yint len = vsnprintf(0, 0, pszFormat, va);
    res.resize(len + 1);
    yint resLen = vsnprintf(&res[0], YSize(res), pszFormat, va);
    res.resize(resLen);
    va_end(va);
    //
    return res;
}

static TString MakeString(char c)
{
    TString res;
    res.push_back(c);
    return res;
}


static bool EndsWith(TString &a, const TString &b)
{
    yint bsz = YSize(b);
    yint idx = YSize(a) - bsz;
    if (idx < 0) {
        return false;
    }
    for (yint i = 0; i < bsz; ++i) {
        if (a[idx + i] != b[i]) {
            return false;
        }
    }
    return true;
}

static TString GetStringPath(const TFolderPath &fp)
{
    if (fp.PathArr.empty()) {
        return "";
    }
    TString res = fp.PathArr[0];
    for (yint i = 1; i < YSize(fp.PathArr); ++i) {
        res += "/" + fp.PathArr[i];
    }
    return res;
}

static TFolderPath GetFolderPath(const TString &fp)
{
    TFolderPath res;
    TString folder;
    for (char c : fp) {
        if (c == '/') {
            res.PathArr.push_back(folder);
            folder = "";
        } else {
            folder += c;
        }
    }
    res.PathArr.push_back(folder);
    return res;
}

inline bool operator==(const TFolderPath &a, const TFolderPath &b)
{
    return a.PathArr == b.PathArr;
}

inline bool operator!=(const TFolderPath &a, const TFolderPath &b)
{
    return a.PathArr != b.PathArr;
}

static TString GetProjectName(const TFolderPath &fp)
{
    if (fp.PathArr.empty()) {
        return "root"; // never happens?
    }
    TString res = fp.PathArr[0];
    for (yint i = 1; i < YSize(fp.PathArr); ++i) {
        res += "-" + fp.PathArr[i];
    }
    return res;
}


//////////////////////////////////////////////////////////////////////////////////////////////
static void ParseSourceFile(const TString &fname, TSourceFolder *pProj)
{
    const char *mainFunc = "int main(";
    yint mainFuncLen = strlen(mainFunc);

    ifstream f(fname);
    while (f.good()) {
        const yint LINE_SIZE = 10000;
        static char szLine[LINE_SIZE];
        f.getline(szLine, LINE_SIZE);
        if (szLine[0] == '#') {
            TVector<TString> wordArr;
            TString word;
            for (const char *p = szLine + 1;; ++p) {
                char c = *p;
                if (c == ' ' || c == 9 || c == 0) {
                    if (!word.empty()) {
                        wordArr.push_back(word);
                        word = "";
                    }
                    if (c == 0) {
                        break;
                    }
                } else {
                    word += c;
                }
            }
            if (wordArr.size() == 2) {
                if (wordArr[0] == "include" && wordArr[1].size() > 2) {
                    TString path = wordArr[1];
                    if (path[0] == '<') {
                        yint fin = path.size() - 1;
                        while (fin > 1 && path[fin] != '/') {
                            --fin;
                        }
                        TString dep = path.substr(1, fin - 1);
                        if (!dep.empty()) {
                            pProj->AllIncludeFolders[dep];
                        }
                    }
                }
            }
        }
        if (strncmp(szLine, mainFunc, mainFuncLen) == 0) {
            pProj->FType = FT_PROGRAM;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
void ParseSourceDir(const TString &prefix, TSourceFolder *p, TVector<TSourceFolder> *resFolders);

static void AddFile(const TString &fnameArg, const TString &prefix, TSourceFolder *res, TVector<TSourceFolder> *resFolders)
{
    TString fname = fnameArg;
    for (char &c : fname) {
        c = (char)tolower((ui8)c);
    }
    TSourceFile sf;
    sf.Name = fnameArg;
    if (EndsWith(fname, ".cpp")) {
        sf.Type = ST_CPP;
        res->Files.push_back(sf);
    } else if (EndsWith(fname, ".h")) {
        sf.Type = ST_H;
        res->Files.push_back(sf);
    } else if (EndsWith(fname, ".cu")) {
        sf.Type = ST_CU;
        res->Files.push_back(sf);
        res->UsingLibs |= LIB_CUDA;
    } else if (EndsWith(fname, ".cuh")) {
        sf.Type = ST_CUH;
        res->Files.push_back(sf);
        res->UsingLibs |= LIB_CUDA;
    } else if (fname == "dockerfile") {
    } else {
        printf("Unkown file %s at %s\n", fnameArg.c_str(), prefix.c_str());
        abort();
    }
    ParseSourceFile(prefix + "/" + fname, res);
}

static void AddDir(const TString &dirname, const TString &prefix, TSourceFolder *res, TVector<TSourceFolder> *resFolders)
{
    TSourceFolder xx;
    xx.Folder = dirname;
    xx.Path = res->Path;
    xx.Path.PathArr.push_back(xx.Folder);
    ParseSourceDir(prefix + "/" + dirname, &xx, resFolders);
    if (!xx.Files.empty()) {
        // add util-allocator and util
        TFolderPath utilAlloc = GetFolderPath("util-allocator");
        if (xx.Path == utilAlloc) {
            xx.PCH = PCH_NONE;
        } else {
            xx.DepArr.push_back(utilAlloc);
            TFolderPath util = GetFolderPath("util");
            if (xx.Path == util) {
                xx.PCH = PCH_GEN;
            } else {
                xx.PCH = PCH_REUSE;
                xx.DepArr.push_back(util);
            }
        }
        // add to project
        resFolders->push_back(xx);
    }
}

void ParseSourceDir(const TString &prefix, TSourceFolder *res, TVector<TSourceFolder> *resFolders)
{
    TVector<TSourceFolder> newFolders;
    TVector<TFindFileResult> allFiles;
    FindAllFiles(prefix, &allFiles);
    for (const TFindFileResult &ff : allFiles) {
        if (ff.IsDir) {
            AddDir(ff.Name, prefix, res, &newFolders);
        } else {
            AddFile(ff.Name, prefix, res, &newFolders);
        }
    }
    // checks
    if (!res->Files.empty() && !newFolders.empty()) {
        printf("folder %s should have either files or subfolders\n", GetStringPath(res->Path).c_str());
        abort();
    }
    resFolders->insert(resFolders->end(), newFolders.begin(), newFolders.end());
}


//////////////////////////////////////////////////////////////////////////////////////////////
static void AddDependencies(TVector<TSourceFolder> *pProjArr)
{
    THashMap<TString, yint> projId;
    for (yint id = 0; id < YSize(*pProjArr); ++id) {
        TSourceFolder &proj = (*pProjArr)[id];
        projId[GetStringPath(proj.Path)] = id;
    }
    for (TSourceFolder &proj : *pProjArr) {
        TFolderPath fpSelf = proj.Path;
        for (auto it = proj.AllIncludeFolders.begin(); it != proj.AllIncludeFolders.end(); ++it) {
            TFolderPath fp = GetFolderPath(it->first);
            if (projId.find(GetStringPath(fp)) != projId.end() && fp != fpSelf) {
                proj.DepArr.push_back(fp);
            }
            if (it->first == "infiniband") {
                proj.UsingLibs |= LIB_IBVERBS;
            }
        }
    }
    // check that dependencies only on libs
    for (TSourceFolder &proj : *pProjArr) {
        for (const TFolderPath &fp : proj.DepArr) {
            yint id = projId[GetStringPath(fp)];
            const TSourceFolder &depProj = (*pProjArr)[id];
            if (depProj.FType != FT_LIBRARY) {
                printf("project %s depends on non-library %s\n", GetStringPath(proj.Path).c_str(), GetStringPath(depProj.Path).c_str());
                abort();
            }
        }
    }
}


static void PropagateUsing(TVector<TSourceFolder> *pProjArr)
{
    THashMap<TString, yint> projId;
    for (yint id = 0; id < YSize(*pProjArr); ++id) {
        TSourceFolder &proj = (*pProjArr)[id];
        projId[GetStringPath(proj.Path)] = id;
    }
    for (;;) {
        bool hasFinished = true;
        for (TSourceFolder &proj : *pProjArr) {
            yint usingLibs = 0;
            for (const TFolderPath &fp : proj.DepArr) {
                auto it = projId.find(GetStringPath(fp));
                if (it == projId.end()) {
                    abort(); // deparr references non existing project
                } else {
                    usingLibs |= (*pProjArr)[it->second].UsingLibs;
                }
            }
            if (usingLibs & ~proj.UsingLibs) {
                proj.UsingLibs |= usingLibs;
                hasFinished = false;
            }
        }
        if (hasFinished) {
            break;
        }
    }
}


static void DisablePchForCudaOnly(TVector<TSourceFolder> *pProjArr)
{
    for (TSourceFolder &proj : *pProjArr) {
        bool hasCpp = false;
        for (const TSourceFile &sf : proj.Files) {
            hasCpp |= (sf.Type == ST_CPP);
        }
        if (!hasCpp) {
            proj.PCH = PCH_NONE;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
struct TBuildOrder
{
    TVector<yint> Order;
    TVector<yint> OrderPlace;

    TBuildOrder(yint projCount)
    {
        OrderPlace.resize(projCount, -1);
    }
    void AddToTail(yint id)
    {
        yint place = OrderPlace[id];
        if (place >= 0) {
            Order[place] = -1;
        }
        OrderPlace[id] = YSize(Order);
        Order.push_back(id);
    }
};

static void MakeProgDependOnLibs(TVector<TSourceFolder> *pProjArr)
{
    THashMap<TString, yint> projId;
    yint projCount = YSize(*pProjArr);
    for (yint id = 0; id < projCount; ++id) {
        TSourceFolder &proj = (*pProjArr)[id];
        projId[GetStringPath(proj.Path)] = id;
    }
    for (TSourceFolder &proj : *pProjArr) {
        if (proj.FType == FT_LIBRARY) {
            continue;
        }
        TVector<yint> depDepth;
        depDepth.resize(projCount, 0);
        TBuildOrder order(projCount);
        for (const TFolderPath &fp : proj.DepArr) {
            yint depId = projId[GetStringPath(fp)];
            depDepth[depId] = 1;
            order.AddToTail(depId);
        }
        for (yint iter = 0;; ++iter) {
            bool hasChanged = false;
            TVector<yint> curOrder = order.Order;
            for (yint id : curOrder) {
                if (id >= 0) {
                    yint depth = depDepth[id];
                    const TSourceFolder &depProj = (*pProjArr)[id];
                    for (const TFolderPath &fp : depProj.DepArr) {
                        yint depId = projId[GetStringPath(fp)];
                        if (depDepth[depId] < depth + 1) {
                            depDepth[depId] = depth + 1;
                            order.AddToTail(depId);
                            hasChanged = true;
                        }
                    }
                }
            }
            if (!hasChanged) {
                break;
            }
            if (iter > projCount) {
                printf("project %s has circular dependencies for ?\n", GetProjectName(proj.Path).c_str());
                abort();
            }
        }
        proj.DepArr.resize(0);
        for (yint depId : order.Order) {
            if (depId >= 0) {
                proj.DepArr.push_back((*pProjArr)[depId].Path);
            }
        }
    }
    for (TSourceFolder &proj : *pProjArr) {
        if (proj.FType == FT_LIBRARY) {
            proj.DepArr.resize(0);
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
static void PrintCMakeProject(TOFStream &f, const TString &srcDir, yint allowLibs, yint hasLibs, const TSourceFolder &proj)
{
    if (proj.UsingLibs & ~allowLibs) {
        printf("Ignoring project %s\n", GetProjectName(proj.Path).c_str());
        return;
    }
    TString projType = (proj.FType == FT_LIBRARY) ? "add_library" : "add_executable";
    TString projName = GetProjectName(proj.Path);
    f << projType << "(" << projName << "\n";
    TString pathToSrc = srcDir + "/" + GetStringPath(proj.Path) + "/";
    for (const TSourceFile &srcFile : proj.Files) {
        if (srcFile.Type == ST_CPP || srcFile.Type == ST_CU) {
            f << "  " << pathToSrc + srcFile.Name << "\n";
        }
    }
    f << ")\n";
    // precompiled headers
    if (proj.PCH == PCH_GEN) {
        f << "target_precompile_headers(" << projName << " PRIVATE " << pathToSrc << "pch.h)\n";
    } else if (proj.PCH == PCH_REUSE) {
        f << "target_precompile_headers(" << projName << " REUSE_FROM util)\n";
    }
    // dependencies
    if (!proj.DepArr.empty()) {
        f << "target_link_libraries(" << projName;
        for (const TFolderPath &dep : proj.DepArr) {
            f << " " << GetProjectName(dep);
        }
        if (proj.UsingLibs & LIB_CUDA) {
            f << " CUDA::cuda_driver";
        }
        if (proj.UsingLibs & hasLibs & LIB_IBVERBS) {
            f << " ${IBVERBS_LIBRARY}";
        }
        f << " ${CMAKE_THREAD_LIBS_INIT}";
        f << ")\n";
    }
    // cuda
    if (proj.UsingLibs & LIB_CUDA) {
        f << "set_target_properties(" << projName << " PROPERTIES\n";
        //f << " CUDA_ARCHITECTURES \"80;86;89\"\n";
        f << " CUDA_ARCHITECTURES \"89\"\n";
        //f << " CUDA_ARCHITECTURES \"90a\"\n";
        f << ")\n";
    }
    f << "\n";
}


void GenerateCMake(const TVector<TSourceFolder> &projArr, const TString &projName, const TString &srcDir, yint allowLibs, yint hasLibs)
{
    TOFStream f("CMakeLists.txt");

    f << "cmake_minimum_required(VERSION 3.22)\n";
    f << "\n";
    if (allowLibs & LIB_CUDA) {
        f << "project(" << projName << " LANGUAGES CXX CUDA)\n";
    } else {
        f << "project(" << projName << " LANGUAGES CXX)\n";
    }
    f << "\n";
    f << "include_directories(\"" << srcDir << "\")\n";
    if (allowLibs & LIB_CUDA) {
        f << "include_directories(\"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}\")\n";
    }
    if (hasLibs & LIB_IBVERBS) {
        f << "find_library(IBVERBS_LIBRARY NAMES ibverbs)\n";
        f << "add_compile_definitions(PLATFORM_HAS_IBVERBS)\n";
    }
    f << "\n";
    //f << "set(CMAKE_CXX_STANDARD 14)\n";
    if (allowLibs & LIB_CUDA) {
        //f << "set(CMAKE_CUDA_STANDARD 14)\n";
        //f << "set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} --ptxas-options=-v\")\n";
        f << "set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} -lineinfo --use_fast_math\")\n";
        f << "find_package(CUDAToolkit REQUIRED)\n";
    }
    f << "find_package(Threads)\n";
    f << "\n";
    f << "if (MSVC_VERSION)\n";
    f << "  set(CMAKE_MSVC_RUNTIME_LIBRARY \"MultiThreaded\")\n";
    f << "  # add everywhere /W3 /permissive- /arch:AVX\n";
    f << "  set (CMAKE_CXX_FLAGS_INIT \"/DWIN32 /D_WINDOWS /EHsc /W3 /permissive- /arch:AVX\")\n";
    f << "  # release, add compiler optimizations /Ox /Ob2 /Oi, add linker optimizations  /OPT:REF /OPT:ICF\n";
    f << "  #   use ltcg /GL (Whole program optimization) compiler, /LTCG linker\n";
    f << "  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT \"/Zi /Ob2 /Ox /Oi /GL /DNDEBUG\")\n";
    f << "  set (CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO_INIT \"/debug /INCREMENTAL /LTCG /OPT:REF /OPT:ICF\")\n";
    f << "else()\n";
    f << "  add_compile_options(-march=native -mavxvnni)\n";
    f << "  if(CMAKE_CXX_COMPILER_ID STREQUAL \"GNU\")\n";
    f << "    add_compile_options(-Wno-stringop-overflow)\n";
    f << "  endif()\n";
    f << "endif()\n";
    f << "\n";

    for (const TSourceFolder &proj : projArr) {
        if (proj.PCH != PCH_REUSE) {
            PrintCMakeProject(f, srcDir, allowLibs, hasLibs, proj);
        }
    }
    for (const TSourceFolder &proj : projArr) {
        if (proj.PCH == PCH_REUSE) {
            PrintCMakeProject(f, srcDir, allowLibs, hasLibs, proj);
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    yint allowLibs = -1;
    yint hasLibs = -1;
    for (int k = 1; k < argc; ++k) {
        if (strcmp(argv[k], "-nocuda") == 0) {
            allowLibs -= (allowLibs & LIB_CUDA);
        }
        if (strcmp(argv[k], "-noib") == 0) {
            hasLibs -= (hasLibs & LIB_IBVERBS);
        }
    }
#ifdef _MSC_VER
    hasLibs -= (hasLibs & LIB_IBVERBS);
#endif
    TString srcDir = "src";
    TString projName = "eden";
    
    TSourceFolder root;
    TVector<TSourceFolder> projArr;
    ParseSourceDir(srcDir, &root, &projArr);
    if (!root.Files.empty()) {
        printf("expected no files at the root\n");
        abort();
    }
    AddDependencies(&projArr);
    PropagateUsing(&projArr);
    MakeProgDependOnLibs(&projArr);
    DisablePchForCudaOnly(&projArr);

    GenerateCMake(projArr, projName, srcDir, allowLibs, hasLibs);
    
    printf("Ok\n");
    return 0;
}
