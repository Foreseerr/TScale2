# fo

Purpose of fo.cpp is to generate CMakeLists.txt automatically determining folder type (executable or library) and inferring dependencies from #include in source files.

fo.cpp traverses source tree and creates a project for each leaf directory. It includes all .cpp and .h files in the directory as C++ source files and all .cu and .cuh files as CUDA source files.

Non leaf directories are expected to have no files, only subdirectories. Leaf directories are expected to contain source files. All source files in the leaf directory are added to the project. Precompiled headers are used, each directory besides util-allocator is expected to contain stdafx.h and stadafx.cpp.

Fo has two optional arguments.
- -nocuda ignores all folders with cuda files and folders that depend on them
- -noib produces CMakeLists.txt which can be compiled with no rdma headers installed
