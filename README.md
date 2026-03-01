
# TScale2

Update of transformer train [TScale](https://github.com/Foreseerr/TScale/) written in C++ and CUDA.

TScale2 differs from TScale in few aspects
- MoE support
- H100 support
- RDMA (infiniband) support
- Async distributed training deprecated

[How TScale MoE is different](doc/moe.md)

[Some train runs results](doc/runs.md)

Scripts are mostly the same, more documentation in TScale repo.

Define SIMULATE_MULTI_GPU can be used to simulate multi GPU host with single GPU machine. Allows using cheaper hw for development.

TScale does not use any external libraries for the sake of fun and full control.

# Build

To build the the code CUDA v13 and C++ compiler needed, Linux and Windows are supported. CMakeLists.txt is generated with [fo](doc/fo.md), lightweight build files generator. To generate CMakeLists.txt build files compile fo/fo.cpp and run it in repo root folder.

```bash
~/TScale2/fo$ clang++17 fo.cpp -o fo
~/TScale2/fo$ cd ..
~/TScale2$ ./fo/fo -noib
~/TScale2$ mkdir build
~/TScale2$ cd build
~/TScale2/build$ cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo ..
~/TScale2/build$ make
```

# License

MIT
