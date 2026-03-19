#pragma once
#include <cuda_fp16.h>
#include <util/fp8.h>


enum {
    MATMUL_FP16,
    MATMUL_INT8,
    MATMUL_FP8,
};

enum {
    ATT_FP16,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// select matmul

// int8 matmul (requires i8 model matrices)
constexpr int FWD_MATMUL_TYPE = MATMUL_INT8;
constexpr int BWD_MATMUL_TYPE = MATMUL_INT8;
//constexpr int BWD_MATMUL_TYPE = MATMUL_FP16;
typedef half TFastGradientFloat;
typedef i8 TNormStateFloat;
typedef i8 TFastModelFloat;

// // fp8 matmul (requires e4m3 model matrices)
// constexpr int FWD_MATMUL_TYPE = MATMUL_FP8;
// constexpr int BWD_MATMUL_TYPE = MATMUL_FP8;
// //constexpr int BWD_MATMUL_TYPE = MATMUL_FP16;
// //typedef e5m2 TFastGradientFloat;
// typedef e4m3 TFastGradientFloat; // sufficient in most cases
// typedef e4m3 TNormStateFloat;
// typedef e4m3 TFastModelFloat;

// // fp16 matmul
// constexpr int FWD_MATMUL_TYPE = MATMUL_FP16;
// constexpr int BWD_MATMUL_TYPE = MATMUL_FP16;
// typedef half TFastGradientFloat;
// //typedef i8 TNormStateFloat;
// //typedef e4m3 TNormStateFloat;
// typedef half TNormStateFloat;
// //typedef i8 TFastModelFloat;
// //typedef e4m3 TFastModelFloat;
// typedef half TFastModelFloat;


///////////////////////////////////////////////////////////////////////////////////////////////////
// select attention

// fp16 attention
constexpr int ATT_TYPE = ATT_FP16;
constexpr int ATT_GROUP = 64;
constexpr int ATT_ALIGN = 16;
//typedef i8 TAttVecFloat;
//typedef e4m3 TAttVecFloat;
typedef half TAttVecFloat;


///////////////////////////////////////////////////////////////////////////////////////////////////
// config

typedef float TStateFloat; // use same float for state and state gradient
// typedef half TStateFloat; // worse models, on the order of 2.576 -> 2.579

typedef half TEmbedFloat;

//typedef float TRopeFloat;
typedef half TRopeFloat;

constexpr float STATE_VEC_SCALE = 1 / 24.0f;
constexpr float Q_VEC_SCALE = 1 / 24.0f;
constexpr float K_VEC_SCALE = 1 / 24.0f;
constexpr float V_VEC_SCALE = 1 / 24.0f;
constexpr float FFN_VEC_SCALE = 1 / 24.0f; // 1/24 is less stable then 1/16 for 1.5B with large lr
constexpr float MOE_VEC_SCALE = 1 / 24.0f;

constexpr int Q_DIM = 128;
constexpr int TT_DIM = 128;
