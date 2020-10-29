#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum PD_DataType { PD_FLOAT32, PD_INT32, PD_INT64, PD_UINT8, PD_UNKDTYPE };

typedef enum PD_DataType PD_DataType;

typedef struct PD_PaddleBuf PD_PaddleBuf;
typedef struct PD_AnalysisConfig PD_AnalysisConfig;
typedef struct PD_Predictor PD_Predictor;

typedef struct PD_Buffer {
  void* data;
  size_t length;
  size_t capacity;
} PD_Buffer;

// ZeroCopyTensor
typedef struct PD_ZeroCopyTensor {
  PD_Buffer data;
  PD_Buffer shape;
  PD_Buffer lod;
  PD_DataType dtype;
  char* name;
} PD_ZeroCopyTensor;

PD_ZeroCopyTensor* PD_NewZeroCopyTensor();
void PD_DeleteZeroCopyTensor(PD_ZeroCopyTensor*);
void PD_InitZeroCopyTensor(PD_ZeroCopyTensor*);
void PD_DestroyZeroCopyTensor(PD_ZeroCopyTensor*);

// AnalysisConfig
enum Precision { kFloat32 = 0, kInt8, kHalf };
typedef enum Precision Precision;

PD_AnalysisConfig* PD_NewAnalysisConfig();

void PD_DeleteAnalysisConfig(PD_AnalysisConfig* config);

void PD_SetModel(PD_AnalysisConfig* config, const char* model_dir, const char* params_path);

void PD_SetProgFile(PD_AnalysisConfig* config, const char* x);

void PD_SetParamsFile(PD_AnalysisConfig* config, const char* x);

void PD_SetOptimCacheDir(PD_AnalysisConfig* config, const char* opt_cache_dir);

const char* PD_ModelDir(const PD_AnalysisConfig* config);

const char* PD_ProgFile(const PD_AnalysisConfig* config);

const char* PD_ParamsFile(const PD_AnalysisConfig* config);

void PD_EnableUseGpu(PD_AnalysisConfig* config, int memory_pool_init_size_mb, int device_id);

void PD_DisableGpu(PD_AnalysisConfig* config);

bool PD_UseGpu(const PD_AnalysisConfig* config);

int PD_GpuDeviceId(const PD_AnalysisConfig* config);

int PD_MemoryPoolInitSizeMb(const PD_AnalysisConfig* config);

float PD_FractionOfGpuMemoryForPool(const PD_AnalysisConfig* config);

void PD_EnableCUDNN(PD_AnalysisConfig* config);

bool PD_CudnnEnabled(const PD_AnalysisConfig* config);

void PD_SwitchIrOptim(PD_AnalysisConfig* config, bool x);

bool PD_IrOptim(const PD_AnalysisConfig* config);

void PD_SwitchUseFeedFetchOps(PD_AnalysisConfig* config, bool x);

bool PD_UseFeedFetchOpsEnabled(const PD_AnalysisConfig* config);

void PD_SwitchSpecifyInputNames(PD_AnalysisConfig* config, bool x);

bool PD_SpecifyInputName(const PD_AnalysisConfig* config);

void PD_EnableTensorRtEngine(PD_AnalysisConfig* config, int workspace_size, int max_batch_size,
    int min_subgraph_size, Precision precision, bool use_static, bool use_calib_mode);

bool PD_TensorrtEngineEnabled(const PD_AnalysisConfig* config);

void PD_SwitchIrDebug(PD_AnalysisConfig* config, bool x);

void PD_EnableMKLDNN(PD_AnalysisConfig* config);

void PD_SetMkldnnCacheCapacity(PD_AnalysisConfig* config, int capacity);

bool PD_MkldnnEnabled(const PD_AnalysisConfig* config);

void PD_SetCpuMathLibraryNumThreads(PD_AnalysisConfig* config, int cpu_math_library_num_threads);

int PD_CpuMathLibraryNumThreads(const PD_AnalysisConfig* config);

void PD_EnableMkldnnQuantizer(PD_AnalysisConfig* config);

bool PD_MkldnnQuantizerEnabled(const PD_AnalysisConfig* config);

void PD_SetModelBuffer(PD_AnalysisConfig* config, const char* prog_buffer,
    size_t prog_buffer_size, const char* params_buffer, size_t params_buffer_size);

bool PD_ModelFromMemory(const PD_AnalysisConfig* config);

void PD_EnableMemoryOptim(PD_AnalysisConfig* config);

bool PD_MemoryOptimEnabled(const PD_AnalysisConfig* config);

void PD_EnableProfile(PD_AnalysisConfig* config);

bool PD_ProfileEnabled(const PD_AnalysisConfig* config);

void PD_SetInValid(PD_AnalysisConfig* config);

bool PD_IsValid(const PD_AnalysisConfig* config);

void PD_DisableGlogInfo(PD_AnalysisConfig* config);

void PD_DeletePass(PD_AnalysisConfig* config, char* pass_name);

// Predictor
PD_Predictor* PD_NewPredictor(const PD_AnalysisConfig* config);

void PD_DeletePredictor(PD_Predictor* predictor);

int PD_GetInputNum(const PD_Predictor*);

int PD_GetOutputNum(const PD_Predictor*);

const char* PD_GetInputName(const PD_Predictor*, int);

const char* PD_GetOutputName(const PD_Predictor*, int);

void PD_SetZeroCopyInput(PD_Predictor* predictor, const PD_ZeroCopyTensor* tensor);

void PD_GetZeroCopyOutput(PD_Predictor* predictor, PD_ZeroCopyTensor* tensor);

void PD_ZeroCopyRun(PD_Predictor* predictor);

#ifdef __cplusplus
}  // extern "C"
#endif