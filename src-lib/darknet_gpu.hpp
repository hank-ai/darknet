/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2025 Stephane Charette
 */

#pragma once

/** @file
 * Some GPU-specific includes and definitions.  Handles differences between NVIDIA CUDA and AMD ROCm.
 */


// ============================
// === START OF NVIDIA CUDA ===
// ============================
#ifdef DARKNET_GPU_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cublas_v2.h>

#ifdef CUDNN
#include <cudnn.h>
#endif // CUDNN

#endif
// === END OF NVIDIA CUDA ===


// =========================
// === START OF AMD ROCm ===
// =========================
#ifdef DARKNET_GPU_ROCM

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <hiprand/hiprand.h>
#include <hipblas/hipblas.h>

#define SHFL_DOWN(val, offset) shfl_xor(val, offset)
#define CUDA_SUCCESS hipSuccess
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError hipError_t
#define cudaError_t hipError_t
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemset hipMemset
#define cudaReadModeElementType hipReadModeElementType
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaStream_t hipStream_t
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaEventDestroy hipEventDestroy
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaHostAlloc hipHostMalloc
#define cudaDeviceProp hipDeviceProp_t
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define cudaSetDeviceFlags hipSetDeviceFlags

// Contexts
#define CUcontext hipCtx_t
#define cuCtxGetCurrent hipCtxGetCurrent
#define CUresult hipError_t

// CUBLAS -> HIPBLAS
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasHandle_t hipblasHandle_t
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define cublasSgemm hipblasSgemm
#define cublasStatus_t hipblasStatus_t
#define cublasCreate hipblasCreate

// CURAND -> HIPRAND
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define curandGenerator_t hiprandGenerator_t
#define curandGenerateUniform hiprandGenerateUniform
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandCreateGenerator hiprandCreateGenerator

// cuDNN -> hipDNN
#define cudnnHandle_t hipdnnHandle_t

#endif
// === END OF AMD ROCm ===
