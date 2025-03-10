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
#include <miopen/miopen.h>

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

// cuDNN -> miopen
#define cudnnHandle_t miopenHandle_t
#define cudnnStatus_t miopenStatus_t
#define cudnnDataType_t miopenDataType_t
#define cudnnDataType_t miopenDataType_t
#define CUDNN_STATUS_SUCCESS miopenStatus_t::miopenStatusSuccess
#define CUDNN_DATA_HALF miopenDataType_t::miopenHalf
#define CUDNN_DATA_FLOAT miopenDataType_t::miopenFloat
#define CUDNN_BATCHNORM_SPATIAL miopenBatchNormMode_t::miopenBNSpatial
#define CUDNN_CROSS_CORRELATION miopenConvolutionMode_t::miopenConvolution
#define cudnnTensorDescriptor_t miopenTensorDescriptor_t
#define cudnnFilterDescriptor_t miopenTensorDescriptor_t
#define cudnnPoolingDescriptor_t miopenPoolingDescriptor_t
#define cudnnConvolutionDescriptor_t miopenConvolutionDescriptor_t
#define cudnnConvolutionFwdAlgo_t miopenConvFwdAlgorithm_t
#define cudnnConvolutionBwdDataAlgo_t miopenConvBwdDataAlgorithm_t
#define cudnnConvolutionBwdFilterAlgo_t miopenConvBwdWeightsAlgorithm_t
#define cudnnDestroyTensorDescriptor miopenDestroyTensorDescriptor
#define cudnnCreateTensorDescriptor miopenCreateTensorDescriptor
#define cudnnBatchNormalizationForwardTraining miopenBatchNormalizationForwardTraining
#define cudnnBatchNormalizationBackward miopenBatchNormalizationBackward
#define cudnnCreateConvolutionDescriptor miopenCreateConvolutionDescriptor
#define cudnnCreateFilterDescriptor miopenCreateTensorDescriptor /* TODO: check this, does tensor replace filter? */
#define cudnnCreate miopenCreate
#define cudnnConvolutionMode_t miopenConvolutionMode_t
#define cudnnSetStream miopenSetStream

// MIOpen doesn't have the NCHW parameter; need to go from 7 parms down to 6 parms
#define CUDNN_TENSOR_NCHW 0 // we'll ignore this parameter, so give it a dummy value
#define cudnnSetTensor4dDescriptor(d, i, t, b, c, h, w) miopenSet4dTensorDescriptor(d, t, b, c, h, w)
#define cudnnSetFilter4dDescriptor(d, t, i, n, c, h, w) miopenSet4dTensorDescriptor(d, t, n, c, h, w)

// Parms are in a different order
#define cudnnSetConvolution2dDescriptor(d, h, w, u, v, x, y, mode)									miopenInitConvolutionDescriptor(d, mode, h, w, u, v, x, y)
#define cudnnConvolutionForward(h, alpha, xd, x, wd, w, cd, algo, ws, s, beta, yd, y)				miopenConvolutionForward(h, alpha, xd, x, wd, w, cd, algo, beta, yd, y, ws, s)
#define cudnnConvolutionBackwardData(h, alpha, wd, w, dyd, dy, cd, algo, ws, s, beta, dxd, dx)		miopenConvolutionBackwardData(h, alpha, dyd, dy, wd, w, cd, algo, beta, dxd, dx, wd, s)
#define cudnnConvolutionBackwardFilter(h, alpha, xd, x, dyd, dy, cd, algo, ws, s, beta, dwd, dw)	miopenConvolutionBackwardWeights(h, alpha, dyd, dy, xd, x, cd, algo, beta, dwd, dw, ws, s)
#define cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd, algo, s)							miopenConvolutionForwardGetWorkSpaceSize(h, wd, xd, cd, yd, s)
#define cudnnGetConvolutionBackwardFilterWorkspaceSize(h, xd, dyd, cd, gd, algo, s)					miopenConvolutionBackwardWeightsGetWorkSpaceSize(h, dyd, xd, cd, dwd, s)
#define cudnnGetConvolutionBackwardDataWorkspaceSize(h, wd, dyd, cd, dxd, algo, s)					miopenConvolutionBackwardDataGetWorkSpaceSize(h, dyd, wd, cd, dxd, s)

#endif
// === END OF AMD ROCm ===
