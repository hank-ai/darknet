#pragma once

#include "darknet_internal.hpp"

/// @todo V3 is this still needed?
extern int cuda_debug_sync;

#ifdef DARKNET_GPU

/// @todo What is this?  See where it is used in all the .cu files.
#define BLOCK 512

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define BLOCK_TRANSPOSE32 256

#include "darknet_gpu.hpp"

/// Use @ref CHECK_CUDA() instead.
void check_cuda_error(cudaError_t status, const char * const filename, const char * const funcname, const int line);

/// Use @ref CHECK_CUDA() instead.
void check_cuda_error_extended(cudaError_t status, const char * const filename, const char * const funcname, const int line);

/// Macro to quickly check if a CUDA error has taken place.  Only calls the CUDA error function if a problem is detected.
#define CHECK_CUDA(X)														\
{																			\
	const auto STATUS = X;													\
	if (STATUS != cudaSuccess)												\
	{																		\
		check_cuda_error_extended(STATUS, __FILE__, __func__, __LINE__);	\
	}																		\
}

/// Use @ref CHECK_CUBLAS() instead.
void cublas_check_error_extended(cublasStatus_t status, const char * const filename, const char * const funcname, const int line);

#define CHECK_CUBLAS(X) cublas_check_error_extended(X, __FILE__, __func__, __LINE__ );

cublasHandle_t blas_handle();
void free_pinned_memory();
void pre_allocate_pinned_memory(size_t size);

float *cuda_make_array_pinned_preallocated(float *x, size_t n);

float *cuda_make_array_pinned(float *x, size_t n);

/** Allocate memory on the GPU.  If @p x is not null, then copy the given floats from the host pointer.
 *
 * @returns a pointer to the CUDA memory allocation.
 *
 * @warning The copy is asynchronous and may not have finished when this function returns!
 */
float *cuda_make_array(float *x, size_t n);

void **cuda_make_array_pointers(void **x, size_t n);

int *cuda_make_int_array(size_t n);
int *cuda_make_int_array_new_api(int *x, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);
int cuda_get_device();
void cuda_free_host(float *x_cpu);
void cuda_free(float *x_gpu);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
cudaStream_t get_cuda_stream();
//cudaStream_t get_cuda_memcpy_stream();
int get_number_of_blocks(int array_size, int block_size);
int get_gpu_compute_capability(int i, char *device_name);
void show_cuda_cudnn_info();

cudaStream_t switch_stream(int i);
void wait_stream(int i);
void reset_wait_stream_events();

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
enum {cudnn_fastest, cudnn_smallest, cudnn_specify};

/// Use @ref CHECK_CUDNN() instead.
void cudnn_check_error_extended(cudnnStatus_t status, const char * const filename, const char * const function, const int line);

/// Macro to quickly check if a CUDNN error has taken place.  Only calls the CUDNN error function if a problem is detected.
#define CHECK_CUDNN(X)														\
{																			\
	const auto STATUS = X;													\
	if (STATUS != CUDNN_STATUS_SUCCESS)										\
	{																		\
		cudnn_check_error_extended(STATUS, __FILE__, __func__, __LINE__);	\
	}																		\
}

#endif

#endif // DARKNET_GPU
