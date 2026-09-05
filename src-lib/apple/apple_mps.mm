#include "darknet_internal.hpp"
#include "darknet_layers.hpp"
#include "apple_mps.hpp"
#include "apple_backend.hpp"
#include "apple_training.hpp"
#include "apple_training_internal.hpp"
#include "apple_objc.hpp"
#include "metal_backend.hpp"
#include "convolutional_layer.hpp"
#include "gemm.hpp"
#include "im2col.hpp"
#include "weights.hpp"

#ifdef DARKNET_USE_MPS

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

@interface MpsBatchNormDataSource : NSObject <MPSCNNBatchNormalizationDataSource>
{
@public
	std::vector<float> gamma;
	std::vector<float> beta;
	std::vector<float> mean;
	std::vector<float> variance;
	float epsilon;
}
- (instancetype)initWithChannels:(NSUInteger)channels
	gamma:(const float *)gamma_in
	beta:(const float *)beta_in
	mean:(const float *)mean_in
	variance:(const float *)variance_in
	epsilon:(float)epsilon_in;
@end

@implementation MpsBatchNormDataSource
- (instancetype)initWithChannels:(NSUInteger)channels
	gamma:(const float *)gamma_in
	beta:(const float *)beta_in
	mean:(const float *)mean_in
	variance:(const float *)variance_in
	epsilon:(float)epsilon_in
{
	self = [super init];
	if (self)
	{
		gamma.assign(gamma_in, gamma_in + channels);
		beta.assign(beta_in, beta_in + channels);
		mean.assign(mean_in, mean_in + channels);
		variance.assign(variance_in, variance_in + channels);
		epsilon = epsilon_in;
	}
	return self;
}

- (NSUInteger)numberOfFeatureChannels
{
	return gamma.size();
}

- (float *)gamma
{
	return gamma.data();
}

- (float *)beta
{
	return beta.data();
}

- (float *)mean
{
	return mean.data();
}

- (float *)variance
{
	return variance.data();
}

- (float)epsilon
{
	return epsilon;
}

- (BOOL)load
{
	return YES;
}

- (void)purge
{
}

- (NSString *)label
{
	return @"darknet_batchnorm";
}

- (id)copyWithZone:(NSZone *)zone
{
	(void)zone;
	return self;
}
@end

@interface MpsConvolutionDataSource : NSObject <MPSCNNConvolutionDataSource>
{
@private
	MPSCNNConvolutionDescriptor *convolution_descriptor;
	std::vector<float> kernel_weights;
	std::vector<float> convolution_biases;
}
- (instancetype)initWithDescriptor:(MPSCNNConvolutionDescriptor *)descriptor
	weights:(const float *)weights
	weightCount:(NSUInteger)weight_count
	biasTerms:(const float *)bias_terms
	biasCount:(NSUInteger)bias_count;
@end

@implementation MpsConvolutionDataSource
- (instancetype)initWithDescriptor:(MPSCNNConvolutionDescriptor *)descriptor
	weights:(const float *)weights
	weightCount:(NSUInteger)weight_count
	biasTerms:(const float *)bias_terms
	biasCount:(NSUInteger)bias_count
{
	self = [super init];
	if (self)
	{
		if (!descriptor || !weights || weight_count == 0)
		{
			return nil;
		}
		convolution_descriptor = descriptor;
		kernel_weights.assign(weights, weights + weight_count);
		if (bias_terms && bias_count > 0)
		{
			convolution_biases.assign(bias_terms, bias_terms + bias_count);
		}
	}
	return self;
}

- (MPSDataType)dataType
{
	return MPSDataTypeFloat32;
}

- (MPSCNNConvolutionDescriptor *)descriptor
{
	return convolution_descriptor;
}

- (void *)weights
{
	return kernel_weights.data();
}

- (float *)biasTerms
{
	return convolution_biases.empty() ? nullptr : convolution_biases.data();
}

- (BOOL)load
{
	return YES;
}

- (void)purge
{
}

- (NSString *)label
{
	return @"darknet_convolution";
}

- (MPSCNNConvolutionWeightsLayout)weightsLayout
{
	return MPSCNNConvolutionWeightsLayoutOHWI;
}

- (MPSDataType)kernelWeightsDataType
{
	// Match the legacy initializer, which converted float weights to FP16.
	return MPSDataTypeFloat16;
}

- (id)copyWithZone:(NSZone *)zone
{
	(void)zone;
	return self;
}

- (instancetype)copyWithZone:(NSZone *)zone device:(id<MTLDevice>)device
{
	(void)zone;
	(void)device;
	return self;
}
@end

namespace
{
	MPSNNNeuronDescriptor *native_neuron_descriptor(ACTIVATION activation);
	using Darknet::Apple::Detail::ObjcStrong;
	using Darknet::Apple::Detail::objc_release_if_needed;
	void sync_mirrored_shared_buffers();

	struct MpsContext
	{
		__unsafe_unretained id<MTLDevice> device = nil;
		__unsafe_unretained id<MTLCommandQueue> queue = nil;
		bool ready = false;
		std::string init_reason = "Apple MPS initialization has not run yet";
	};

	struct MpsCommandBufferState
	{
		id<MTLCommandBuffer> last_buffer = nil;

		~MpsCommandBufferState()
		{
			if (last_buffer)
			{
				[last_buffer waitUntilCompleted];
				sync_mirrored_shared_buffers();
				objc_release_if_needed(last_buffer);
				last_buffer = nil;
			}
		}
	};

	struct MpsMatrixOpKey
	{
		bool transpose_a = false;
		bool transpose_b = false;
		NSUInteger rows = 0;
		NSUInteger cols = 0;
		NSUInteger inner = 0;
		NSUInteger batch_size = 0;
		float alpha = 0.0f;
		float beta = 0.0f;

		bool operator==(const MpsMatrixOpKey & other) const
		{
			return transpose_a == other.transpose_a &&
				transpose_b == other.transpose_b &&
				rows == other.rows &&
				cols == other.cols &&
				inner == other.inner &&
				batch_size == other.batch_size &&
				alpha == other.alpha &&
				beta == other.beta;
		}
	};

	struct MpsMatrixOpKeyHash
	{
		size_t operator()(const MpsMatrixOpKey & key) const noexcept
		{
			size_t value = std::hash<bool>{}(key.transpose_a);
				value ^= std::hash<bool>{}(key.transpose_b) << 1U;
				value ^= std::hash<NSUInteger>{}(key.rows) << 2U;
				value ^= std::hash<NSUInteger>{}(key.cols) << 3U;
				value ^= std::hash<NSUInteger>{}(key.inner) << 4U;
				value ^= std::hash<NSUInteger>{}(key.batch_size) << 5U;
				value ^= std::hash<float>{}(key.alpha) << 6U;
				value ^= std::hash<float>{}(key.beta) << 7U;
				return value;
			}
		};

	struct MpsMatrixDescriptorKey
	{
		NSUInteger rows = 0;
		NSUInteger cols = 0;
		NSUInteger matrices = 1;
		NSUInteger row_bytes = 0;
		NSUInteger matrix_bytes = 0;
		MPSDataType data_type = MPSDataTypeInvalid;

		bool operator==(const MpsMatrixDescriptorKey & other) const
		{
			return rows == other.rows &&
				cols == other.cols &&
				matrices == other.matrices &&
				row_bytes == other.row_bytes &&
				matrix_bytes == other.matrix_bytes &&
				data_type == other.data_type;
		}
	};

	struct MpsMatrixDescriptorKeyHash
	{
		size_t operator()(const MpsMatrixDescriptorKey & key) const noexcept
		{
			size_t value = std::hash<NSUInteger>{}(key.rows);
			value ^= std::hash<NSUInteger>{}(key.cols) << 1U;
			value ^= std::hash<NSUInteger>{}(key.matrices) << 2U;
			value ^= std::hash<NSUInteger>{}(key.row_bytes) << 3U;
			value ^= std::hash<NSUInteger>{}(key.matrix_bytes) << 4U;
			value ^= std::hash<int>{}(static_cast<int>(key.data_type)) << 5U;
			return value;
		}
	};

	MpsContext & get_mps_context()
	{
		static MpsContext ctx;
		static std::once_flag once;
		std::call_once(once, [&]()
		{
			@autoreleasepool
			{
				const auto & runtime = Darknet::Apple::runtime_info();
				ctx.device = (__bridge id<MTLDevice>)Darknet::Apple::runtime_metal_device_handle();
				if (!ctx.device)
				{
					ctx.init_reason = runtime.reason.empty() ? "canonical Metal runtime has no device" : runtime.reason;
					return;
				}
				if (!MPSSupportsMTLDevice(ctx.device))
				{
					ctx.init_reason = "MPSSupportsMTLDevice() returned false";
					return;
				}
				ctx.queue = (__bridge id<MTLCommandQueue>)Darknet::Apple::runtime_metal_command_queue_handle();
				ctx.ready = (ctx.queue != nil);
				if (!ctx.ready)
				{
					ctx.init_reason = runtime.reason.empty() ? "canonical Metal runtime has no command queue" : runtime.reason;
					return;
				}
				ctx.init_reason = "ok";
			}
		});
		return ctx;
	}

	MpsCommandBufferState & get_mps_command_buffer_state()
	{
		thread_local MpsCommandBufferState state;
		return state;
	}

	id<MTLCommandBuffer> create_mps_command_buffer(MpsContext & ctx)
	{
		return [ctx.queue commandBuffer];
	}

	void track_mps_command_buffer(id<MTLCommandBuffer> buffer)
	{
		if (!buffer)
		{
			return;
		}
		auto & state = get_mps_command_buffer_state();
		if (state.last_buffer == buffer)
		{
			return;
		}
		// Every inference submission uses the canonical serial queue. Retaining
		// only the newest command buffer is sufficient: completion of that buffer
		// also establishes completion of all earlier submissions on the queue.
		objc_release_if_needed(state.last_buffer);
#if !__has_feature(objc_arc)
		[buffer retain];
#endif
		state.last_buffer = buffer;
	}

	bool wait_mps_command_buffer_if_needed(std::string & reason)
	{
		auto & state = get_mps_command_buffer_state();
		if (!state.last_buffer)
		{
			return true;
		}
		id<MTLCommandBuffer> buffer = state.last_buffer;
		state.last_buffer = nil;
		[buffer waitUntilCompleted];
		const bool completed = buffer.status == MTLCommandBufferStatusCompleted;
		if (!completed)
		{
			std::string failure = "MPS command buffer execution failed";
			NSString *description = buffer.error.localizedDescription;
			if (description.UTF8String)
			{
				failure += ": ";
				failure += description.UTF8String;
			}
			reason = "MPS command buffer execution failed";
			sync_mirrored_shared_buffers();
			objc_release_if_needed(buffer);
			darknet_fatal_error(DARKNET_LOC, "%s", failure.c_str());
		}
		sync_mirrored_shared_buffers();
		objc_release_if_needed(buffer);
		return true;
	}

	bool wait_mps_command_buffer_if_needed()
	{
		std::string reason;
		return wait_mps_command_buffer_if_needed(reason);
	}

	std::unordered_map<MpsMatrixOpKey, ObjcStrong, MpsMatrixOpKeyHash> & get_matrix_op_cache()
	{
		static std::unordered_map<MpsMatrixOpKey, ObjcStrong, MpsMatrixOpKeyHash> cache;
		return cache;
	}

	std::unordered_map<MpsMatrixDescriptorKey, ObjcStrong, MpsMatrixDescriptorKeyHash> & get_matrix_descriptor_cache()
	{
		static std::unordered_map<MpsMatrixDescriptorKey, ObjcStrong, MpsMatrixDescriptorKeyHash> cache;
		return cache;
	}

	std::mutex & get_matrix_op_mutex()
	{
		static std::mutex mutex;
		return mutex;
	}

	std::mutex & get_matrix_descriptor_mutex()
	{
		static std::mutex mutex;
		return mutex;
	}

	void sync_mirrored_shared_buffers()
	{
		Darknet::Apple::TrainingDetail::sync_all_training_shared_buffers_to_host();
	}

	struct MpsCallBuffer
	{
		id<MTLBuffer> buffer = nil;
		bool aliases_host = false;

		bool initialize(id<MTLDevice> device, void *host, const size_t bytes, const bool allow_alias)
		{
			if (!device || !host || bytes == 0)
			{
				return false;
			}

			if (allow_alias && Darknet::Apple::Detail::no_copy_buffer_eligible(host, bytes))
			{
				buffer = [device newBufferWithBytesNoCopy:host
					length:bytes
					options:MTLResourceStorageModeShared
					deallocator:nil];
				aliases_host = buffer != nil;
			}

			if (!buffer)
			{
				buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
				if (buffer)
				{
					std::memcpy(buffer.contents, host, bytes);
				}
			}
			return buffer != nil;
		}

		void copy_to_host(void *host, const size_t bytes) const
		{
			if (!aliases_host)
			{
				std::memcpy(host, buffer.contents, bytes);
			}
		}
	};

	MPSMatrixMultiplication * get_matrix_op(MpsContext & ctx, bool transpose_a, bool transpose_b,
		NSUInteger rows, NSUInteger cols, NSUInteger inner, NSUInteger batch_size, float alpha, float beta)
	{
		if (!ctx.ready)
		{
			return nil;
		}

		const MpsMatrixOpKey key{transpose_a, transpose_b, rows, cols, inner, batch_size, alpha, beta};
		{
			std::scoped_lock lock(get_matrix_op_mutex());
			auto & cache = get_matrix_op_cache();
			auto it = cache.find(key);
			if (it != cache.end())
			{
				return (MPSMatrixMultiplication *)it->second.get();
			}
		}

		MPSMatrixMultiplication *op = [[MPSMatrixMultiplication alloc]
			initWithDevice:ctx.device
			transposeLeft:transpose_a
			transposeRight:transpose_b
			resultRows:rows
			resultColumns:cols
			interiorColumns:inner
			alpha:static_cast<double>(alpha)
			beta:static_cast<double>(beta)];
		if (!op)
		{
			return nil;
		}
		if (batch_size > 0)
		{
			op.batchStart = 0;
			op.batchSize = batch_size;
		}

		std::scoped_lock lock(get_matrix_op_mutex());
		auto & cache = get_matrix_op_cache();
		auto [it, inserted] = cache.emplace(key, ObjcStrong(op));
		objc_release_if_needed(op);
		if (!inserted)
		{
			return (MPSMatrixMultiplication *)it->second.get();
		}
		return op;
	}

	MPSMatrixDescriptor * get_matrix_descriptor(NSUInteger rows,
		NSUInteger cols,
		NSUInteger matrices,
		NSUInteger row_bytes,
		NSUInteger matrix_bytes,
		MPSDataType data_type)
	{
		const MpsMatrixDescriptorKey key{rows, cols, matrices, row_bytes, matrix_bytes, data_type};
		{
			std::scoped_lock lock(get_matrix_descriptor_mutex());
			auto & cache = get_matrix_descriptor_cache();
			auto it = cache.find(key);
			if (it != cache.end())
			{
				return (MPSMatrixDescriptor *)it->second.get();
			}
		}

		MPSMatrixDescriptor *desc = nil;
		if (matrices <= 1)
		{
			desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
				columns:cols
				rowBytes:row_bytes
				dataType:data_type];
		}
		else
		{
			desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
				columns:cols
				matrices:matrices
				rowBytes:row_bytes
				matrixBytes:matrix_bytes
				dataType:data_type];
		}
		if (!desc)
		{
			return nil;
		}

		std::scoped_lock lock(get_matrix_descriptor_mutex());
		auto & cache = get_matrix_descriptor_cache();
		auto [it, inserted] = cache.emplace(key, ObjcStrong(desc));
		if (!inserted)
		{
			return (MPSMatrixDescriptor *)it->second.get();
		}
		return desc;
	}

	bool encode_matrix_multiply(MpsContext & ctx,
		id<MTLCommandBuffer> command_buffer,
		bool transpose_a, bool transpose_b,
		int M, int N, int K,
		float alpha, float beta,
		id<MTLBuffer> buffer_a, int lda,
		id<MTLBuffer> buffer_b, int ldb,
		id<MTLBuffer> buffer_c, int ldc)
	{
		if (!ctx.ready || !command_buffer || !buffer_a || !buffer_b || !buffer_c || M <= 0 || N <= 0 || K <= 0)
		{
			return false;
		}

		const NSUInteger a_rows = (transpose_a ? K : M);
		const NSUInteger a_cols = (transpose_a ? M : K);
		const NSUInteger b_rows = (transpose_b ? N : K);
		const NSUInteger b_cols = (transpose_b ? K : N);
		const NSUInteger c_rows = static_cast<NSUInteger>(M);
		const NSUInteger c_cols = static_cast<NSUInteger>(N);

		const NSUInteger a_row_bytes = static_cast<NSUInteger>(lda) * sizeof(float);
		const NSUInteger b_row_bytes = static_cast<NSUInteger>(ldb) * sizeof(float);
		const NSUInteger c_row_bytes = static_cast<NSUInteger>(ldc) * sizeof(float);

		@autoreleasepool
		{
			MPSMatrixDescriptor *desc_a = get_matrix_descriptor(a_rows, a_cols, 1, a_row_bytes, a_rows * a_row_bytes, MPSDataTypeFloat32);
			MPSMatrixDescriptor *desc_b = get_matrix_descriptor(b_rows, b_cols, 1, b_row_bytes, b_rows * b_row_bytes, MPSDataTypeFloat32);
			MPSMatrixDescriptor *desc_c = get_matrix_descriptor(c_rows, c_cols, 1, c_row_bytes, c_rows * c_row_bytes, MPSDataTypeFloat32);
			if (!desc_a || !desc_b || !desc_c)
			{
				return false;
			}

			MPSMatrix *matrix_a = [[MPSMatrix alloc] initWithBuffer:buffer_a offset:0 descriptor:desc_a];
			MPSMatrix *matrix_b = [[MPSMatrix alloc] initWithBuffer:buffer_b offset:0 descriptor:desc_b];
			MPSMatrix *matrix_c = [[MPSMatrix alloc] initWithBuffer:buffer_c offset:0 descriptor:desc_c];
			if (!matrix_a || !matrix_b || !matrix_c)
			{
				objc_release_if_needed(matrix_a);
				objc_release_if_needed(matrix_b);
				objc_release_if_needed(matrix_c);
				return false;
			}

			MPSMatrixMultiplication *op = get_matrix_op(ctx, transpose_a, transpose_b, c_rows, c_cols,
				static_cast<NSUInteger>(K), 0, alpha, beta);
			if (!op)
			{
				objc_release_if_needed(matrix_a);
				objc_release_if_needed(matrix_b);
				objc_release_if_needed(matrix_c);
				return false;
			}

			[op encodeToCommandBuffer:command_buffer leftMatrix:matrix_a rightMatrix:matrix_b resultMatrix:matrix_c];
			objc_release_if_needed(matrix_a);
			objc_release_if_needed(matrix_b);
			objc_release_if_needed(matrix_c);
			return true;
		}
	}

	struct MpsConvCache
	{
		MPSCNNConvolution *conv = nil;
		MpsConvolutionDataSource *data_source = nil;
		int size = 0;
		int c = 0;
		int n = 0;
		int groups = 0;
		int stride_x = 0;
		int stride_y = 0;
		int dilation = 0;
		int pad = 0;
		bool batch_normalize = false;
		ACTIVATION activation = LINEAR;
		bool activation_fused = false;
		uint64_t parameter_generation = 0;
		MPSImage *input_image = nil;
		MPSImage *output_image = nil;
		int input_w = 0;
		int input_h = 0;
		int input_c = 0;
		int input_batch = 0;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsPoolCache
	{
		MPSCNNPooling *pool = nil;
		bool avgpool = false;
		int size = 0;
		int stride_x = 0;
		int stride_y = 0;
		int pad = 0;
		MPSImage *input_image = nil;
		MPSImage *output_image = nil;
		int input_w = 0;
		int input_h = 0;
		int input_c = 0;
		int input_batch = 0;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsGlobalAvgPoolCache
	{
		MPSCNNPoolingAverage *pool = nil;
		int kernel_w = 0;
		int kernel_h = 0;
		int stride_x = 0;
		int stride_y = 0;
		MPSImage *input_image = nil;
		MPSImage *output_image = nil;
		int input_w = 0;
		int input_h = 0;
		int input_c = 0;
		int input_batch = 0;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsConnectedCache
	{
		MPSCNNBatchNormalization *batchnorm = nil;
		MpsBatchNormDataSource *batchnorm_source = nil;
		ACTIVATION activation = LINEAR;
		bool activation_fused = false;
		bool batch_normalize = false;
		int outputs = 0;
		uint64_t parameter_generation = 0;
		MPSImage *intermediate_image = nil;
		MPSImage *output_image = nil;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsAddCache
	{
		MPSCNNAdd *add = nil;
		ACTIVATION activation = LINEAR;
		MPSImage *output_image = nil;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsRouteCache
	{
		MPSImage *output_image = nil;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsUpsampleCache
	{
		MPSImage *input_image = nil;
		MPSImage *output_image = nil;
		int input_w = 0;
		int input_h = 0;
		int input_c = 0;
		int input_batch = 0;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
		int stride = 0;
		float scale = 0.0f;
	};

	struct MpsReorgCache
	{
		MPSImage *input_image = nil;
		MPSImage *output_image = nil;
		int input_w = 0;
		int input_h = 0;
		int input_c = 0;
		int input_batch = 0;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
		int stride = 0;
	};

	struct MpsSoftmaxCache
	{
		MetalBufferOwner input_buffer;
		MetalBufferOwner output_buffer;
		size_t bytes = 0;
	};

	struct MpsInferenceContext
	{
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConvCache>> convolution;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsPoolCache>> pooling;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsGlobalAvgPoolCache>> global_avgpool;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConnectedCache>> connected;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsAddCache>> add;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsRouteCache>> route;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsUpsampleCache>> upsample;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsReorgCache>> reorg;
		std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsSoftmaxCache>> softmax;
		std::unordered_set<const Darknet::Layer *> deferred;
	};

	MpsInferenceContext & get_mps_inference_context()
	{
		thread_local MpsInferenceContext context;
		return context;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConvCache>> & get_mps_conv_cache()
	{
		return get_mps_inference_context().convolution;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsPoolCache>> & get_mps_pool_cache()
	{
		return get_mps_inference_context().pooling;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsGlobalAvgPoolCache>> & get_mps_global_avgpool_cache()
	{
		return get_mps_inference_context().global_avgpool;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConnectedCache>> & get_mps_connected_cache()
	{
		return get_mps_inference_context().connected;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsAddCache>> & get_mps_add_cache()
	{
		return get_mps_inference_context().add;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsRouteCache>> & get_mps_route_cache()
	{
		return get_mps_inference_context().route;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsUpsampleCache>> & get_mps_upsample_cache()
	{
		return get_mps_inference_context().upsample;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsReorgCache>> & get_mps_reorg_cache()
	{
		return get_mps_inference_context().reorg;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsSoftmaxCache>> & get_mps_softmax_cache()
	{
		return get_mps_inference_context().softmax;
	}

	/// Tracks layers with deferred readback to reduce CPU syncs.
	std::unordered_set<const Darknet::Layer *> & get_mps_deferred_layers()
	{
		return get_mps_inference_context().deferred;
	}

	void reset_conv_images(MpsConvCache & cache);
	void reset_pool_images(MpsPoolCache & cache);
	void reset_global_avgpool_images(MpsGlobalAvgPoolCache & cache);
	void reset_connected_images(MpsConnectedCache & cache);
	void reset_add_images(MpsAddCache & cache);

	std::atomic<uint64_t> & parameter_generation()
	{
		static std::atomic<uint64_t> generation{1};
		return generation;
	}

	uint64_t current_parameter_generation()
	{
		return parameter_generation().load(std::memory_order_acquire);
	}

	void advance_parameter_generation()
	{
		parameter_generation().fetch_add(1, std::memory_order_acq_rel);
	}

	bool cache_matches_layer(const MpsConvCache & cache, const Darknet::Layer & l)
	{
		return cache.conv &&
			cache.size == l.size &&
			cache.c == l.c &&
			cache.n == l.n &&
			cache.groups == l.groups &&
			cache.stride_x == l.stride_x &&
			cache.stride_y == l.stride_y &&
			cache.dilation == l.dilation &&
			cache.pad == l.pad &&
			cache.batch_normalize == (l.batch_normalize != 0) &&
			cache.activation == l.activation &&
			cache.parameter_generation == current_parameter_generation();
	}

	void reset_conv_images(MpsConvCache & cache)
	{
		cache.input_image = nil;
		cache.output_image = nil;
		cache.input_w = 0;
		cache.input_h = 0;
		cache.input_c = 0;
		cache.input_batch = 0;
		cache.output_w = 0;
		cache.output_h = 0;
		cache.output_c = 0;
		cache.output_batch = 0;
	}

	void reset_pool_images(MpsPoolCache & cache)
	{
		cache.input_image = nil;
		cache.output_image = nil;
		cache.input_w = 0;
		cache.input_h = 0;
		cache.input_c = 0;
		cache.input_batch = 0;
		cache.output_w = 0;
		cache.output_h = 0;
		cache.output_c = 0;
		cache.output_batch = 0;
	}

	void reset_global_avgpool_images(MpsGlobalAvgPoolCache & cache)
	{
		cache.input_image = nil;
		cache.output_image = nil;
		cache.input_w = 0;
		cache.input_h = 0;
		cache.input_c = 0;
		cache.input_batch = 0;
		cache.output_w = 0;
		cache.output_h = 0;
		cache.output_c = 0;
		cache.output_batch = 0;
	}

	void reset_connected_images(MpsConnectedCache & cache)
	{
		cache.intermediate_image = nil;
		cache.output_image = nil;
		cache.output_w = 0;
		cache.output_h = 0;
		cache.output_c = 0;
		cache.output_batch = 0;
	}

	void reset_add_images(MpsAddCache & cache)
	{
		cache.output_image = nil;
		cache.output_w = 0;
		cache.output_h = 0;
		cache.output_c = 0;
		cache.output_batch = 0;
	}

	void invalidate_layer_cache_impl(const Darknet::Layer *layer)
	{
		if (!layer)
		{
			return;
		}

		get_mps_conv_cache().erase(layer);
		get_mps_pool_cache().erase(layer);
		get_mps_global_avgpool_cache().erase(layer);
		get_mps_connected_cache().erase(layer);
		get_mps_add_cache().erase(layer);
		get_mps_route_cache().erase(layer);
		get_mps_upsample_cache().erase(layer);
		get_mps_reorg_cache().erase(layer);
		get_mps_softmax_cache().erase(layer);
		get_mps_deferred_layers().erase(layer);
	}

	bool cache_matches_add(const MpsAddCache & cache, const Darknet::Layer & l)
	{
		return cache.add &&
			cache.activation == l.activation;
	}

	bool ensure_add_output_image(MpsAddCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const int out_c = l.out_c;
		const int batch = l.batch;

		if (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:static_cast<NSUInteger>(out_w)
				height:static_cast<NSUInteger>(out_h)
				featureChannels:static_cast<NSUInteger>(out_c)
				numberOfImages:static_cast<NSUInteger>(batch)
				usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		return cache.output_image != nil;
	}

	bool build_add_cache(MpsAddCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		cache.add = [[MPSCNNAdd alloc] initWithDevice:device];
		cache.activation = l.activation;
		reset_add_images(cache);
		return cache.add != nil;
	}

	bool ensure_route_output_image(MpsRouteCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const int out_c = l.out_c;
		const int batch = l.batch;

		if (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:static_cast<NSUInteger>(out_w)
				height:static_cast<NSUInteger>(out_h)
				featureChannels:static_cast<NSUInteger>(out_c)
				numberOfImages:static_cast<NSUInteger>(batch)
				usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		return cache.output_image != nil;
	}

	bool ensure_upsample_images(MpsUpsampleCache & cache, const Darknet::Layer & l, id<MTLDevice> device, bool need_input)
	{
		if (!device)
		{
			return false;
		}

		const int batch = l.batch;
		const int in_w = l.w;
		const int in_h = l.h;
		const int in_c = l.c;
		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const int out_c = l.out_c;

		if (need_input)
		{
			if (!cache.input_image ||
				cache.input_w != in_w ||
				cache.input_h != in_h ||
				cache.input_c != in_c ||
				cache.input_batch != batch)
			{
				MPSImageDescriptor *input_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
					width:static_cast<NSUInteger>(in_w)
					height:static_cast<NSUInteger>(in_h)
					featureChannels:static_cast<NSUInteger>(in_c)
					numberOfImages:static_cast<NSUInteger>(batch)
					usage:MTLTextureUsageShaderRead];
				cache.input_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:input_desc];
				cache.input_w = in_w;
				cache.input_h = in_h;
				cache.input_c = in_c;
				cache.input_batch = batch;
			}
		}
		else
		{
			cache.input_image = nil;
			cache.input_w = 0;
			cache.input_h = 0;
			cache.input_c = 0;
			cache.input_batch = 0;
		}

		if (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:static_cast<NSUInteger>(out_w)
				height:static_cast<NSUInteger>(out_h)
				featureChannels:static_cast<NSUInteger>(out_c)
				numberOfImages:static_cast<NSUInteger>(batch)
				usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		cache.stride = l.stride;
		cache.scale = l.scale;

		return (need_input ? (cache.input_image != nil) : true) && cache.output_image;
	}

	bool ensure_reorg_images(MpsReorgCache & cache, const Darknet::Layer & l, id<MTLDevice> device, bool need_input)
	{
		if (!device)
		{
			return false;
		}

		const int in_w = l.w;
		const int in_h = l.h;
		const int in_c = l.c;
		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const int out_c = l.out_c;
		const int batch = l.batch;

		if (need_input)
		{
			if (!cache.input_image ||
				cache.input_w != in_w ||
				cache.input_h != in_h ||
				cache.input_c != in_c ||
				cache.input_batch != batch)
			{
				MPSImageDescriptor *input_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
					width:in_w
					height:in_h
					featureChannels:in_c
					numberOfImages:batch
					usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
				cache.input_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:input_desc];
				cache.input_w = in_w;
				cache.input_h = in_h;
				cache.input_c = in_c;
				cache.input_batch = batch;
			}
		}
		else
		{
			cache.input_image = nil;
			cache.input_w = 0;
			cache.input_h = 0;
			cache.input_c = 0;
			cache.input_batch = 0;
		}

		if (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:out_w
				height:out_h
				featureChannels:out_c
				numberOfImages:batch
				usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		cache.stride = l.stride;

		return (need_input ? (cache.input_image != nil) : true) && cache.output_image;
	}

	bool encode_mps_activation(id<MTLCommandBuffer> command_buffer, MPSImage *image, ACTIVATION activation, std::string & reason)
	{
		if (!command_buffer || !image)
		{
			reason = "invalid activation inputs";
			return false;
		}

		std::string kernel_name;
		uint32_t channels = 0;
		uint32_t slices_per_image = 0;
		uint32_t use_max_val = 0;
		MetalDispatchBytes bytes[3] = {};
		size_t bytes_count = 0;
		switch (activation)
		{
			case RELU: kernel_name = "relu_kernel"; break;
			case LEAKY: kernel_name = "leaky_kernel"; break;
			case SWISH: kernel_name = "swish_kernel"; break;
			case MISH: kernel_name = "mish_kernel"; break;
			case HARD_MISH: kernel_name = "hard_mish_kernel"; break;
			case LOGISTIC: kernel_name = "logistic_kernel"; break;
			case LOGGY: kernel_name = "loggy_kernel"; break;
			case TANH: kernel_name = "tanh_kernel"; break;
			case RELU6: kernel_name = "relu6_kernel"; break;
			case ELU: kernel_name = "elu_kernel"; break;
			case SELU: kernel_name = "selu_kernel"; break;
			case GELU: kernel_name = "gelu_kernel"; break;
			case RELIE: kernel_name = "relie_kernel"; break;
			case RAMP: kernel_name = "ramp_kernel"; break;
			case HARDTAN: kernel_name = "hardtan_kernel"; break;
			case LHTAN: kernel_name = "lhtan_kernel"; break;
			case PLSE: kernel_name = "plse_kernel"; break;
			case STAIR: kernel_name = "stair_kernel"; break;
			case REVLEAKY: kernel_name = "revleaky_kernel"; break;
			case NORM_CHAN:
				kernel_name = "normalize_channels_kernel";
				channels = static_cast<uint32_t>(image.featureChannels);
				slices_per_image = (channels + 3u) / 4u;
				bytes[0] = { &channels, sizeof(channels) };
				bytes[1] = { &slices_per_image, sizeof(slices_per_image) };
				bytes_count = 2;
				break;
			case NORM_CHAN_SOFTMAX:
				kernel_name = "normalize_channels_softmax_kernel";
				use_max_val = 0;
				channels = static_cast<uint32_t>(image.featureChannels);
				slices_per_image = (channels + 3u) / 4u;
				bytes[0] = { &channels, sizeof(channels) };
				bytes[1] = { &slices_per_image, sizeof(slices_per_image) };
				bytes[2] = { &use_max_val, sizeof(use_max_val) };
				bytes_count = 3;
				break;
			case NORM_CHAN_SOFTMAX_MAXVAL:
				kernel_name = "normalize_channels_softmax_kernel";
				use_max_val = 1;
				channels = static_cast<uint32_t>(image.featureChannels);
				slices_per_image = (channels + 3u) / 4u;
				bytes[0] = { &channels, sizeof(channels) };
				bytes[1] = { &slices_per_image, sizeof(slices_per_image) };
				bytes[2] = { &use_max_val, sizeof(use_max_val) };
				bytes_count = 3;
				break;
			default:
				reason = "activation not supported";
				return false;
		}

		id<MTLTexture> texture = image.texture;
		if (!texture)
		{
			reason = "missing activation texture";
			return false;
		}

		if ((bytes_count > 0) && (channels == 0 || slices_per_image == 0))
		{
			reason = "activation not supported";
			return false;
		}

		MetalDispatchTexture textures[] = { { (__bridge void *)texture } };
		if (!metal_dispatch_texture_kernel(kernel_name,
				(__bridge void *)texture,
				8, 8,
				textures, 1,
				nullptr, 0,
				bytes_count > 0 ? bytes : nullptr, bytes_count,
				(__bridge void *)command_buffer))
		{
			reason = "failed to dispatch activation kernel";
			return false;
		}
		return true;
	}

	bool encode_mps_upsample(id<MTLCommandBuffer> command_buffer, MPSImage *input_image, MPSImage *output_image,
		int stride, float scale, std::string & reason)
	{
		if (!command_buffer || !input_image || !output_image)
		{
			reason = "invalid upsample inputs";
			return false;
		}

		if (stride <= 0)
		{
			reason = "invalid stride";
			return false;
		}

		id<MTLTexture> in_tex = input_image.texture;
		id<MTLTexture> out_tex = output_image.texture;
		if (!in_tex || !out_tex)
		{
			reason = "missing upsample texture";
			return false;
		}

		const uint32_t stride_u = static_cast<uint32_t>(stride);
		MetalDispatchTexture textures[] = {
			{ (__bridge void *)in_tex },
			{ (__bridge void *)out_tex }
		};
		MetalDispatchBytes bytes[] = {
			{ &stride_u, sizeof(stride_u) },
			{ &scale, sizeof(scale) }
		};
		if (!metal_dispatch_texture_kernel("upsample_kernel",
				(__bridge void *)out_tex,
				8, 8,
				textures, 2,
				nullptr, 0,
				bytes, 2,
				(__bridge void *)command_buffer))
		{
			reason = "failed to dispatch upsample kernel";
			return false;
		}

		return true;
	}

	bool encode_metal_pool(id<MTLCommandBuffer> command_buffer, MPSImage *input_image, MPSImage *output_image,
		const Darknet::Layer & l, bool avgpool, std::string & reason)
	{
		if (!command_buffer || !input_image || !output_image)
		{
			reason = "invalid pool inputs";
			return false;
		}

		id<MTLTexture> in_tex = input_image.texture;
		id<MTLTexture> out_tex = output_image.texture;
		if (!in_tex || !out_tex)
		{
			reason = "missing pool texture";
			return false;
		}

		const uint32_t channels = static_cast<uint32_t>(l.c);
		const uint32_t size = static_cast<uint32_t>(l.size);
		const uint32_t stride_x = static_cast<uint32_t>(l.stride_x);
		const uint32_t stride_y = static_cast<uint32_t>(l.stride_y);
		const int32_t pad = static_cast<int32_t>(l.pad / 2);
		const std::string & kernel_name = avgpool ? "avgpool_kernel" : "maxpool_kernel";
		MetalDispatchTexture textures[] = {
			{ (__bridge void *)in_tex },
			{ (__bridge void *)out_tex }
		};
		MetalDispatchBytes bytes[] = {
			{ &channels, sizeof(channels) },
			{ &size, sizeof(size) },
			{ &stride_x, sizeof(stride_x) },
			{ &stride_y, sizeof(stride_y) },
			{ &pad, sizeof(pad) }
		};
		if (!metal_dispatch_texture_kernel(kernel_name,
				(__bridge void *)out_tex,
				8, 8,
				textures, 2,
				nullptr, 0,
				bytes, 5,
				(__bridge void *)command_buffer))
		{
			reason = "failed to dispatch pool kernel";
			return false;
		}
		return true;
	}

	void pack_weights_ohwi(const Darknet::Layer & l, std::vector<float> & packed)
	{
		const int in_channels = l.c / l.groups;
		const int kernel_area = l.size * l.size;
		const size_t total = static_cast<size_t>(l.n) * kernel_area * in_channels;
		packed.assign(total, 0.0f);

		for (int oc = 0; oc < l.n; ++oc)
		{
			for (int ic = 0; ic < in_channels; ++ic)
			{
				for (int ky = 0; ky < l.size; ++ky)
				{
					for (int kx = 0; kx < l.size; ++kx)
					{
						const size_t src = static_cast<size_t>(((oc * in_channels + ic) * l.size + ky) * l.size + kx);
						const size_t dst = static_cast<size_t>(((oc * l.size + ky) * l.size + kx) * in_channels + ic);
						packed[dst] = l.weights[src];
					}
				}
			}
		}
	}


	MPSNNNeuronDescriptor *native_neuron_descriptor(ACTIVATION activation)
	{
		switch (activation)
		{
			case RELU:
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU a:0.0f];
			case LEAKY:
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU a:0.1f];
			case RELIE:
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU a:0.01f];
			case LOGISTIC:
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeSigmoid];
			case LOGGY:
				// 2 * sigmoid(x) - 1 is exactly tanh(x / 2).
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeTanH a:1.0f b:0.5f];
			case TANH:
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeTanH a:1.0f b:1.0f];
			case RELU6:
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLUN a:0.0f b:6.0f];
			case ELU:
				return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeELU a:1.0f];
			case LINEAR:
				return nil;
			default:
				return nil;
		}
	}

	bool mps_activation_supported(ACTIVATION activation)
	{
		switch (activation)
		{
			case LINEAR:
			case RELU:
			case LEAKY:
			case SWISH:
			case MISH:
			case HARD_MISH:
			case LOGISTIC:
			case LOGGY:
			case TANH:
			case RELU6:
			case ELU:
			case SELU:
			case GELU:
			case RELIE:
			case RAMP:
			case HARDTAN:
			case LHTAN:
			case PLSE:
			case STAIR:
			case REVLEAKY:
			case NORM_CHAN:
			case NORM_CHAN_SOFTMAX:
			case NORM_CHAN_SOFTMAX_MAXVAL:
				return true;
			default:
				return false;
		}
	}

	bool build_conv_cache(MpsConvCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device || !l.weights || l.size <= 0 || l.c <= 0 || l.n <= 0 || l.dilation <= 0)
		{
			return false;
		}

		if (l.groups <= 0 || (l.c % l.groups) != 0 || (l.n % l.groups) != 0)
		{
			return false;
		}

		const bool depthwise = (l.groups == l.c && l.n == l.c);
		const int channels_per_group = l.c / l.groups;
		const int outputs_per_group = l.n / l.groups;
		if (l.groups > 1 && !depthwise &&
			((channels_per_group % 4) != 0 || (outputs_per_group % 4) != 0))
		{
			return false;
		}

		std::vector<float> packed_weights;
		pack_weights_ohwi(l, packed_weights);
		std::vector<float> biases(static_cast<size_t>(l.n), 0.0f);
		if (!l.batch_normalize && l.biases)
		{
			for (int i = 0; i < l.n; ++i)
			{
				biases[i] = l.biases[i];
			}
		}

		MPSCNNConvolutionDescriptor *desc = nil;
		if (depthwise)
		{
			desc = [MPSCNNDepthWiseConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:l.size
				kernelHeight:l.size
				inputFeatureChannels:l.c
				outputFeatureChannels:l.n];
		}
		else
		{
			desc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:l.size
				kernelHeight:l.size
				inputFeatureChannels:l.c
				outputFeatureChannels:l.n];
			desc.groups = l.groups;
		}
		if (!desc)
		{
			return false;
		}

		desc.strideInPixelsX = l.stride_x;
		desc.strideInPixelsY = l.stride_y;
		desc.dilationRateX = l.dilation;
		desc.dilationRateY = l.dilation;

		if (l.batch_normalize)
		{
			if (!l.scales || !l.biases || !l.rolling_mean || !l.rolling_variance)
			{
				return false;
			}
			[desc setBatchNormalizationParametersForInferenceWithMean:l.rolling_mean
				variance:l.rolling_variance
				gamma:l.scales
				beta:l.biases
				epsilon:0.00001f];
		}

		MPSNNNeuronDescriptor *neuron = native_neuron_descriptor(l.activation);
		if (neuron)
		{
			desc.fusedNeuronDescriptor = neuron;
		}

		const NSUInteger weight_count = static_cast<NSUInteger>(packed_weights.size());
		cache.data_source = [[MpsConvolutionDataSource alloc]
			initWithDescriptor:desc
			weights:packed_weights.data()
			weightCount:weight_count
			biasTerms:biases.data()
			biasCount:static_cast<NSUInteger>(biases.size())];
		if (!cache.data_source)
		{
			return false;
		}

		cache.conv = [[MPSCNNConvolution alloc] initWithDevice:device weights:cache.data_source];
		if (!cache.conv)
		{
			return false;
		}

		cache.conv.edgeMode = MPSImageEdgeModeZero;
		const int offset = (l.size / 2) - l.pad;
		const MPSOffset mps_offset = { offset, offset, 0 };
		cache.conv.offset = mps_offset;

		cache.size = l.size;
		cache.c = l.c;
		cache.n = l.n;
		cache.groups = l.groups;
		cache.stride_x = l.stride_x;
		cache.stride_y = l.stride_y;
		cache.dilation = l.dilation;
		cache.pad = l.pad;
		cache.batch_normalize = (l.batch_normalize != 0);
		cache.activation = l.activation;
		cache.activation_fused = (neuron != nil);
		cache.parameter_generation = current_parameter_generation();
		reset_conv_images(cache);

		return true;
	}

	bool ensure_conv_images(MpsConvCache & cache, const Darknet::Layer & l, id<MTLDevice> device, bool need_input)
	{
		if (!device)
		{
			return false;
		}

		const int batch = l.batch;
		const int in_w = l.w;
		const int in_h = l.h;
		const int in_c = l.c;
		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const int out_c = l.n;
		const bool output_shape_changed = (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch);

		if (need_input)
		{
			if (!cache.input_image ||
				cache.input_w != in_w ||
				cache.input_h != in_h ||
				cache.input_c != in_c ||
				cache.input_batch != batch)
			{
				MPSImageDescriptor *input_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
					width:static_cast<NSUInteger>(in_w)
					height:static_cast<NSUInteger>(in_h)
					featureChannels:static_cast<NSUInteger>(in_c)
					numberOfImages:static_cast<NSUInteger>(batch)
					usage:MTLTextureUsageShaderRead];
				cache.input_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:input_desc];
				cache.input_w = in_w;
				cache.input_h = in_h;
				cache.input_c = in_c;
				cache.input_batch = batch;
			}
		}
		else
		{
			cache.input_image = nil;
			cache.input_w = 0;
			cache.input_h = 0;
			cache.input_c = 0;
			cache.input_batch = 0;
		}

		if (output_shape_changed)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:static_cast<NSUInteger>(out_w)
				height:static_cast<NSUInteger>(out_h)
				featureChannels:static_cast<NSUInteger>(out_c)
				numberOfImages:static_cast<NSUInteger>(batch)
				usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		return (need_input ? (cache.input_image != nil) : true) && cache.output_image;
	}

	bool cache_matches_pool(const MpsPoolCache & cache, const Darknet::Layer & l, bool avgpool)
	{
		return cache.pool &&
			cache.avgpool == avgpool &&
			cache.size == l.size &&
			cache.stride_x == l.stride_x &&
			cache.stride_y == l.stride_y &&
			cache.pad == l.pad;
	}

	bool build_pool_cache(MpsPoolCache & cache, const Darknet::Layer & l, id<MTLDevice> device, bool avgpool)
	{
		if (!device)
		{
			return false;
		}

		if (avgpool)
		{
			cache.pool = [[MPSCNNPoolingAverage alloc] initWithDevice:device
				kernelWidth:l.size
				kernelHeight:l.size
				strideInPixelsX:l.stride_x
				strideInPixelsY:l.stride_y];
		}
		else
		{
			cache.pool = [[MPSCNNPoolingMax alloc] initWithDevice:device
				kernelWidth:l.size
				kernelHeight:l.size
				strideInPixelsX:l.stride_x
				strideInPixelsY:l.stride_y];
		}

		if (!cache.pool)
		{
			return false;
		}

		cache.pool.edgeMode = MPSImageEdgeModeClamp;
		const int offset = (l.size / 2) - (l.pad / 2);
		const MPSOffset mps_offset = { offset, offset, 0 };
		cache.pool.offset = mps_offset;

		cache.avgpool = avgpool;
		cache.size = l.size;
		cache.stride_x = l.stride_x;
		cache.stride_y = l.stride_y;
		cache.pad = l.pad;
		reset_pool_images(cache);

		return true;
	}

	bool ensure_pool_images(MpsPoolCache & cache, const Darknet::Layer & l, id<MTLDevice> device, bool need_input)
	{
		if (!device)
		{
			return false;
		}

		const int batch = l.batch;
		const int in_w = l.w;
		const int in_h = l.h;
		const int in_c = l.c;
		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const int out_c = l.out_c;

		if (need_input)
		{
			if (!cache.input_image ||
				cache.input_w != in_w ||
				cache.input_h != in_h ||
				cache.input_c != in_c ||
				cache.input_batch != batch)
			{
				MPSImageDescriptor *input_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
					width:static_cast<NSUInteger>(in_w)
					height:static_cast<NSUInteger>(in_h)
					featureChannels:static_cast<NSUInteger>(in_c)
					numberOfImages:static_cast<NSUInteger>(batch)
					usage:MTLTextureUsageShaderRead];
				cache.input_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:input_desc];
				cache.input_w = in_w;
				cache.input_h = in_h;
				cache.input_c = in_c;
				cache.input_batch = batch;
			}
		}
		else
		{
			cache.input_image = nil;
			cache.input_w = 0;
			cache.input_h = 0;
			cache.input_c = 0;
			cache.input_batch = 0;
		}

		if (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:static_cast<NSUInteger>(out_w)
				height:static_cast<NSUInteger>(out_h)
				featureChannels:static_cast<NSUInteger>(out_c)
				numberOfImages:static_cast<NSUInteger>(batch)
				usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		return (need_input ? (cache.input_image != nil) : true) && cache.output_image;
	}

	bool cache_matches_global_avgpool(const MpsGlobalAvgPoolCache & cache, const Darknet::Layer & l)
	{
		return cache.pool &&
			cache.kernel_w == l.w &&
			cache.kernel_h == l.h &&
			cache.stride_x == l.w &&
			cache.stride_y == l.h;
	}

	bool build_global_avgpool_cache(MpsGlobalAvgPoolCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		if (l.w <= 0 || l.h <= 0)
		{
			return false;
		}

		cache.pool = [[MPSCNNPoolingAverage alloc] initWithDevice:device
			kernelWidth:l.w
			kernelHeight:l.h
			strideInPixelsX:l.w
			strideInPixelsY:l.h];

		if (!cache.pool)
		{
			return false;
		}

		cache.pool.edgeMode = MPSImageEdgeModeZero;
		const MPSOffset mps_offset = { l.w / 2, l.h / 2, 0 };
		cache.pool.offset = mps_offset;

		cache.kernel_w = l.w;
		cache.kernel_h = l.h;
		cache.stride_x = l.w;
		cache.stride_y = l.h;
		reset_global_avgpool_images(cache);

		return true;
	}

	bool ensure_global_avgpool_images(MpsGlobalAvgPoolCache & cache, const Darknet::Layer & l, id<MTLDevice> device, bool need_input)
	{
		if (!device)
		{
			return false;
		}

		const int batch = l.batch;
		const int in_w = l.w;
		const int in_h = l.h;
		const int in_c = l.c;
		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const int out_c = l.out_c;

		if (need_input)
		{
			if (!cache.input_image ||
				cache.input_w != in_w ||
				cache.input_h != in_h ||
				cache.input_c != in_c ||
				cache.input_batch != batch)
			{
				MPSImageDescriptor *input_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
					width:static_cast<NSUInteger>(in_w)
					height:static_cast<NSUInteger>(in_h)
					featureChannels:static_cast<NSUInteger>(in_c)
					numberOfImages:static_cast<NSUInteger>(batch)
					usage:MTLTextureUsageShaderRead];
				cache.input_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:input_desc];
				cache.input_w = in_w;
				cache.input_h = in_h;
				cache.input_c = in_c;
				cache.input_batch = batch;
			}
		}
		else
		{
			cache.input_image = nil;
			cache.input_w = 0;
			cache.input_h = 0;
			cache.input_c = 0;
			cache.input_batch = 0;
		}

		if (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:static_cast<NSUInteger>(out_w)
				height:static_cast<NSUInteger>(out_h)
				featureChannels:static_cast<NSUInteger>(out_c)
				numberOfImages:static_cast<NSUInteger>(batch)
				usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		return (need_input ? (cache.input_image != nil) : true) && cache.output_image;
	}

	bool cache_matches_connected(const MpsConnectedCache & cache, const Darknet::Layer & l)
	{
		return cache.outputs == l.outputs &&
			cache.batch_normalize == (l.batch_normalize != 0) &&
			cache.activation == l.activation &&
			cache.parameter_generation == current_parameter_generation();
	}

	bool build_connected_cache(MpsConnectedCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		cache.batchnorm = nil;
		cache.batchnorm_source = nil;
		cache.activation_fused = false;

		if (l.batch_normalize)
		{
			if (!l.scales || !l.biases || !l.rolling_mean || !l.rolling_variance)
			{
				return false;
			}
			cache.batchnorm_source = [[MpsBatchNormDataSource alloc] initWithChannels:static_cast<NSUInteger>(l.outputs)
				gamma:l.scales
				beta:l.biases
				mean:l.rolling_mean
				variance:l.rolling_variance
				epsilon:0.00001f];
			MPSNNNeuronDescriptor *neuron = native_neuron_descriptor(l.activation);
			cache.batchnorm = [[MPSCNNBatchNormalization alloc] initWithDevice:device
				dataSource:cache.batchnorm_source
				fusedNeuronDescriptor:neuron];
			if (!cache.batchnorm)
			{
				return false;
			}
			cache.activation_fused = (neuron != nil);
		}

		cache.outputs = l.outputs;
		cache.batch_normalize = (l.batch_normalize != 0);
		cache.activation = l.activation;
		cache.parameter_generation = current_parameter_generation();
		reset_connected_images(cache);

		return true;
	}

	bool ensure_connected_images(MpsConnectedCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		const int batch = l.batch;
		const int out_c = l.outputs;
		const int out_w = l.out_w;
		const int out_h = l.out_h;
		const bool output_shape_changed = (!cache.output_image ||
			cache.output_w != out_w ||
			cache.output_h != out_h ||
			cache.output_c != out_c ||
			cache.output_batch != batch);

		if (output_shape_changed)
		{
			MPSImageDescriptor *output_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
				width:static_cast<NSUInteger>(out_w)
				height:static_cast<NSUInteger>(out_h)
				featureChannels:static_cast<NSUInteger>(out_c)
				numberOfImages:static_cast<NSUInteger>(batch)
				usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
			cache.output_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:output_desc];
			cache.output_w = out_w;
			cache.output_h = out_h;
			cache.output_c = out_c;
			cache.output_batch = batch;
		}

		if (cache.batchnorm)
		{
			if (output_shape_changed || !cache.intermediate_image)
			{
				MPSImageDescriptor *intermediate_desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
					width:static_cast<NSUInteger>(out_w)
					height:static_cast<NSUInteger>(out_h)
					featureChannels:static_cast<NSUInteger>(out_c)
					numberOfImages:static_cast<NSUInteger>(batch)
					usage:(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)];
				cache.intermediate_image = [[MPSImage alloc] initWithDevice:device imageDescriptor:intermediate_desc];
			}
		}
		else
		{
			cache.intermediate_image = nil;
		}

		return cache.output_image != nil && (!cache.batchnorm || cache.intermediate_image != nil);
	}

	MPSImage *get_cached_output_image(const Darknet::Layer *layer, int w, int h, int c, int batch)
	{
		if (!layer)
		{
			return nil;
		}

		auto & conv_map = get_mps_conv_cache();
		auto conv_it = conv_map.find(layer);
		if (conv_it != conv_map.end() && conv_it->second)
		{
			const auto & cache = *conv_it->second;
			if (cache.output_image &&
				cache.output_w == w &&
				cache.output_h == h &&
				cache.output_c == c &&
				cache.output_batch == batch)
			{
				return cache.output_image;
			}
		}

		auto & pool_map = get_mps_pool_cache();
		auto pool_it = pool_map.find(layer);
		if (pool_it != pool_map.end() && pool_it->second)
		{
			const auto & cache = *pool_it->second;
			if (cache.output_image &&
				cache.output_w == w &&
				cache.output_h == h &&
				cache.output_c == c &&
				cache.output_batch == batch)
			{
				return cache.output_image;
			}
		}

		auto & global_pool_map = get_mps_global_avgpool_cache();
		auto global_pool_it = global_pool_map.find(layer);
		if (global_pool_it != global_pool_map.end() && global_pool_it->second)
		{
			const auto & cache = *global_pool_it->second;
			if (cache.output_image &&
				cache.output_w == w &&
				cache.output_h == h &&
				cache.output_c == c &&
				cache.output_batch == batch)
			{
				return cache.output_image;
			}
		}

		auto & add_map = get_mps_add_cache();
		auto add_it = add_map.find(layer);
		if (add_it != add_map.end() && add_it->second)
		{
			const auto & cache = *add_it->second;
			if (cache.output_image &&
				cache.output_w == w &&
				cache.output_h == h &&
				cache.output_c == c &&
				cache.output_batch == batch)
			{
				return cache.output_image;
			}
		}

	auto & route_map = get_mps_route_cache();
	auto route_it = route_map.find(layer);
	if (route_it != route_map.end() && route_it->second)
	{
			const auto & cache = *route_it->second;
			if (cache.output_image &&
				cache.output_w == w &&
				cache.output_h == h &&
				cache.output_c == c &&
				cache.output_batch == batch)
			{
			return cache.output_image;
		}
	}

	auto & upsample_map = get_mps_upsample_cache();
	auto upsample_it = upsample_map.find(layer);
	if (upsample_it != upsample_map.end() && upsample_it->second)
	{
		const auto & cache = *upsample_it->second;
		if (cache.output_image &&
			cache.output_w == w &&
			cache.output_h == h &&
			cache.output_c == c &&
			cache.output_batch == batch)
		{
			return cache.output_image;
		}
	}

	auto & reorg_map = get_mps_reorg_cache();
	auto reorg_it = reorg_map.find(layer);
	if (reorg_it != reorg_map.end() && reorg_it->second)
	{
		const auto & cache = *reorg_it->second;
		if (cache.output_image &&
			cache.output_w == w &&
			cache.output_h == h &&
			cache.output_c == c &&
			cache.output_batch == batch)
		{
			return cache.output_image;
		}
	}

	auto & connected_map = get_mps_connected_cache();
	auto connected_it = connected_map.find(layer);
	if (connected_it != connected_map.end() && connected_it->second)
	{
		const auto & cache = *connected_it->second;
		if (cache.output_image &&
			cache.output_w == w &&
			cache.output_h == h &&
			cache.output_c == c &&
			cache.output_batch == batch)
		{
			return cache.output_image;
		}
	}

	return nil;
}

	bool read_mps_output_image(MPSImage *image, int w, int h, int c, int batch, float *output)
	{
		if (!image || !output)
		{
			return false;
		}

		const NSUInteger out_channels = static_cast<NSUInteger>(c);
		const NSUInteger out_w = static_cast<NSUInteger>(w);
		const NSUInteger out_h = static_cast<NSUInteger>(h);
		const NSUInteger out_batch = static_cast<NSUInteger>(batch);
		const NSUInteger output_bytes_per_row = out_w * sizeof(float);
		const MTLRegion out_region = MTLRegionMake2D(0, 0, out_w, out_h);

		MPSImageReadWriteParams out_params = {};
		out_params.featureChannelOffset = 0;
		out_params.numberOfFeatureChannelsToReadWrite = out_channels;

		for (NSUInteger b = 0; b < out_batch; ++b)
		{
			float *output_ptr = output + b * (c * h * w);
			[image readBytes:output_ptr
				dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
				bytesPerRow:output_bytes_per_row
				region:out_region
				featureChannelInfo:out_params
				imageIndex:b];
		}

		return true;
	}
}

bool Darknet::Apple::mps_is_available()
{
	return get_mps_context().ready;
}

bool Darknet::Apple::mps_runtime_available(std::string & reason)
{
	auto & ctx = get_mps_context();
	reason = (ctx.ready ? "ok" : ctx.init_reason);
	return ctx.ready;
}

bool Darknet::Apple::mps_training_device_count_supported(const int ngpus, std::string & reason)
{
	if (ngpus == 1)
	{
		reason = "ok";
		return true;
	}
	reason = "Apple MPSGraph training supports exactly one GPU/device; received ngpus=" + std::to_string(ngpus);
	return false;
}

bool Darknet::Apple::mps_gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		return false;
	}

	if (M <= 0 || N <= 0)
	{
		return true;
	}
	if (!C || ldc < N)
	{
		return false;
	}
	if (K <= 0)
	{
		if (BETA != 1.0f)
		{
			for (int i = 0; i < M; ++i)
			{
				for (int j = 0; j < N; ++j)
				{
					C[i * ldc + j] *= BETA;
				}
			}
		}
		return true;
	}
	if (!A || !B || lda < (TA ? M : K) || ldb < (TB ? K : N))
	{
		return false;
	}

	const NSUInteger a_rows = (TA ? K : M);
	const NSUInteger b_rows = (TB ? N : K);
	const NSUInteger c_rows = static_cast<NSUInteger>(M);

	const NSUInteger a_row_bytes = static_cast<NSUInteger>(lda) * sizeof(float);
	const NSUInteger b_row_bytes = static_cast<NSUInteger>(ldb) * sizeof(float);
	const NSUInteger c_row_bytes = static_cast<NSUInteger>(ldc) * sizeof(float);

	const NSUInteger a_bytes = a_rows * a_row_bytes;
	const NSUInteger b_bytes = b_rows * b_row_bytes;
	const NSUInteger c_bytes = c_rows * c_row_bytes;

	@autoreleasepool
	{
		MpsCallBuffer buffer_a;
		MpsCallBuffer buffer_b;
		MpsCallBuffer buffer_c;
		if (!buffer_a.initialize(ctx.device, A, a_bytes, true) ||
			!buffer_b.initialize(ctx.device, B, b_bytes, true) ||
			!buffer_c.initialize(ctx.device, C, c_bytes, false))
		{
			return false;
		}

		id<MTLCommandBuffer> command_buffer = [ctx.queue commandBuffer];
		if (!command_buffer)
		{
			return false;
		}

		if (!encode_matrix_multiply(ctx, command_buffer, TA != 0, TB != 0, M, N, K, ALPHA, BETA,
				buffer_a.buffer, lda, buffer_b.buffer, ldb, buffer_c.buffer, ldc))
		{
			return false;
		}

		[command_buffer commit];
		[command_buffer waitUntilCompleted];
		if (command_buffer.status != MTLCommandBufferStatusCompleted)
		{
			return false;
		}
		buffer_c.copy_to_host(C, c_bytes);
	}

	return true;
}

bool Darknet::Apple::mps_layer_can_run(const Darknet::Layer & l, const bool train)
{
	if (train)
	{
		return false;
	}

	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		return false;
	}

	switch (l.type)
	{
		case Darknet::ELayerType::CONVOLUTIONAL:
		{
			if (l.binary || l.xnor)
			{
				return false;
			}
			if (l.groups <= 0 || (l.c % l.groups) != 0 || (l.n % l.groups) != 0)
			{
				return false;
			}
			const bool depthwise = (l.groups == l.c && l.n == l.c);
			if (l.groups > 1 && !depthwise &&
				(((l.c / l.groups) % 4) != 0 || ((l.n / l.groups) % 4) != 0))
			{
				return false;
			}
			if (l.dilation <= 0)
			{
				return false;
			}
			if (l.pad < 0 || l.pad > (l.size / 2))
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::MAXPOOL:
		{
			if (l.maxpool_depth || l.antialiasing)
			{
				return false;
			}
			if (l.pad < 0)
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::LOCAL_AVGPOOL:
		{
			if (l.antialiasing)
			{
				return false;
			}
			if (l.pad < 0)
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::AVGPOOL:
		{
			if (l.w <= 0 || l.h <= 0 || l.c <= 0 || l.batch <= 0)
			{
				return false;
			}
			if (l.out_w != 1 || l.out_h != 1)
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::SHORTCUT:
		{
			if (l.n != 1 || l.nweights != 0 || !l.input_sizes)
			{
				return false;
			}
			if (l.input_sizes[0] != l.outputs)
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::ROUTE:
		{
			if (l.n <= 0 || !l.input_sizes)
			{
				return false;
			}
			if (l.groups <= 0 || l.group_id < 0 || l.group_id >= l.groups)
			{
				return false;
			}
			if (l.out_c <= 0)
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::REORG:
		{
			if (l.stride <= 0)
			{
				return false;
			}
			if (!l.reverse)
			{
				if ((l.w % l.stride) != 0 || (l.h % l.stride) != 0)
				{
					return false;
				}
			}
			else
			{
				const int stride_sq = l.stride * l.stride;
				if (stride_sq <= 0 || (l.c % stride_sq) != 0)
				{
					return false;
				}
			}
			return true;
		}
		case Darknet::ELayerType::CONNECTED:
		{
			if (l.inputs <= 0 || l.outputs <= 0 || l.batch <= 0)
			{
				return false;
			}
			if (!l.batch_normalize && l.activation == LINEAR)
			{
				return false;
			}
			if (!l.batch_normalize && !mps_activation_supported(l.activation))
			{
				return false;
			}
			if (l.batch_normalize && (!l.scales || !l.biases || !l.rolling_mean || !l.rolling_variance))
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::SOFTMAX:
		{
			if (l.softmax_tree || l.spatial)
			{
				return false;
			}
			if (l.inputs <= 0 || l.outputs <= 0 || l.groups <= 0)
			{
				return false;
			}
			if ((l.inputs % l.groups) != 0)
			{
				return false;
			}
			return true;
		}
		case Darknet::ELayerType::UPSAMPLE:
		{
			if (l.reverse)
			{
				return false;
			}
			if (l.stride <= 0)
			{
				return false;
			}
			return true;
		}
		default:
			break;
	}

	return false;
}

const Darknet::Layer *Darknet::Apple::mps_prev_layer(const Darknet::NetworkState &state)
{
	if (state.net.layers && state.index > 0)
	{
		return &state.net.layers[state.index - 1];
	}
	return nullptr;
}

bool Darknet::Apple::mps_should_defer_readback(const Darknet::NetworkState &state)
{
	if (state.net.layers && (state.index + 1) < state.net.n)
	{
		const Darknet::Layer & next = state.net.layers[state.index + 1];
		return Darknet::Apple::mps_layer_can_run(next, state.train);
	}
	return false;
}

bool Darknet::Apple::mps_is_output_deferred(const Darknet::Layer *layer)
{
	if (!layer)
	{
		return false;
	}

	auto & deferred = get_mps_deferred_layers();
	return deferred.find(layer) != deferred.end();
}

namespace
{
	bool try_materialize_mps_output(const Darknet::Layer *layer, float *output, std::string & reason)
	{
		if (!layer)
		{
			return true;
		}

		auto & deferred = get_mps_deferred_layers();
		auto it = deferred.find(layer);
		if (it == deferred.end())
		{
			return true;
		}
		if (!output)
		{
			reason = "host output buffer is null";
			return false;
		}

		auto & ctx = get_mps_context();
		if (!ctx.ready)
		{
			reason = ctx.init_reason.empty() ? "MPS runtime is unavailable" : ctx.init_reason;
			return false;
		}

		std::string wait_reason;
		if (!wait_mps_command_buffer_if_needed(wait_reason))
		{
			reason = not wait_reason.empty() ? wait_reason : "MPS command buffer execution failed";
			return false;
		}

		MPSImage *output_image = get_cached_output_image(layer, layer->out_w, layer->out_h, layer->out_c, layer->batch);
		if (!output_image)
		{
			reason = "deferred MPS output image is unavailable";
			return false;
		}

		@autoreleasepool
		{
			if (!read_mps_output_image(output_image, layer->out_w, layer->out_h, layer->out_c, layer->batch, output))
			{
				reason = "failed to read deferred MPS output image";
				return false;
			}
		}

		deferred.erase(it);
		return true;
	}
}

void Darknet::Apple::mps_materialize_output(const Darknet::Layer *layer, float *output)
{
	std::string reason;
	if (!try_materialize_mps_output(layer, output ? output : (layer ? layer->output : nullptr), reason))
	{
		const int layer_index = layer ? layer->index : -1;
		darknet_fatal_error(DARKNET_LOC, "cannot materialize Apple MPS output for layer #%d: %s",
			layer_index, reason.c_str());
	}
}

void Darknet::Apple::mps_flush_deferred_output(const Darknet::Layer *layer)
{
	Darknet::Apple::mps_materialize_output(layer);
}

void Darknet::Apple::mps_flush_output_if_needed(const Darknet::Layer *layer, float *output)
{
	Darknet::Apple::mps_materialize_output(layer, output);
}

void Darknet::Apple::mps_invalidate_layer_cache(const Darknet::Layer *layer)
{
	wait_mps_command_buffer_if_needed();
	invalidate_layer_cache_impl(layer);
	advance_parameter_generation();
	if (layer)
	{
		Darknet::Apple::TrainingDetail::invalidate_layer_training_storage(*layer);
	}
}

void Darknet::Apple::mps_notify_layer_parameters_updated_cpu(const Darknet::Layer *layer)
{
	if (!layer)
	{
		return;
	}

	Darknet::Apple::TrainingDetail::note_layer_training_parameters_updated_cpu(*layer);
	advance_parameter_generation();
}

void Darknet::Apple::mps_notify_layer_parameters_updated_gpu(const Darknet::Layer *layer)
{
	if (!layer)
	{
		return;
	}

	Darknet::Apple::TrainingDetail::note_layer_training_parameters_updated_gpu(*layer);
	advance_parameter_generation();
}

void Darknet::Apple::mps_notify_layer_optimizer_state_updated_cpu(const Darknet::Layer *layer)
{
	if (layer)
	{
		Darknet::Apple::TrainingDetail::note_layer_training_optimizer_state_updated_cpu(*layer);
	}
}

void Darknet::Apple::mps_notify_layer_optimizer_state_updated_gpu(const Darknet::Layer *layer)
{
	if (layer)
	{
		Darknet::Apple::TrainingDetail::note_layer_training_optimizer_state_updated_gpu(*layer);
	}
}

void Darknet::Apple::mps_notify_layer_running_stats_updated_cpu(const Darknet::Layer *layer)
{
	if (!layer)
	{
		return;
	}

	Darknet::Apple::TrainingDetail::note_layer_training_running_stats_updated_cpu(*layer);
	advance_parameter_generation();
}

void Darknet::Apple::mps_notify_layer_running_stats_updated_gpu(const Darknet::Layer *layer)
{
	if (!layer)
	{
		return;
	}

	Darknet::Apple::TrainingDetail::note_layer_training_running_stats_updated_gpu(*layer);
	advance_parameter_generation();
}

bool Darknet::Apple::mps_convolution_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, bool *activation_applied, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (activation_applied)
	{
		*activation_applied = false;
	}

	if (l.groups <= 0 || (l.c % l.groups) != 0 || (l.n % l.groups) != 0)
	{
		reason = "invalid groups";
		return false;
	}

	const bool depthwise = (l.groups == l.c && l.n == l.c);
	if (l.groups > 1 && !depthwise &&
		(((l.c / l.groups) % 4) != 0 || ((l.n / l.groups) % 4) != 0))
	{
		reason = "grouped convolution channels must be multiples of four";
		return false;
	}

	if (l.pad < 0 || l.pad > (l.size / 2))
	{
		reason = "unsupported padding";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	auto & cache_map = get_mps_conv_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsConvCache>();
	}
	auto & cache = *cache_ptr;
	if (!cache_matches_layer(cache, l))
	{
		if (!build_conv_cache(cache, l, ctx.device))
		{
			reason = "failed to build convolution";
			return false;
		}
	}

	@autoreleasepool
	{
		bool need_input = true;
		MPSImage *input_image = nil;
		if (prev && input == prev->output)
		{
			input_image = get_cached_output_image(prev, l.w, l.h, l.c, l.batch);
			if (input_image)
			{
				need_input = false;
			}
		}

		if (!ensure_conv_images(cache, l, ctx.device, need_input))
		{
			reason = "failed to allocate MPSImage";
			return false;
		}

		if (!input_image)
		{
			input_image = cache.input_image;
		}
		MPSImage *output_image = cache.output_image;

		const NSUInteger batch = static_cast<NSUInteger>(l.batch);
		const NSUInteger in_channels = static_cast<NSUInteger>(l.c);
		const NSUInteger in_w = static_cast<NSUInteger>(l.w);
		const NSUInteger in_h = static_cast<NSUInteger>(l.h);

		const NSUInteger input_bytes_per_row = in_w * sizeof(float);
		const MTLRegion in_region = MTLRegionMake2D(0, 0, in_w, in_h);

		MPSImageReadWriteParams in_params = {};
		in_params.featureChannelOffset = 0;
		in_params.numberOfFeatureChannelsToReadWrite = in_channels;

		if (input_image == cache.input_image)
		{
			for (NSUInteger b = 0; b < batch; ++b)
			{
				const float *input_ptr = input + b * (l.c * l.h * l.w);
				[input_image writeBytes:input_ptr
					dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
					bytesPerRow:input_bytes_per_row
					region:in_region
					featureChannelInfo:in_params
					imageIndex:b];
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		[cache.conv encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
		bool activation_done = false;
		std::string status = "ok";
		if (cache.activation_fused || l.activation == LINEAR)
		{
			activation_done = true;
		}
		else
		{
			if (encode_mps_activation(command_buffer, output_image, l.activation, status))
			{
				activation_done = true;
			}
			else
			{
				status = "activation fallback";
			}
		}
		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		const bool should_defer = defer_readback && activation_done;
		if (!should_defer)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}

		if (activation_applied)
		{
			*activation_applied = activation_done;
		}

		reason = status;

	}
	return true;
}

bool Darknet::Apple::mps_maxpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (l.maxpool_depth || l.antialiasing)
	{
		reason = "unsupported maxpool mode";
		return false;
	}

	if (l.pad < 0)
	{
		reason = "padding not supported";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	auto & cache_map = get_mps_pool_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsPoolCache>();
	}
	auto & cache = *cache_ptr;
	const bool use_mps_pool = (l.pad <= l.size);
	if (use_mps_pool && !cache_matches_pool(cache, l, false))
	{
		if (!build_pool_cache(cache, l, ctx.device, false))
		{
			reason = "failed to build pooling";
			return false;
		}
	}

	@autoreleasepool
	{
		MPSImage *input_image = nil;
		bool need_input = true;
		if (prev && input == prev->output)
		{
			input_image = get_cached_output_image(prev, l.w, l.h, l.c, l.batch);
			if (input_image)
			{
				need_input = false;
			}
		}

		if (!ensure_pool_images(cache, l, ctx.device, need_input))
		{
			reason = "failed to allocate MPSImage";
			return false;
		}

		if (!input_image)
		{
			input_image = cache.input_image;
		}
		MPSImage *output_image = cache.output_image;

		const NSUInteger batch = static_cast<NSUInteger>(l.batch);
		const NSUInteger in_channels = static_cast<NSUInteger>(l.c);
		const NSUInteger in_w = static_cast<NSUInteger>(l.w);
		const NSUInteger in_h = static_cast<NSUInteger>(l.h);

		const NSUInteger input_bytes_per_row = in_w * sizeof(float);
		const MTLRegion in_region = MTLRegionMake2D(0, 0, in_w, in_h);

		MPSImageReadWriteParams in_params = {};
		in_params.featureChannelOffset = 0;
		in_params.numberOfFeatureChannelsToReadWrite = in_channels;

		if (input_image == cache.input_image)
		{
			for (NSUInteger b = 0; b < batch; ++b)
			{
				const float *input_ptr = input + b * (l.c * l.h * l.w);
				[input_image writeBytes:input_ptr
					dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
					bytesPerRow:input_bytes_per_row
					region:in_region
					featureChannelInfo:in_params
					imageIndex:b];
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		if (use_mps_pool && cache.pool)
		{
			[cache.pool encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
		}
		else
		{
			if (!encode_metal_pool(command_buffer, input_image, output_image, l, false, reason))
			{
				return false;
			}
		}
		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}

	}

	reason = "ok";
	return true;
}

bool Darknet::Apple::mps_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (l.antialiasing)
	{
		reason = "antialiasing not supported";
		return false;
	}

	if (l.pad < 0)
	{
		reason = "padding not supported";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	auto & cache_map = get_mps_pool_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsPoolCache>();
	}
	auto & cache = *cache_ptr;
	const bool use_mps_pool = (l.pad <= l.size);
	if (use_mps_pool && !cache_matches_pool(cache, l, true))
	{
		if (!build_pool_cache(cache, l, ctx.device, true))
		{
			reason = "failed to build pooling";
			return false;
		}
	}

	@autoreleasepool
	{
		MPSImage *input_image = nil;
		bool need_input = true;
		if (prev && input == prev->output)
		{
			input_image = get_cached_output_image(prev, l.w, l.h, l.c, l.batch);
			if (input_image)
			{
				need_input = false;
			}
		}

		if (!ensure_pool_images(cache, l, ctx.device, need_input))
		{
			reason = "failed to allocate MPSImage";
			return false;
		}

		if (!input_image)
		{
			input_image = cache.input_image;
		}
		MPSImage *output_image = cache.output_image;

		const NSUInteger batch = static_cast<NSUInteger>(l.batch);
		const NSUInteger in_channels = static_cast<NSUInteger>(l.c);
		const NSUInteger in_w = static_cast<NSUInteger>(l.w);
		const NSUInteger in_h = static_cast<NSUInteger>(l.h);

		const NSUInteger input_bytes_per_row = in_w * sizeof(float);
		const MTLRegion in_region = MTLRegionMake2D(0, 0, in_w, in_h);

		MPSImageReadWriteParams in_params = {};
		in_params.featureChannelOffset = 0;
		in_params.numberOfFeatureChannelsToReadWrite = in_channels;

		if (input_image == cache.input_image)
		{
			for (NSUInteger b = 0; b < batch; ++b)
			{
				const float *input_ptr = input + b * (l.c * l.h * l.w);
				[input_image writeBytes:input_ptr
					dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
					bytesPerRow:input_bytes_per_row
					region:in_region
					featureChannelInfo:in_params
					imageIndex:b];
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		if (use_mps_pool && cache.pool)
		{
			[cache.pool encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
		}
		else
		{
			if (!encode_metal_pool(command_buffer, input_image, output_image, l, true, reason))
			{
				return false;
			}
		}
		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}

	}

	reason = "ok";
	return true;
}

bool Darknet::Apple::mps_global_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (l.w <= 0 || l.h <= 0)
	{
		reason = "invalid avgpool size";
		return false;
	}

	if (l.c <= 0 || l.batch <= 0)
	{
		reason = "invalid channel/batch";
		return false;
	}

	if (l.out_w != 1 || l.out_h != 1)
	{
		reason = "not global avgpool";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	auto & cache_map = get_mps_global_avgpool_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsGlobalAvgPoolCache>();
	}
	auto & cache = *cache_ptr;
	if (!cache_matches_global_avgpool(cache, l))
	{
		if (!build_global_avgpool_cache(cache, l, ctx.device))
		{
			reason = "failed to build global avgpool";
			return false;
		}
	}

	@autoreleasepool
	{
		MPSImage *input_image = nil;
		bool need_input = true;
		if (prev && input == prev->output)
		{
			input_image = get_cached_output_image(prev, l.w, l.h, l.c, l.batch);
			if (input_image)
			{
				need_input = false;
			}
		}

		if (!ensure_global_avgpool_images(cache, l, ctx.device, need_input))
		{
			reason = "failed to allocate global avgpool images";
			return false;
		}

		if (!input_image)
		{
			input_image = cache.input_image;
		}
		MPSImage *output_image = cache.output_image;

		const NSUInteger batch = static_cast<NSUInteger>(l.batch);
		const NSUInteger in_channels = static_cast<NSUInteger>(l.c);
		const NSUInteger in_w = static_cast<NSUInteger>(l.w);
		const NSUInteger in_h = static_cast<NSUInteger>(l.h);

		const NSUInteger input_bytes_per_row = in_w * sizeof(float);
		const MTLRegion in_region = MTLRegionMake2D(0, 0, in_w, in_h);

		MPSImageReadWriteParams in_params = {};
		in_params.featureChannelOffset = 0;
		in_params.numberOfFeatureChannelsToReadWrite = in_channels;

		if (input_image == cache.input_image)
		{
			for (NSUInteger b = 0; b < batch; ++b)
			{
				const float *input_ptr = input + b * (l.c * l.h * l.w);
				[input_image writeBytes:input_ptr
					dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
					bytesPerRow:input_bytes_per_row
					region:in_region
					featureChannelInfo:in_params
					imageIndex:b];
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		[cache.pool encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}
	}

	reason = "ok";
	return true;
}

bool Darknet::Apple::mps_reorg_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (l.stride <= 0)
	{
		reason = "invalid stride";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	auto & cache_map = get_mps_reorg_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsReorgCache>();
	}
	auto & cache = *cache_ptr;

	@autoreleasepool
	{
		MPSImage *input_image = nil;
		bool need_input = true;
		if (prev && input == prev->output)
		{
			input_image = get_cached_output_image(prev, l.w, l.h, l.c, l.batch);
			if (input_image)
			{
				need_input = false;
			}
		}

		if (!ensure_reorg_images(cache, l, ctx.device, need_input))
		{
			reason = "failed to allocate reorg images";
			return false;
		}

		if (!input_image)
		{
			input_image = cache.input_image;
		}

		const NSUInteger batch = static_cast<NSUInteger>(l.batch);
		const NSUInteger in_channels = static_cast<NSUInteger>(l.c);
		const NSUInteger in_w = static_cast<NSUInteger>(l.w);
		const NSUInteger in_h = static_cast<NSUInteger>(l.h);

		const NSUInteger input_bytes_per_row = in_w * sizeof(float);
		const MTLRegion in_region = MTLRegionMake2D(0, 0, in_w, in_h);

		MPSImageReadWriteParams in_params = {};
		in_params.featureChannelOffset = 0;
		in_params.numberOfFeatureChannelsToReadWrite = in_channels;

		if (input_image == cache.input_image)
		{
			for (NSUInteger b = 0; b < batch; ++b)
			{
				const float *input_ptr = input + b * (l.c * l.h * l.w);
				[input_image writeBytes:input_ptr
					dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
					bytesPerRow:input_bytes_per_row
					region:in_region
					featureChannelInfo:in_params
					imageIndex:b];
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		id<MTLTexture> in_tex = input_image.texture;
		id<MTLTexture> out_tex = cache.output_image.texture;
		if (!in_tex || !out_tex)
		{
			reason = "missing reorg texture";
			return false;
		}

		const uint32_t in_channels_u = static_cast<uint32_t>(l.c);
		const uint32_t out_channels_u = static_cast<uint32_t>(l.out_c);
		const uint32_t stride_u = static_cast<uint32_t>(l.stride);
		MetalDispatchTexture textures[] = {
			{ (__bridge void *)in_tex },
			{ (__bridge void *)out_tex }
		};
		MetalDispatchBytes bytes[] = {
			{ &in_channels_u, sizeof(in_channels_u) },
			{ &out_channels_u, sizeof(out_channels_u) },
			{ &stride_u, sizeof(stride_u) }
		};
		const std::string & kernel_name = l.reverse ? "reorg_reverse_kernel" : "reorg_forward_kernel";
		if (!metal_dispatch_texture_kernel(kernel_name,
				(__bridge void *)out_tex,
				8, 8,
				textures, 2,
				nullptr, 0,
				bytes, 3,
				(__bridge void *)command_buffer))
		{
			reason = "failed to dispatch reorg kernel";
			return false;
		}

		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}
	}

	reason = "ok";
	return true;
}

bool Darknet::Apple::mps_connected_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, bool *activation_applied, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	if (l.inputs <= 0 || l.outputs <= 0 || l.batch <= 0)
	{
		reason = "invalid connected size";
		return false;
	}

	const bool needs_batchnorm = (l.batch_normalize != 0);
	const bool needs_activation = (l.activation != LINEAR);
	const bool activation_supported = mps_activation_supported(l.activation);
	if (!needs_batchnorm && !needs_activation)
	{
		reason = "no MPS work";
		return false;
	}
	if (!needs_batchnorm && needs_activation && !activation_supported)
	{
		reason = "activation not supported";
		return false;
	}

	if (prev && input == prev->output)
	{
		Darknet::Apple::mps_materialize_output(prev);
	}

	fill_cpu(l.outputs*l.batch, 0, output, 1);
	const int m = l.batch;
	const int k = l.inputs;
	const int n = l.outputs;
	gemm_cpu(0, 1, m, n, k, 1, const_cast<float *>(input), k, l.weights, k, 1, output, n);

	if (!needs_batchnorm)
	{
		for (int i = 0; i < l.batch; ++i)
		{
			axpy_cpu(l.outputs, 1, l.biases, 1, output + i*l.outputs, 1);
		}
	}

	auto & cache_map = get_mps_connected_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsConnectedCache>();
	}
	auto & cache = *cache_ptr;
	if (!cache_matches_connected(cache, l))
	{
		if (!build_connected_cache(cache, l, ctx.device))
		{
			reason = "failed to build connected cache";
			return false;
		}
	}

	@autoreleasepool
	{
		if (!ensure_connected_images(cache, l, ctx.device))
		{
			reason = "failed to allocate connected images";
			return false;
		}

		MPSImage *output_image = cache.output_image;
		MPSImage *gemm_output_image = cache.batchnorm ? cache.intermediate_image : output_image;
		const NSUInteger batch = static_cast<NSUInteger>(l.batch);
		const NSUInteger out_channels = static_cast<NSUInteger>(l.outputs);
		const NSUInteger out_w = static_cast<NSUInteger>(l.out_w);
		const NSUInteger out_h = static_cast<NSUInteger>(l.out_h);
		const NSUInteger output_bytes_per_row = out_w * sizeof(float);
		const MTLRegion out_region = MTLRegionMake2D(0, 0, out_w, out_h);

		MPSImageReadWriteParams out_params = {};
		out_params.featureChannelOffset = 0;
		out_params.numberOfFeatureChannelsToReadWrite = out_channels;

		for (NSUInteger b = 0; b < batch; ++b)
		{
			const float *output_ptr = output + b * l.outputs;
			[gemm_output_image writeBytes:output_ptr
				dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
				bytesPerRow:output_bytes_per_row
				region:out_region
				featureChannelInfo:out_params
				imageIndex:b];
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		if (cache.batchnorm)
		{
			[cache.batchnorm encodeToCommandBuffer:command_buffer sourceImage:gemm_output_image destinationImage:output_image];
		}

		bool activation_done = false;
		std::string status = "ok";
		if (cache.activation_fused || l.activation == LINEAR)
		{
			activation_done = true;
		}
		else
		{
			if (encode_mps_activation(command_buffer, output_image, l.activation, status))
			{
				activation_done = true;
			}
			else
			{
				status = "activation fallback";
			}
		}

		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		const bool should_defer = defer_readback && activation_done;
		if (!should_defer)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}

		if (activation_applied)
		{
			*activation_applied = activation_done;
		}

		reason = status;
	}

	return true;
}

bool Darknet::Apple::mps_softmax_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, std::string & reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (l.softmax_tree || l.spatial)
	{
		reason = "softmax mode not supported";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	const size_t bytes = static_cast<size_t>(l.inputs) * static_cast<size_t>(l.batch) * sizeof(float);
	auto & cache_map = get_mps_softmax_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsSoftmaxCache>();
	}
	auto & cache = *cache_ptr;

	if (cache.bytes != bytes || !cache.input_buffer || !cache.output_buffer)
	{
		if (!cache.input_buffer.allocate(bytes) || !cache.output_buffer.allocate(bytes))
		{
			reason = "failed to allocate softmax buffers";
			return false;
		}
		cache.bytes = bytes;
	}

	float *input_ptr = nullptr;
	if (cache.input_buffer)
	{
		id<MTLBuffer> buf = (__bridge id<MTLBuffer>)cache.input_buffer.get()->handle;
		input_ptr = static_cast<float *>([buf contents]);
	}
	if (!input_ptr)
	{
		reason = "softmax input map failed";
		return false;
	}

	bool filled_from_mps = false;
	if (prev && input == prev->output)
	{
		MPSImage *image = get_cached_output_image(prev, prev->out_w, prev->out_h, prev->out_c, prev->batch);
		if (image)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(image, prev->out_w, prev->out_h, prev->out_c, prev->batch, input_ptr))
			{
				reason = "failed to read MPS softmax input";
				return false;
			}
			filled_from_mps = true;
		}
	}
	if (!filled_from_mps)
	{
		if (!cache.input_buffer.upload(0, input, bytes))
		{
			reason = "failed to upload softmax input";
			return false;
		}
	}

	const uint32_t n = static_cast<uint32_t>(l.inputs / l.groups);
	const uint32_t batch = static_cast<uint32_t>(l.batch);
	const uint32_t batch_offset = static_cast<uint32_t>(l.inputs);
	const uint32_t groups = static_cast<uint32_t>(l.groups);
	const uint32_t group_offset = static_cast<uint32_t>(l.inputs / l.groups);
	const uint32_t stride = 1;
	const float temp = l.temperature;
	const size_t threads = static_cast<size_t>(n) * batch * groups;
	if (threads == 0)
	{
		reason = "softmax size invalid";
		return false;
	}
	const size_t threads_per_group = (threads < 256) ? threads : 256;

	MetalDispatchBuffer buffers[] = {
		{ cache.input_buffer.get(), 0 },
		{ cache.output_buffer.get(), 0 }
	};
	MetalDispatchBytes bytes_args[] = {
		{ &n, sizeof(n) },
		{ &batch, sizeof(batch) },
		{ &batch_offset, sizeof(batch_offset) },
		{ &groups, sizeof(groups) },
		{ &group_offset, sizeof(group_offset) },
		{ &stride, sizeof(stride) },
		{ &temp, sizeof(temp) }
	};
	if (!metal_dispatch_1d("softmax_kernel",
			threads, threads_per_group,
			nullptr, 0,
			buffers, 2,
			bytes_args, 7,
			nullptr))
	{
		reason = "failed to dispatch softmax";
		return false;
	}

	if (!cache.output_buffer.download(0, output, bytes))
	{
		reason = "failed to download softmax output";
		return false;
	}

	reason = "ok";
	return true;
}

bool Darknet::Apple::mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const Darknet::Layer *from,
	const float *input, float *output, bool defer_readback, bool *activation_applied, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (activation_applied)
	{
		*activation_applied = false;
	}

	if (!from || l.n != 1 || l.nweights != 0 || !l.input_sizes || l.input_sizes[0] != l.outputs)
	{
		reason = "unsupported shortcut";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	MPSImage *primary_image = nil;
	if (prev && input == prev->output)
	{
		primary_image = get_cached_output_image(prev, l.w, l.h, l.c, l.batch);
	}
	MPSImage *secondary_image = get_cached_output_image(from, l.w, l.h, l.c, l.batch);
	if (!primary_image || !secondary_image)
	{
		reason = "missing MPS input";
		return false;
	}

	auto & cache_map = get_mps_add_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsAddCache>();
	}
	auto & cache = *cache_ptr;
	if (!cache_matches_add(cache, l))
	{
		if (!build_add_cache(cache, l, ctx.device))
		{
			reason = "failed to build add";
			return false;
		}
	}

	@autoreleasepool
	{
		if (!ensure_add_output_image(cache, l, ctx.device))
		{
			reason = "failed to allocate MPSImage";
			return false;
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		[cache.add encodeToCommandBuffer:command_buffer
			primaryImage:primary_image
			secondaryImage:secondary_image
			destinationImage:cache.output_image];
		bool activation_done = false;
		std::string status = "ok";
		if (l.activation == LINEAR)
		{
			activation_done = true;
		}
		else if (encode_mps_activation(command_buffer, cache.output_image, l.activation, status))
		{
			activation_done = true;
		}
		else
		{
			status = "activation fallback";
		}

		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		const bool should_defer = defer_readback && activation_done;
		if (!should_defer)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read shortcut output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}

		if (activation_applied)
		{
			*activation_applied = activation_done;
		}

		reason = status;
	}

	return true;
}

bool Darknet::Apple::mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
	float *output, bool defer_readback, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (!output)
	{
		reason = "null output";
		return false;
	}

	if (!net.layers || net.n <= 0)
	{
		reason = "missing network layers";
		return false;
	}

	if (l.n <= 0 || !l.input_layers || !l.input_sizes)
	{
		reason = "missing route inputs";
		return false;
	}

	if (l.groups <= 0 || l.group_id < 0 || l.group_id >= l.groups)
	{
		reason = "invalid route groups";
		return false;
	}

	if (l.out_c <= 0)
	{
		reason = "route channels invalid";
		return false;
	}

	struct RouteInput
	{
		const Darknet::Layer *layer = nullptr;
		MPSImage *image = nil;
		int input_channels = 0;
		int input_offset = 0;
		int copy_channels = 0;
		int output_offset = 0;
	};

	std::vector<RouteInput> inputs;
	inputs.reserve(static_cast<size_t>(l.n));
	int total_channels = 0;

	for (int i = 0; i < l.n; ++i)
	{
		const int index = l.input_layers[i];
		if (index < 0 || index >= net.n)
		{
			reason = "route input out of range";
			return false;
		}
		const Darknet::Layer & src = net.layers[index];
		if (src.out_w != l.out_w || src.out_h != l.out_h)
		{
			reason = "route input size mismatch";
			return false;
		}
		if (src.batch != l.batch)
		{
			reason = "route input batch mismatch";
			return false;
		}
		if (src.out_c <= 0)
		{
			reason = "route input channels invalid";
			return false;
		}
		if ((src.out_c % l.groups) != 0)
		{
			reason = "route input groups mismatch";
			return false;
		}

		const int part_c = src.out_c / l.groups;
		if (part_c <= 0)
		{
			reason = "route group channels invalid";
			return false;
		}

		MPSImage *image = get_cached_output_image(&src, src.out_w, src.out_h, src.out_c, src.batch);
		if (!image)
		{
			reason = "missing MPS input";
			return false;
		}
		const int input_offset = part_c * l.group_id;
		inputs.push_back({ &src, image, src.out_c, input_offset, part_c, total_channels });
		total_channels += part_c;
	}

	if (total_channels != l.out_c)
	{
		reason = "route channel mismatch";
		return false;
	}

	auto & cache_map = get_mps_route_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsRouteCache>();
	}
	auto & cache = *cache_ptr;

	@autoreleasepool
	{
		if (!ensure_route_output_image(cache, l, ctx.device))
		{
			reason = "failed to allocate route output";
			return false;
		}

		id<MTLTexture> dst_texture = cache.output_image.texture;
		if (!dst_texture)
		{
			reason = "missing output texture";
			return false;
		}

		for (const auto & entry : inputs)
		{
			if (!entry.image || !entry.image.texture)
			{
				reason = "missing input texture";
				return false;
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		MetalDispatchTexture clear_textures[] = {
			{ (__bridge void *)dst_texture }
		};
		if (!metal_dispatch_texture_kernel("route_clear_kernel",
				(__bridge void *)dst_texture,
				8, 8,
				clear_textures, 1,
				nullptr, 0,
				nullptr, 0,
				(__bridge void *)command_buffer))
		{
			reason = "failed to clear route output";
			return false;
		}

		for (const auto & entry : inputs)
		{
			const uint32_t in_channels = static_cast<uint32_t>(entry.input_channels);
			const uint32_t out_channels = static_cast<uint32_t>(l.out_c);
			const uint32_t in_offset = static_cast<uint32_t>(entry.input_offset);
			const uint32_t out_offset = static_cast<uint32_t>(entry.output_offset);
			const uint32_t copy_channels = static_cast<uint32_t>(entry.copy_channels);
			MetalDispatchTexture textures[] = {
				{ (__bridge void *)entry.image.texture },
				{ (__bridge void *)dst_texture }
			};
			MetalDispatchBytes bytes[] = {
				{ &in_channels, sizeof(in_channels) },
				{ &out_channels, sizeof(out_channels) },
				{ &in_offset, sizeof(in_offset) },
				{ &out_offset, sizeof(out_offset) },
				{ &copy_channels, sizeof(copy_channels) }
			};
			if (!metal_dispatch_texture_kernel("route_copy_kernel",
					(__bridge void *)dst_texture,
					8, 8,
					textures, 2,
					nullptr, 0,
					bytes, 5,
					(__bridge void *)command_buffer))
			{
				reason = "failed to encode route copy";
				return false;
			}
		}
		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}

	}

	reason = "ok";
	return true;
}

bool Darknet::Apple::mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, std::string & reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		reason = "MPS not available";
		return false;
	}

	if (l.reverse || l.stride <= 0)
	{
		reason = "unsupported upsample";
		return false;
	}

	if (!input || !output)
	{
		reason = "null input/output";
		return false;
	}

	auto & cache_map = get_mps_upsample_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsUpsampleCache>();
	}
	auto & cache = *cache_ptr;

	@autoreleasepool
	{
		MPSImage *input_image = nil;
		bool need_input = true;
		if (prev && input == prev->output)
		{
			input_image = get_cached_output_image(prev, l.w, l.h, l.c, l.batch);
			if (input_image)
			{
				need_input = false;
			}
		}

		if (!ensure_upsample_images(cache, l, ctx.device, need_input))
		{
			reason = "failed to allocate upsample images";
			return false;
		}

		if (!input_image)
		{
			input_image = cache.input_image;
		}

		const NSUInteger batch = static_cast<NSUInteger>(l.batch);
		const NSUInteger in_channels = static_cast<NSUInteger>(l.c);
		const NSUInteger in_w = static_cast<NSUInteger>(l.w);
		const NSUInteger in_h = static_cast<NSUInteger>(l.h);

		const NSUInteger input_bytes_per_row = in_w * sizeof(float);
		const MTLRegion in_region = MTLRegionMake2D(0, 0, in_w, in_h);

		MPSImageReadWriteParams in_params = {};
		in_params.featureChannelOffset = 0;
		in_params.numberOfFeatureChannelsToReadWrite = in_channels;

		if (input_image == cache.input_image)
		{
			for (NSUInteger b = 0; b < batch; ++b)
			{
				const float *input_ptr = input + b * (l.c * l.h * l.w);
				[input_image writeBytes:input_ptr
					dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
					bytesPerRow:input_bytes_per_row
					region:in_region
					featureChannelInfo:in_params
					imageIndex:b];
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			reason = "failed to create command buffer";
			return false;
		}

		if (!encode_mps_upsample(command_buffer, input_image, cache.output_image, l.stride, l.scale, reason))
		{
			return false;
		}

		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			if (!wait_mps_command_buffer_if_needed(reason))
			{
				return false;
			}
			if (!read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output))
			{
				reason = "failed to read MPS output";
				return false;
			}
		}
		else
		{
			deferred.insert(&l);
		}

	}

	reason = "ok";
	return true;
}

bool Darknet::Apple::mps_configure_training_graph(Darknet::Network & net, std::string & reason)
{
	return Darknet::Apple::configure_training_graph(net, reason);
}

bool Darknet::Apple::mps_training_graph_enabled(const Darknet::Network & net)
{
	return Darknet::Apple::training_graph_enabled(net);
}

void Darknet::Apple::mps_invalidate_training_graph(Darknet::Network & net)
{
	Darknet::Apple::invalidate_training_graph(net);
}

float Darknet::Apple::mps_train_network_datum(Darknet::Network & net, const FloatBufferView & input, const FloatBufferView & truth)
{
	return Darknet::Apple::train_graph_network_datum(net, input, truth);
}

namespace Darknet::Apple
{
	bool training_mps_ready()
	{
		return get_mps_context().ready;
	}

	id<MTLDevice> training_mps_device()
	{
		return get_mps_context().device;
	}

	void * training_mps_command_queue_handle()
	{
		return (__bridge void *)get_mps_context().queue;
	}

	id<MTLCommandBuffer> create_training_command_buffer_handle()
	{
		return create_mps_command_buffer(get_mps_context());
	}

	void wait_for_training_command_buffer()
	{
		wait_mps_command_buffer_if_needed();
	}
}
void Darknet::Apple::mps_update_network(Darknet::Network & net)
{
	Darknet::Apple::update_training_graph(net);
}

void Darknet::Apple::mps_sync_training_host_state(Darknet::Network & net, const bool sync_rolling_stats, const bool sync_parameters)
{
	Darknet::Apple::sync_training_host_state(net, sync_rolling_stats, sync_parameters);
}

void Darknet::Apple::mps_prepare_network_for_inference(Darknet::Network & net)
{
	if (Darknet::Apple::mps_training_graph_enabled(net))
	{
		Darknet::Apple::mps_sync_training_host_state(net, true, true);
	}
}

bool Darknet::Apple::mps_prepare_validation_network(Darknet::Network & training, Darknet::Network & validation, std::string & reason)
{
	return Darknet::Apple::prepare_validation_network(training, validation, reason);
}

void Darknet::Apple::show_mps_info()
{
	TAT(TATPARMS);

	auto & cfg_and_state = Darknet::CfgAndState::get();
	auto & ctx = get_mps_context();

	if (!ctx.ready)
	{
		*cfg_and_state.output
			<< Darknet::in_colour(Darknet::EColour::kBrightRed, "Apple MPS not available");
		if (!ctx.init_reason.empty() && ctx.init_reason != "ok")
		{
			*cfg_and_state.output << ": " << ctx.init_reason;
		}
		*cfg_and_state.output << std::endl;
		return;
	}

	NSString *device_name = [ctx.device name];
	const char *name = device_name ? [device_name UTF8String] : "unknown";
	*cfg_and_state.output << "Apple MPS device: " << Darknet::in_colour(Darknet::EColour::kBrightGreen, name);

	size_t working_set = static_cast<size_t>(ctx.device.recommendedMaxWorkingSetSize);
	*cfg_and_state.output << ", " << Darknet::in_colour(Darknet::EColour::kYellow, size_to_IEC_string(working_set));

	*cfg_and_state.output << std::endl;
}

#endif // DARKNET_USE_MPS
