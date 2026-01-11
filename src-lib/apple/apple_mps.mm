#include "darknet_internal.hpp"
#include "darknet_layers.hpp"
#include "apple_mps.hpp"
#include "metal_backend.hpp"
#include "gemm.hpp"

#ifdef DARKNET_USE_MPS

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

/// @todo denizz Why are all these needed?  Aren't they already in darknet_internal.hpp?
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <mutex>
#include <limits>

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

namespace
{
	struct MpsContext
	{
		id<MTLDevice> device = nil;
		id<MTLCommandQueue> queue = nil;
		bool ready = false;
		id<MTLBuffer> bufferA = nil;
		id<MTLBuffer> bufferB = nil;
		id<MTLBuffer> bufferC = nil;
		size_t bufferA_bytes = 0;
		size_t bufferB_bytes = 0;
		size_t bufferC_bytes = 0;
	};

	struct MpsCommandBufferState
	{
		id<MTLCommandBuffer> last_buffer = nil;
	};

	MpsContext & get_mps_context()
	{
		static MpsContext ctx;
		static std::once_flag once;
		std::call_once(once, [&]()
		{
			@autoreleasepool
			{
				ctx.device = MTLCreateSystemDefaultDevice();
				if (ctx.device && MPSSupportsMTLDevice(ctx.device))
				{
					ctx.queue = [ctx.device newCommandQueue];
					ctx.ready = (ctx.queue != nil);
				}
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
		state.last_buffer = buffer;
	}

	void wait_mps_command_buffer_if_needed()
	{
		auto & state = get_mps_command_buffer_state();
		if (!state.last_buffer)
		{
			return;
		}
		[state.last_buffer waitUntilCompleted];
		state.last_buffer = nil;
	}

	std::mutex & get_mps_gemm_mutex()
	{
		static std::mutex mutex;
		return mutex;
	}

	bool ensure_buffer(__strong id<MTLBuffer> & buffer, size_t & current_bytes, size_t required_bytes, id<MTLDevice> device)
	{
		if (required_bytes == 0)
		{
			return false;
		}
		if (!buffer || required_bytes > current_bytes)
		{
			buffer = [device newBufferWithLength:required_bytes options:MTLResourceStorageModeShared];
			current_bytes = buffer ? required_bytes : 0;
		}
		return buffer != nil;
	}

	struct MpsConvCache
	{
		MPSCNNConvolution *conv = nil;
		MPSCNNNeuron *neuron = nil;
		MPSCNNBatchNormalization *batchnorm = nil;
		MpsBatchNormDataSource *batchnorm_source = nil;
		std::vector<float> weights;
		std::vector<float> biases;
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
		float leaky_alpha = 0.0f;
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
		MPSCNNNeuron *neuron = nil;
		ACTIVATION activation = LINEAR;
		float leaky_alpha = 0.0f;
		bool batch_normalize = false;
		int outputs = 0;
		MPSImage *output_image = nil;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsAddCache
	{
		MPSCNNAdd *add = nil;
		MPSCNNNeuron *neuron = nil;
		ACTIVATION activation = LINEAR;
		float leaky_alpha = 0.0f;
		MPSImage *output_image = nil;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
	};

	struct MpsWeightedAddCache
	{
		MPSCNNNeuron *neuron = nil;
		ACTIVATION activation = LINEAR;
		float leaky_alpha = 0.0f;
		MPSImage *output_image = nil;
		int output_w = 0;
		int output_h = 0;
		int output_c = 0;
		int output_batch = 0;
		MetalBuffer weights_in;
		MetalBuffer weights_add;
		const float *weights_ptr = nullptr;
		int weights_channels = 0;
		bool weights_ready = false;
		WEIGHTS_NORMALIZATION_T weights_norm = NO_NORMALIZATION;
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
		MetalBuffer input_buffer;
		MetalBuffer output_buffer;
		size_t bytes = 0;
	};

	struct MpsYoloCache
	{
		MetalBuffer buffer;
		size_t bytes = 0;
	};

	struct MpsYoloDecodeCache
	{
		MetalBuffer input_buffer;
		MetalBuffer output_buffer;
		MetalBuffer biases_buffer;
		MetalBuffer mask_buffer;
		size_t input_bytes = 0;
		size_t output_bytes = 0;
		size_t bias_bytes = 0;
		size_t mask_bytes = 0;
	};

	struct MpsYoloCandidatesCache
	{
		MetalBuffer input_buffer;
		MetalBuffer indices_buffer;
		MetalBuffer count_buffer;
		size_t input_bytes = 0;
		size_t indices_bytes = 0;
	};

	struct MpsNmsCache
	{
		MetalBuffer boxes_buffer;
		MetalBuffer scores_buffer;
		MetalBuffer order_buffer;
		size_t boxes_bytes = 0;
		size_t scores_bytes = 0;
		size_t order_bytes = 0;
	};

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConvCache>> & get_mps_conv_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConvCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsPoolCache>> & get_mps_pool_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsPoolCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsGlobalAvgPoolCache>> & get_mps_global_avgpool_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsGlobalAvgPoolCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConnectedCache>> & get_mps_connected_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsConnectedCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsAddCache>> & get_mps_add_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsAddCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsWeightedAddCache>> & get_mps_weighted_add_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsWeightedAddCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsRouteCache>> & get_mps_route_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsRouteCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsUpsampleCache>> & get_mps_upsample_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsUpsampleCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsReorgCache>> & get_mps_reorg_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsReorgCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsSoftmaxCache>> & get_mps_softmax_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsSoftmaxCache>> cache;
		return cache;
	}

	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsYoloCache>> & get_mps_yolo_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsYoloCache>> cache;
		return cache;
	}

	/** \brief Thread-local cache for YOLO box decode buffers. \ingroup mps_postproc */
	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsYoloDecodeCache>> & get_mps_yolo_decode_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsYoloDecodeCache>> cache;
		return cache;
	}

	/** \brief Thread-local cache for YOLO candidate indices. \ingroup mps_postproc */
	std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsYoloCandidatesCache>> & get_mps_yolo_candidates_cache()
	{
		thread_local std::unordered_map<const Darknet::Layer *, std::unique_ptr<MpsYoloCandidatesCache>> cache;
		return cache;
	}

	/** \brief Thread-local cache for GPU NMS buffers. \ingroup mps_postproc */
	MpsNmsCache & get_mps_nms_cache()
	{
		thread_local MpsNmsCache cache;
		return cache;
	}

	/** \brief Tracks layers with deferred readback to reduce CPU syncs. \ingroup mps_backend */
	std::unordered_set<const Darknet::Layer *> & get_mps_deferred_layers()
	{
		thread_local std::unordered_set<const Darknet::Layer *> deferred;
		return deferred;
	}

	void reset_conv_images(MpsConvCache & cache);
	void reset_pool_images(MpsPoolCache & cache);
	void reset_global_avgpool_images(MpsGlobalAvgPoolCache & cache);
	void reset_connected_images(MpsConnectedCache & cache);
	void reset_add_images(MpsAddCache & cache);
	void reset_route_images(MpsRouteCache & cache);
	void reset_upsample_images(MpsUpsampleCache & cache);
	void reset_reorg_images(MpsReorgCache & cache);

	bool mps_supports_grouped_conv()
	{
		return [MPSCNNConvolutionDescriptor instancesRespondToSelector:@selector(setGroups:)];
	}

	bool mps_supports_dilated_conv()
	{
		return [MPSCNNConvolutionDescriptor instancesRespondToSelector:@selector(setDilationRateX:)];
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
			cache.activation == l.activation;
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

	void reset_route_images(MpsRouteCache & cache)
	{
		cache.output_image = nil;
		cache.output_w = 0;
		cache.output_h = 0;
		cache.output_c = 0;
		cache.output_batch = 0;
	}

	void reset_weighted_add_images(MpsWeightedAddCache & cache)
	{
		cache.output_image = nil;
		cache.output_w = 0;
		cache.output_h = 0;
		cache.output_c = 0;
		cache.output_batch = 0;
	}

	void reset_upsample_images(MpsUpsampleCache & cache)
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
		cache.stride = 0;
		cache.scale = 0.0f;
	}

	void reset_reorg_images(MpsReorgCache & cache)
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
		cache.stride = 0;
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

	bool cache_matches_weighted_add(const MpsWeightedAddCache & cache, const Darknet::Layer & l)
	{
		return cache.activation == l.activation &&
			cache.weights_norm == l.weights_normalization;
	}

	bool ensure_weighted_add_output_image(MpsWeightedAddCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
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

	MPSCNNNeuron *build_neuron(ACTIVATION activation, id<MTLDevice> device, float *alpha_out);

	bool build_add_cache(MpsAddCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		cache.add = [[MPSCNNAdd alloc] initWithDevice:device];
		cache.activation = l.activation;
		cache.neuron = build_neuron(l.activation, device, &cache.leaky_alpha);
		reset_add_images(cache);
		return cache.add != nil;
	}

	bool build_weighted_add_cache(MpsWeightedAddCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		cache.activation = l.activation;
		cache.weights_norm = l.weights_normalization;
		cache.neuron = build_neuron(l.activation, device, &cache.leaky_alpha);
		reset_weighted_add_images(cache);
		if (cache.weights_in.handle)
		{
			metal_buffer_free(&cache.weights_in);
		}
		if (cache.weights_add.handle)
		{
			metal_buffer_free(&cache.weights_add);
		}
		cache.weights_ptr = nullptr;
		cache.weights_channels = 0;
		cache.weights_ready = false;
		return true;
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

	bool encode_mps_activation(id<MTLCommandBuffer> command_buffer, MPSImage *image, ACTIVATION activation, const char **reason)
	{
		if (!command_buffer || !image)
		{
			if (reason) *reason = "invalid activation inputs";
			return false;
		}

		const char *kernel_name = nullptr;
		uint32_t channels = 0;
		uint32_t slices_per_image = 0;
		uint32_t use_max_val = 0;
		MetalDispatchBytes bytes[3] = {};
		size_t bytes_count = 0;
		switch (activation)
		{
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
				if (reason) *reason = "activation not supported";
				return false;
		}

		id<MTLTexture> texture = image.texture;
		if (!texture)
		{
			if (reason) *reason = "missing activation texture";
			return false;
		}

		if ((bytes_count > 0) && (channels == 0 || slices_per_image == 0))
		{
			if (reason) *reason = "activation not supported";
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
			if (reason) *reason = "failed to dispatch activation kernel";
			return false;
		}
		return true;
	}

	bool encode_mps_upsample(id<MTLCommandBuffer> command_buffer, MPSImage *input_image, MPSImage *output_image,
		int stride, float scale, const char **reason)
	{
		if (!command_buffer || !input_image || !output_image)
		{
			if (reason) *reason = "invalid upsample inputs";
			return false;
		}

		if (stride <= 0)
		{
			if (reason) *reason = "invalid stride";
			return false;
		}

		id<MTLTexture> in_tex = input_image.texture;
		id<MTLTexture> out_tex = output_image.texture;
		if (!in_tex || !out_tex)
		{
			if (reason) *reason = "missing upsample texture";
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
			if (reason) *reason = "failed to dispatch upsample kernel";
			return false;
		}

		return true;
	}

	bool encode_metal_pool(id<MTLCommandBuffer> command_buffer, MPSImage *input_image, MPSImage *output_image,
		const Darknet::Layer & l, bool avgpool, const char **reason)
	{
		if (!command_buffer || !input_image || !output_image)
		{
			if (reason) *reason = "invalid pool inputs";
			return false;
		}

		id<MTLTexture> in_tex = input_image.texture;
		id<MTLTexture> out_tex = output_image.texture;
		if (!in_tex || !out_tex)
		{
			if (reason) *reason = "missing pool texture";
			return false;
		}

		const uint32_t channels = static_cast<uint32_t>(l.c);
		const uint32_t size = static_cast<uint32_t>(l.size);
		const uint32_t stride_x = static_cast<uint32_t>(l.stride_x);
		const uint32_t stride_y = static_cast<uint32_t>(l.stride_y);
		const int32_t pad = static_cast<int32_t>(l.pad / 2);
		const char *kernel_name = avgpool ? "avgpool_kernel" : "maxpool_kernel";
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
			if (reason) *reason = "failed to dispatch pool kernel";
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

	void pack_weights_hwio(const Darknet::Layer & l, std::vector<float> & packed)
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
						const size_t dst = static_cast<size_t>(((ky * l.size + kx) * in_channels + ic) * l.n + oc);
						packed[dst] = l.weights[src];
					}
				}
			}
		}
	}

	MPSCNNNeuron *build_neuron(ACTIVATION activation, id<MTLDevice> device, float *alpha_out)
	{
		if (!device)
		{
			return nil;
		}

		if (alpha_out)
		{
			*alpha_out = 0.0f;
		}

		switch (activation)
		{
			case RELU:
			{
				return [[MPSCNNNeuronReLU alloc] initWithDevice:device a:0.0f];
			}
			case LEAKY:
			{
				const float alpha = 0.1f;
				if (alpha_out)
				{
					*alpha_out = alpha;
				}
				return [[MPSCNNNeuronReLU alloc] initWithDevice:device a:alpha];
			}
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
		if (!device)
		{
			return false;
		}

		if (l.groups <= 0 || (l.c % l.groups) != 0 || (l.n % l.groups) != 0)
		{
			return false;
		}

		const int in_channels = l.c / l.groups;
		pack_weights_ohwi(l, cache.weights);

		cache.biases.assign(static_cast<size_t>(l.n), 0.0f);
		if (!l.batch_normalize && l.biases)
		{
			for (int i = 0; i < l.n; ++i)
			{
				cache.biases[i] = l.biases[i];
			}
		}

	MPSCNNConvolutionDescriptor *desc = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:l.size
		kernelHeight:l.size
		inputFeatureChannels:in_channels
		outputFeatureChannels:l.n];
	desc.strideInPixelsX = l.stride_x;
	desc.strideInPixelsY = l.stride_y;

	if (l.groups > 1)
	{
		if (!mps_supports_grouped_conv())
		{
			return false;
		}
		desc.groups = l.groups;
	}

	if (l.dilation != 1 && ![desc respondsToSelector:@selector(setDilationRateX:)])
	{
		return false;
	}

	if ([desc respondsToSelector:@selector(setDilationRateX:)])
	{
		desc.dilationRateX = l.dilation;
		desc.dilationRateY = l.dilation;
	}

		cache.conv = [[MPSCNNConvolution alloc] initWithDevice:device
			convolutionDescriptor:desc
			kernelWeights:cache.weights.data()
			biasTerms:cache.biases.data()
			flags:MPSCNNConvolutionFlagsNone];

		if (!cache.conv)
		{
			return false;
		}

		cache.conv.edgeMode = MPSImageEdgeModeZero;
		const int offset = (l.size / 2) - l.pad;
		const MPSOffset mps_offset = { offset, offset, 0 };
		cache.conv.offset = mps_offset;

		cache.neuron = nil;
		cache.leaky_alpha = 0.0f;
		cache.batchnorm = nil;
		cache.batchnorm_source = nil;

		if (l.batch_normalize)
		{
			if (!l.scales || !l.biases || !l.rolling_mean || !l.rolling_variance)
			{
				return false;
			}
			cache.batchnorm_source = [[MpsBatchNormDataSource alloc]
				initWithChannels:static_cast<NSUInteger>(l.n)
				gamma:l.scales
				beta:l.biases
				mean:l.rolling_mean
				variance:l.rolling_variance
				epsilon:0.00001f];
			cache.batchnorm = [[MPSCNNBatchNormalization alloc] initWithDevice:device dataSource:cache.batchnorm_source];
			if (!cache.batchnorm)
			{
				return false;
			}
		}

		cache.neuron = build_neuron(l.activation, device, &cache.leaky_alpha);

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

		cache.pool.edgeMode = MPSImageEdgeModeZero;
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
			cache.activation == l.activation;
	}

	bool build_connected_cache(MpsConnectedCache & cache, const Darknet::Layer & l, id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		cache.batchnorm = nil;
		cache.batchnorm_source = nil;
		cache.neuron = nil;
		cache.leaky_alpha = 0.0f;

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
			cache.batchnorm = [[MPSCNNBatchNormalization alloc] initWithDevice:device
				dataSource:cache.batchnorm_source];
			if (!cache.batchnorm)
			{
				return false;
			}
		}

		cache.neuron = build_neuron(l.activation, device, &cache.leaky_alpha);

		cache.outputs = l.outputs;
		cache.batch_normalize = (l.batch_normalize != 0);
		cache.activation = l.activation;
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

bool mps_is_available()
{
	return get_mps_context().ready;
}

bool mps_gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	std::scoped_lock<std::mutex> lock(get_mps_gemm_mutex());

	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		return false;
	}

	if (M <= 0 || N <= 0)
	{
		return true;
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

	const NSUInteger a_rows = (TA ? K : M);
	const NSUInteger a_cols = (TA ? M : K);
	const NSUInteger b_rows = (TB ? N : K);
	const NSUInteger b_cols = (TB ? K : N);
	const NSUInteger c_rows = static_cast<NSUInteger>(M);
	const NSUInteger c_cols = static_cast<NSUInteger>(N);

	const NSUInteger a_row_bytes = static_cast<NSUInteger>(lda) * sizeof(float);
	const NSUInteger b_row_bytes = static_cast<NSUInteger>(ldb) * sizeof(float);
	const NSUInteger c_row_bytes = static_cast<NSUInteger>(ldc) * sizeof(float);

	const NSUInteger a_bytes = a_rows * a_row_bytes;
	const NSUInteger b_bytes = b_rows * b_row_bytes;
	const NSUInteger c_bytes = c_rows * c_row_bytes;

	@autoreleasepool
	{
		if (!ensure_buffer(ctx.bufferA, ctx.bufferA_bytes, a_bytes, ctx.device) ||
			!ensure_buffer(ctx.bufferB, ctx.bufferB_bytes, b_bytes, ctx.device) ||
			!ensure_buffer(ctx.bufferC, ctx.bufferC_bytes, c_bytes, ctx.device))
		{
			return false;
		}

		std::memcpy([ctx.bufferA contents], A, a_bytes);
		std::memcpy([ctx.bufferB contents], B, b_bytes);
		std::memcpy([ctx.bufferC contents], C, c_bytes);

		if (!ctx.bufferA || !ctx.bufferB || !ctx.bufferC)
		{
			return false;
		}

		MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:a_rows
			columns:a_cols
			rowBytes:a_row_bytes
			dataType:MPSDataTypeFloat32];
		MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:b_rows
			columns:b_cols
			rowBytes:b_row_bytes
			dataType:MPSDataTypeFloat32];
		MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:c_rows
			columns:c_cols
			rowBytes:c_row_bytes
			dataType:MPSDataTypeFloat32];

		MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:ctx.bufferA descriptor:descA];
		MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:ctx.bufferB descriptor:descB];
		MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:ctx.bufferC descriptor:descC];

		id<MTLCommandBuffer> command_buffer = [ctx.queue commandBuffer];
		if (!command_buffer)
		{
			return false;
		}

		MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc]
			initWithDevice:ctx.device
			transposeLeft:(TA != 0)
			transposeRight:(TB != 0)
			resultRows:c_rows
			resultColumns:c_cols
			interiorColumns:static_cast<NSUInteger>(K)
			alpha:static_cast<double>(ALPHA)
			beta:static_cast<double>(BETA)];

		[gemm encodeToCommandBuffer:command_buffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
		[command_buffer commit];
		[command_buffer waitUntilCompleted];

		std::memcpy(C, [ctx.bufferC contents], c_bytes);
	}

	return true;
}

bool mps_layer_can_run(const Darknet::Layer & l, bool train)
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
			if (l.groups > 1 && !mps_supports_grouped_conv())
			{
				return false;
			}
			if (l.dilation != 1 && !mps_supports_dilated_conv())
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

const Darknet::Layer *mps_prev_layer(const Darknet::NetworkState &state)
{
	if (state.net.layers && state.index > 0)
	{
		return &state.net.layers[state.index - 1];
	}
	return nullptr;
}

bool mps_should_defer_readback(const Darknet::NetworkState &state)
{
	if (state.net.layers && (state.index + 1) < state.net.n)
	{
		const Darknet::Layer & next = state.net.layers[state.index + 1];
		return mps_layer_can_run(next, state.train);
	}
	return false;
}

void mps_flush_deferred_output(const Darknet::Layer *layer)
{
	if (layer && mps_is_output_deferred(layer))
	{
		mps_flush_output_if_needed(layer, layer->output);
	}
}

bool mps_is_output_deferred(const Darknet::Layer *layer)
{
	if (!layer)
	{
		return false;
	}

	auto & deferred = get_mps_deferred_layers();
	return deferred.find(layer) != deferred.end();
}

void mps_flush_output_if_needed(const Darknet::Layer *layer, float *output)
{
	if (!layer || !output)
	{
		return;
	}

	auto & deferred = get_mps_deferred_layers();
	auto it = deferred.find(layer);
	if (it == deferred.end())
	{
		return;
	}

	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		deferred.erase(it);
		return;
	}

	wait_mps_command_buffer_if_needed();

	MPSImage *output_image = get_cached_output_image(layer, layer->out_w, layer->out_h, layer->out_c, layer->batch);
	if (!output_image)
	{
		deferred.erase(it);
		return;
	}

	@autoreleasepool
	{
		read_mps_output_image(output_image, layer->out_w, layer->out_h, layer->out_c, layer->batch, output);
	}
	deferred.erase(it);
}

bool mps_convolution_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (activation_applied)
	{
		*activation_applied = false;
	}

	if (l.groups <= 0 || (l.c % l.groups) != 0 || (l.n % l.groups) != 0)
	{
		if (reason) *reason = "invalid groups";
		return false;
	}

	if (l.groups > 1 && !mps_supports_grouped_conv())
	{
		if (reason) *reason = "grouped convolution not supported";
		return false;
	}

	if (l.pad < 0 || l.pad > (l.size / 2))
	{
		if (reason) *reason = "unsupported padding";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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
			if (reason)
			{
				*reason = (l.dilation != 1) ? "dilation not supported" : "failed to build convolution";
			}
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
			if (reason) *reason = "failed to allocate MPSImage";
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
			if (reason) *reason = "failed to create command buffer";
			return false;
		}

		[cache.conv encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
		if (cache.batchnorm)
		{
			[cache.batchnorm encodeToCommandBuffer:command_buffer sourceImage:output_image destinationImage:output_image];
		}
		bool activation_done = false;
		const char *status = "ok";
		if (cache.neuron)
		{
			[cache.neuron encodeToCommandBuffer:command_buffer sourceImage:output_image destinationImage:output_image];
			activation_done = true;
		}
		else if (l.activation == LINEAR)
		{
			activation_done = true;
		}
		else
		{
			if (encode_mps_activation(command_buffer, output_image, l.activation, nullptr))
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
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}

		if (activation_applied)
		{
			*activation_applied = activation_done;
		}

		if (reason) *reason = status;

	}
	return true;
}

bool mps_maxpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (l.maxpool_depth || l.antialiasing)
	{
		if (reason) *reason = "unsupported maxpool mode";
		return false;
	}

	if (l.pad < 0)
	{
		if (reason) *reason = "padding not supported";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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
			if (reason) *reason = "failed to build pooling";
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
			if (reason) *reason = "failed to allocate MPSImage";
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
			if (reason) *reason = "failed to create command buffer";
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
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}

	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (l.antialiasing)
	{
		if (reason) *reason = "antialiasing not supported";
		return false;
	}

	if (l.pad < 0)
	{
		if (reason) *reason = "padding not supported";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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
			if (reason) *reason = "failed to build pooling";
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
			if (reason) *reason = "failed to allocate MPSImage";
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
			if (reason) *reason = "failed to create command buffer";
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
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}

	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_global_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (l.w <= 0 || l.h <= 0)
	{
		if (reason) *reason = "invalid avgpool size";
		return false;
	}

	if (l.c <= 0 || l.batch <= 0)
	{
		if (reason) *reason = "invalid channel/batch";
		return false;
	}

	if (l.out_w != 1 || l.out_h != 1)
	{
		if (reason) *reason = "not global avgpool";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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
			if (reason) *reason = "failed to build global avgpool";
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
			if (reason) *reason = "failed to allocate global avgpool images";
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
			if (reason) *reason = "failed to create command buffer";
			return false;
		}

		[cache.pool encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_reorg_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (l.stride <= 0)
	{
		if (reason) *reason = "invalid stride";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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
			if (reason) *reason = "failed to allocate reorg images";
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
			if (reason) *reason = "failed to create command buffer";
			return false;
		}

		id<MTLTexture> in_tex = input_image.texture;
		id<MTLTexture> out_tex = cache.output_image.texture;
		if (!in_tex || !out_tex)
		{
			if (reason) *reason = "missing reorg texture";
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
		const char *kernel_name = l.reverse ? "reorg_reverse_kernel" : "reorg_forward_kernel";
		if (!metal_dispatch_texture_kernel(kernel_name,
				(__bridge void *)out_tex,
				8, 8,
				textures, 2,
				nullptr, 0,
				bytes, 3,
				(__bridge void *)command_buffer))
		{
			if (reason) *reason = "failed to dispatch reorg kernel";
			return false;
		}

		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_connected_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
		return false;
	}

	if (l.inputs <= 0 || l.outputs <= 0 || l.batch <= 0)
	{
		if (reason) *reason = "invalid connected size";
		return false;
	}

	const bool needs_batchnorm = (l.batch_normalize != 0);
	const bool needs_activation = (l.activation != LINEAR);
	const bool activation_supported = mps_activation_supported(l.activation);
	if (!needs_batchnorm && !needs_activation)
	{
		if (reason) *reason = "no MPS work";
		return false;
	}
	if (!needs_batchnorm && needs_activation && !activation_supported)
	{
		if (reason) *reason = "activation not supported";
		return false;
	}

	if (prev && input == prev->output)
	{
		mps_flush_deferred_output(prev);
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
			if (reason) *reason = "failed to build connected cache";
			return false;
		}
	}

	@autoreleasepool
	{
		if (!ensure_connected_images(cache, l, ctx.device))
		{
			if (reason) *reason = "failed to allocate connected images";
			return false;
		}

		MPSImage *output_image = cache.output_image;
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
			[output_image writeBytes:output_ptr
				dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
				bytesPerRow:output_bytes_per_row
				region:out_region
				featureChannelInfo:out_params
				imageIndex:b];
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			if (reason) *reason = "failed to create command buffer";
			return false;
		}

		if (cache.batchnorm)
		{
			[cache.batchnorm encodeToCommandBuffer:command_buffer sourceImage:output_image destinationImage:output_image];
		}

		bool activation_done = false;
		const char *status = "ok";
		if (cache.neuron)
		{
			[cache.neuron encodeToCommandBuffer:command_buffer sourceImage:output_image destinationImage:output_image];
			activation_done = true;
		}
		else if (l.activation == LINEAR)
		{
			activation_done = true;
		}
		else
		{
			if (encode_mps_activation(command_buffer, output_image, l.activation, nullptr))
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
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}

		if (activation_applied)
		{
			*activation_applied = activation_done;
		}

		if (reason) *reason = status;
	}

	return true;
}

bool mps_softmax_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (l.softmax_tree || l.spatial)
	{
		if (reason) *reason = "softmax mode not supported";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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

	if (cache.bytes != bytes || !cache.input_buffer.handle || !cache.output_buffer.handle)
	{
		metal_buffer_free(&cache.input_buffer);
		metal_buffer_free(&cache.output_buffer);
		if (!metal_buffer_alloc(bytes, &cache.input_buffer) || !metal_buffer_alloc(bytes, &cache.output_buffer))
		{
			if (reason) *reason = "failed to allocate softmax buffers";
			return false;
		}
		cache.bytes = bytes;
	}

	float *input_ptr = nullptr;
	if (cache.input_buffer.handle)
	{
		id<MTLBuffer> buf = (__bridge id<MTLBuffer>)cache.input_buffer.handle;
		input_ptr = static_cast<float *>([buf contents]);
	}
	if (!input_ptr)
	{
		if (reason) *reason = "softmax input map failed";
		return false;
	}

	bool filled_from_mps = false;
	if (prev && input == prev->output)
	{
		MPSImage *image = get_cached_output_image(prev, prev->out_w, prev->out_h, prev->out_c, prev->batch);
		if (image)
		{
			if (!read_mps_output_image(image, prev->out_w, prev->out_h, prev->out_c, prev->batch, input_ptr))
			{
				if (reason) *reason = "failed to read MPS softmax input";
				return false;
			}
			filled_from_mps = true;
		}
	}
	if (!filled_from_mps)
	{
		if (!metal_buffer_upload(&cache.input_buffer, 0, input, bytes))
		{
			if (reason) *reason = "failed to upload softmax input";
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
		if (reason) *reason = "softmax size invalid";
		return false;
	}
	const size_t threads_per_group = (threads < 256) ? threads : 256;

	MetalDispatchBuffer buffers[] = {
		{ &cache.input_buffer, 0 },
		{ &cache.output_buffer, 0 }
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
		if (reason) *reason = "failed to dispatch softmax";
		return false;
	}

	if (!metal_buffer_download(&cache.output_buffer, 0, output, bytes))
	{
		if (reason) *reason = "failed to download softmax output";
		return false;
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_yolo_activate(const Darknet::Layer & l, const float *input, float *output, const char **reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
		return false;
	}

	const size_t bytes = static_cast<size_t>(l.outputs) * static_cast<size_t>(l.batch) * sizeof(float);
	auto & cache_map = get_mps_yolo_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsYoloCache>();
	}
	auto & cache = *cache_ptr;

	if (cache.bytes != bytes || !cache.buffer.handle)
	{
		metal_buffer_free(&cache.buffer);
		if (!metal_buffer_alloc(bytes, &cache.buffer))
		{
			if (reason) *reason = "failed to allocate yolo buffer";
			return false;
		}
		cache.bytes = bytes;
	}

	if (!metal_buffer_upload(&cache.buffer, 0, input, bytes))
	{
		if (reason) *reason = "failed to upload yolo input";
		return false;
	}

	const uint32_t entries = static_cast<uint32_t>(l.classes + 5);
	const uint32_t wh = static_cast<uint32_t>(l.w * l.h);
	const uint32_t new_coords = static_cast<uint32_t>(l.new_coords ? 1 : 0);
	const float scale_x_y = l.scale_x_y;
	const float bias = -0.5f * (scale_x_y - 1.0f);
	const size_t total = static_cast<size_t>(entries) * static_cast<size_t>(wh) *
		static_cast<size_t>(l.n) * static_cast<size_t>(l.batch);
	if (total == 0 || total > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
	{
		if (reason) *reason = "yolo size invalid";
		return false;
	}
	const uint32_t total_u = static_cast<uint32_t>(total);

	const size_t threads_per_group = (total < 256) ? total : 256;
	MetalDispatchBuffer buffers[] = { { &cache.buffer, 0 } };
	MetalDispatchBytes bytes_args[] = {
		{ &entries, sizeof(entries) },
		{ &wh, sizeof(wh) },
		{ &scale_x_y, sizeof(scale_x_y) },
		{ &bias, sizeof(bias) },
		{ &new_coords, sizeof(new_coords) },
		{ &total_u, sizeof(total_u) }
	};
	if (!metal_dispatch_1d("yolo_activate_kernel",
			total, threads_per_group,
			nullptr, 0,
			buffers, 1,
			bytes_args, 6,
			nullptr))
	{
		if (reason) *reason = "failed to dispatch yolo kernel";
		return false;
	}

	if (!metal_buffer_download(&cache.buffer, 0, output, bytes))
	{
		if (reason) *reason = "failed to download yolo output";
		return false;
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_yolo_decode_boxes(const Darknet::Layer & l, const float *input, int netw, int neth, float *boxes, const char **reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (!input || !boxes)
	{
		if (reason) *reason = "null input/output";
		return false;
	}

	if (l.w <= 0 || l.h <= 0 || l.n <= 0 || l.batch <= 0 || l.outputs <= 0 || netw <= 0 || neth <= 0)
	{
		if (reason) *reason = "invalid yolo dimensions";
		return false;
	}

	const size_t input_bytes = static_cast<size_t>(l.outputs) * static_cast<size_t>(l.batch) * sizeof(float);
	const size_t box_count = static_cast<size_t>(l.w) * static_cast<size_t>(l.h) * static_cast<size_t>(l.n) * static_cast<size_t>(l.batch);
	const size_t output_bytes = box_count * sizeof(float) * 4;

	if (box_count == 0 || box_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
	{
		if (reason) *reason = "yolo decode size invalid";
		return false;
	}

	auto & cache_map = get_mps_yolo_decode_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsYoloDecodeCache>();
	}
	auto & cache = *cache_ptr;

	if (cache.input_bytes != input_bytes || !cache.input_buffer.handle)
	{
		metal_buffer_free(&cache.input_buffer);
		if (!metal_buffer_alloc(input_bytes, &cache.input_buffer))
		{
			if (reason) *reason = "failed to allocate yolo decode input buffer";
			return false;
		}
		cache.input_bytes = input_bytes;
	}

	if (cache.output_bytes != output_bytes || !cache.output_buffer.handle)
	{
		metal_buffer_free(&cache.output_buffer);
		if (!metal_buffer_alloc(output_bytes, &cache.output_buffer))
		{
			if (reason) *reason = "failed to allocate yolo decode output buffer";
			return false;
		}
		cache.output_bytes = output_bytes;
	}

	const size_t bias_bytes = static_cast<size_t>(l.total) * 2u * sizeof(float);
	if (cache.bias_bytes != bias_bytes || !cache.biases_buffer.handle)
	{
		metal_buffer_free(&cache.biases_buffer);
		if (!metal_buffer_alloc(bias_bytes, &cache.biases_buffer))
		{
			if (reason) *reason = "failed to allocate yolo biases buffer";
			return false;
		}
		cache.bias_bytes = bias_bytes;
		if (!metal_buffer_upload(&cache.biases_buffer, 0, l.biases, bias_bytes))
		{
			if (reason) *reason = "failed to upload yolo biases";
			return false;
		}
	}

	const size_t mask_bytes = static_cast<size_t>(l.n) * sizeof(int);
	if (cache.mask_bytes != mask_bytes || !cache.mask_buffer.handle)
	{
		metal_buffer_free(&cache.mask_buffer);
		if (!metal_buffer_alloc(mask_bytes, &cache.mask_buffer))
		{
			if (reason) *reason = "failed to allocate yolo mask buffer";
			return false;
		}
		cache.mask_bytes = mask_bytes;
		if (!metal_buffer_upload(&cache.mask_buffer, 0, l.mask, mask_bytes))
		{
			if (reason) *reason = "failed to upload yolo mask";
			return false;
		}
	}

	if (!metal_buffer_upload(&cache.input_buffer, 0, input, input_bytes))
	{
		if (reason) *reason = "failed to upload yolo decode input";
		return false;
	}

	const uint32_t w = static_cast<uint32_t>(l.w);
	const uint32_t h = static_cast<uint32_t>(l.h);
	const uint32_t entries = static_cast<uint32_t>(l.classes + 5);
	const uint32_t n = static_cast<uint32_t>(l.n);
	const uint32_t batch = static_cast<uint32_t>(l.batch);
	const uint32_t outputs = static_cast<uint32_t>(l.outputs);
	const uint32_t netw_u = static_cast<uint32_t>(netw);
	const uint32_t neth_u = static_cast<uint32_t>(neth);
	const uint32_t new_coords = static_cast<uint32_t>(l.new_coords ? 1 : 0);
	const size_t threads = box_count;
	const size_t threads_per_group = (threads < 256) ? threads : 256;

	MetalDispatchBuffer buffers[] = {
		{ &cache.input_buffer, 0 },
		{ &cache.output_buffer, 0 },
		{ &cache.biases_buffer, 0 },
		{ &cache.mask_buffer, 0 }
	};
	MetalDispatchBytes bytes_args[] = {
		{ &w, sizeof(w) },
		{ &h, sizeof(h) },
		{ &entries, sizeof(entries) },
		{ &n, sizeof(n) },
		{ &batch, sizeof(batch) },
		{ &outputs, sizeof(outputs) },
		{ &netw_u, sizeof(netw_u) },
		{ &neth_u, sizeof(neth_u) },
		{ &new_coords, sizeof(new_coords) }
	};

	if (!metal_dispatch_1d("yolo_decode_boxes_kernel",
			threads, threads_per_group,
			nullptr, 0,
			buffers, 4,
			bytes_args, 9,
			nullptr))
	{
		if (reason) *reason = "failed to dispatch yolo decode kernel";
		return false;
	}

	if (!metal_buffer_download(&cache.output_buffer, 0, boxes, output_bytes))
	{
		if (reason) *reason = "failed to download yolo boxes";
		return false;
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_yolo_collect_candidates(const Darknet::Layer & l, const float *input, float thresh,
	uint32_t *indices, uint32_t max_candidates, uint32_t *count, const char **reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (!input || !indices || !count)
	{
		if (reason) *reason = "null input/output";
		return false;
	}

	if (l.w <= 0 || l.h <= 0 || l.n <= 0 || l.batch <= 0 || l.outputs <= 0)
	{
		if (reason) *reason = "invalid yolo dimensions";
		return false;
	}

	const size_t total = static_cast<size_t>(l.w) * static_cast<size_t>(l.h) *
		static_cast<size_t>(l.n) * static_cast<size_t>(l.batch);
	if (total == 0 || total > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
	{
		if (reason) *reason = "yolo candidate size invalid";
		return false;
	}

	const uint32_t total_u = static_cast<uint32_t>(total);
	if (max_candidates == 0 || max_candidates > total_u)
	{
		max_candidates = total_u;
	}

	const size_t input_bytes = static_cast<size_t>(l.outputs) * static_cast<size_t>(l.batch) * sizeof(float);
	const size_t indices_bytes = static_cast<size_t>(max_candidates) * sizeof(uint32_t);

	auto & cache_map = get_mps_yolo_candidates_cache();
	auto & cache_ptr = cache_map[&l];
	if (!cache_ptr)
	{
		cache_ptr = std::make_unique<MpsYoloCandidatesCache>();
	}
	auto & cache = *cache_ptr;

	if (cache.input_bytes != input_bytes || !cache.input_buffer.handle)
	{
		metal_buffer_free(&cache.input_buffer);
		if (!metal_buffer_alloc(input_bytes, &cache.input_buffer))
		{
			if (reason) *reason = "failed to allocate yolo candidates input buffer";
			return false;
		}
		cache.input_bytes = input_bytes;
	}

	if (cache.indices_bytes != indices_bytes || !cache.indices_buffer.handle)
	{
		metal_buffer_free(&cache.indices_buffer);
		if (!metal_buffer_alloc(indices_bytes, &cache.indices_buffer))
		{
			if (reason) *reason = "failed to allocate yolo candidates buffer";
			return false;
		}
		cache.indices_bytes = indices_bytes;
	}

	if (!cache.count_buffer.handle)
	{
		if (!metal_buffer_alloc(sizeof(uint32_t), &cache.count_buffer))
		{
			if (reason) *reason = "failed to allocate yolo count buffer";
			return false;
		}
	}

	if (!metal_buffer_upload(&cache.input_buffer, 0, input, input_bytes))
	{
		if (reason) *reason = "failed to upload yolo candidate input";
		return false;
	}

	uint32_t zero = 0;
	if (!metal_buffer_upload(&cache.count_buffer, 0, &zero, sizeof(zero)))
	{
		if (reason) *reason = "failed to reset yolo count buffer";
		return false;
	}

	const uint32_t w = static_cast<uint32_t>(l.w);
	const uint32_t h = static_cast<uint32_t>(l.h);
	const uint32_t entries = static_cast<uint32_t>(l.classes + 5);
	const uint32_t n = static_cast<uint32_t>(l.n);
	const uint32_t batch = static_cast<uint32_t>(l.batch);
	const uint32_t outputs = static_cast<uint32_t>(l.outputs);

	const size_t threads = total;
	const size_t threads_per_group = (threads < 256) ? threads : 256;

	MetalDispatchBuffer buffers[] = {
		{ &cache.input_buffer, 0 },
		{ &cache.indices_buffer, 0 },
		{ &cache.count_buffer, 0 }
	};
	MetalDispatchBytes bytes_args[] = {
		{ &w, sizeof(w) },
		{ &h, sizeof(h) },
		{ &entries, sizeof(entries) },
		{ &n, sizeof(n) },
		{ &batch, sizeof(batch) },
		{ &outputs, sizeof(outputs) },
		{ &thresh, sizeof(thresh) },
		{ &max_candidates, sizeof(max_candidates) }
	};

	if (!metal_dispatch_1d("yolo_candidates_kernel",
			threads, threads_per_group,
			nullptr, 0,
			buffers, 3,
			bytes_args, 8,
			nullptr))
	{
		if (reason) *reason = "failed to dispatch yolo candidates kernel";
		return false;
	}

	uint32_t found = 0;
	if (!metal_buffer_download(&cache.count_buffer, 0, &found, sizeof(found)))
	{
		if (reason) *reason = "failed to download yolo candidates count";
		return false;
	}
	if (found > max_candidates)
	{
		found = max_candidates;
	}

	*count = found;
	if (found > 0)
	{
		const size_t download_bytes = static_cast<size_t>(found) * sizeof(uint32_t);
		if (!metal_buffer_download(&cache.indices_buffer, 0, indices, download_bytes))
		{
			if (reason) *reason = "failed to download yolo candidates";
			return false;
		}
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_nms_suppress(const Darknet::Box *boxes, float *scores, const uint32_t *order,
	uint32_t order_count, uint32_t total, float thresh, const char **reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (!boxes || !scores || !order)
	{
		if (reason) *reason = "null boxes/scores/order";
		return false;
	}

	auto & cache = get_mps_nms_cache();
	const size_t boxes_bytes = static_cast<size_t>(total) * sizeof(float) * 4;
	const size_t scores_bytes = static_cast<size_t>(total) * sizeof(float);
	const size_t order_bytes = static_cast<size_t>(order_count) * sizeof(uint32_t);

	if (cache.boxes_bytes != boxes_bytes || !cache.boxes_buffer.handle)
	{
		metal_buffer_free(&cache.boxes_buffer);
		if (!metal_buffer_alloc(boxes_bytes, &cache.boxes_buffer))
		{
			if (reason) *reason = "failed to allocate nms boxes buffer";
			return false;
		}
		cache.boxes_bytes = boxes_bytes;
	}

	if (cache.scores_bytes != scores_bytes || !cache.scores_buffer.handle)
	{
		metal_buffer_free(&cache.scores_buffer);
		if (!metal_buffer_alloc(scores_bytes, &cache.scores_buffer))
		{
			if (reason) *reason = "failed to allocate nms scores buffer";
			return false;
		}
		cache.scores_bytes = scores_bytes;
	}

	if (cache.order_bytes < order_bytes || !cache.order_buffer.handle)
	{
		metal_buffer_free(&cache.order_buffer);
		if (!metal_buffer_alloc(order_bytes, &cache.order_buffer))
		{
			if (reason) *reason = "failed to allocate nms order buffer";
			return false;
		}
		cache.order_bytes = order_bytes;
	}

	if (!metal_buffer_upload(&cache.boxes_buffer, 0, boxes, boxes_bytes))
	{
		if (reason) *reason = "failed to upload nms boxes";
		return false;
	}
	if (!metal_buffer_upload(&cache.scores_buffer, 0, scores, scores_bytes))
	{
		if (reason) *reason = "failed to upload nms scores";
		return false;
	}
	if (!metal_buffer_upload(&cache.order_buffer, 0, order, order_bytes))
	{
		if (reason) *reason = "failed to upload nms order";
		return false;
	}

	const uint32_t count = order_count;
	if (count < 2)
	{
		if (reason) *reason = "ok";
		return true;
	}

	MetalDispatchBuffer buffers[] = {
		{ &cache.boxes_buffer, 0 },
		{ &cache.scores_buffer, 0 },
		{ &cache.order_buffer, 0 }
	};

	for (uint32_t base = 0; base + 1 < count; ++base)
	{
		const size_t threads = static_cast<size_t>(count - base - 1u);
		const size_t threads_per_group = (threads < 256) ? threads : 256;
		const uint32_t base_u = base;
		MetalDispatchBytes bytes_args[] = {
			{ &count, sizeof(count) },
			{ &base_u, sizeof(base_u) },
			{ &thresh, sizeof(thresh) }
		};
		if (!metal_dispatch_1d("nms_suppress_kernel",
				threads, threads_per_group,
				nullptr, 0,
				buffers, 3,
				bytes_args, 3,
				nullptr))
		{
			if (reason) *reason = "failed to dispatch nms kernel";
			return false;
		}
	}

	if (!metal_buffer_download(&cache.scores_buffer, 0, scores, scores_bytes))
	{
		if (reason) *reason = "failed to download nms scores";
		return false;
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_nms_sort(DarknetDetection *dets, int total, int classes, float thresh, const char **reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (!dets || total <= 0 || classes <= 0)
	{
		if (reason) *reason = "invalid nms inputs";
		return false;
	}

	std::vector<Darknet::Box> boxes;
	boxes.reserve(static_cast<size_t>(total));
	for (int i = 0; i < total; ++i)
	{
		boxes.push_back(dets[i].bbox);
	}

	std::vector<float> scores(static_cast<size_t>(total), 0.0f);
	std::vector<uint32_t> order;
	order.reserve(static_cast<size_t>(total));

	for (int k = 0; k < classes; ++k)
	{
		order.clear();
		bool any = false;
		for (int i = 0; i < total; ++i)
		{
			float v = 0.0f;
			if (dets[i].prob)
			{
				v = dets[i].prob[k];
			}
			scores[static_cast<size_t>(i)] = v;
			if (v > 0.0f)
			{
				any = true;
				order.push_back(static_cast<uint32_t>(i));
			}
		}

		if (!any)
		{
			continue;
		}

		std::sort(order.begin(), order.end(),
			[&scores](uint32_t a, uint32_t b)
			{
				return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
			});

		if (!mps_nms_suppress(boxes.data(), scores.data(), order.data(),
				static_cast<uint32_t>(order.size()),
				static_cast<uint32_t>(total), thresh, reason))
		{
			return false;
		}

		for (int i = 0; i < total; ++i)
		{
			if (dets[i].prob)
			{
				dets[i].prob[k] = scores[static_cast<size_t>(i)];
			}
		}
	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const Darknet::Layer *from,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (activation_applied)
	{
		*activation_applied = false;
	}

	if (!from || l.n != 1 || !l.input_sizes || l.input_sizes[0] != l.outputs)
	{
		if (reason) *reason = "unsupported shortcut";
		return false;
	}

	const bool has_weights = (l.nweights > 0 && l.weights && l.weights_type != NO_WEIGHTS);
	const bool norm_supported = (l.weights_normalization == NO_NORMALIZATION ||
		l.weights_normalization == RELU_NORMALIZATION ||
		l.weights_normalization == SOFTMAX_NORMALIZATION);
	const bool per_feature = (l.weights_type == PER_FEATURE && l.nweights == (l.n + 1) &&
		norm_supported);
	const bool per_channel = (l.weights_type == PER_CHANNEL && l.nweights == (l.n + 1) * l.c &&
		norm_supported);
	const bool supported_weights = (!has_weights) || per_feature || per_channel;
	if (!supported_weights)
	{
		if (reason) *reason = "unsupported shortcut weights";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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
		if (reason) *reason = "missing MPS input";
		return false;
	}

	if (!has_weights)
	{
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
				if (reason) *reason = "failed to build add";
				return false;
			}
		}

		@autoreleasepool
		{
			if (!ensure_add_output_image(cache, l, ctx.device))
			{
				if (reason) *reason = "failed to allocate MPSImage";
				return false;
			}

			id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
			if (!command_buffer)
			{
				if (reason) *reason = "failed to create command buffer";
				return false;
			}

			[cache.add encodeToCommandBuffer:command_buffer
				primaryImage:primary_image
				secondaryImage:secondary_image
				destinationImage:cache.output_image];
			bool activation_done = false;
			const char *status = "ok";
			if (cache.neuron)
			{
				[cache.neuron encodeToCommandBuffer:command_buffer sourceImage:cache.output_image destinationImage:cache.output_image];
				activation_done = true;
			}
		else if (l.activation == LINEAR)
		{
			activation_done = true;
		}
		else
		{
			if (encode_mps_activation(command_buffer, cache.output_image, l.activation, nullptr))
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
				wait_mps_command_buffer_if_needed();
				read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
			}
			else
			{
				deferred.insert(&l);
			}

			if (activation_applied)
			{
				*activation_applied = activation_done;
			}

			if (reason) *reason = status;

		}

		return true;
	}

	auto & weighted_cache_map = get_mps_weighted_add_cache();
	auto & weighted_cache_ptr = weighted_cache_map[&l];
	if (!weighted_cache_ptr)
	{
		weighted_cache_ptr = std::make_unique<MpsWeightedAddCache>();
	}
	auto & weighted_cache = *weighted_cache_ptr;
	if (!cache_matches_weighted_add(weighted_cache, l))
	{
		if (!build_weighted_add_cache(weighted_cache, l, ctx.device))
		{
			if (reason) *reason = "failed to build weighted add";
			return false;
		}
	}

	@autoreleasepool
	{
		if (!ensure_weighted_add_output_image(weighted_cache, l, ctx.device))
		{
			if (reason) *reason = "failed to allocate MPSImage";
			return false;
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			if (reason) *reason = "failed to create command buffer";
			return false;
		}

		id<MTLTexture> in_tex = primary_image.texture;
		id<MTLTexture> add_tex = secondary_image.texture;
		id<MTLTexture> out_tex = weighted_cache.output_image.texture;
		if (!in_tex || !add_tex || !out_tex)
		{
			if (reason) *reason = "missing shortcut texture";
			return false;
		}

	const float w_in = l.weights[0];
	const float w_add = l.weights[1];
	const NSUInteger width = out_tex.width;
	const NSUInteger height = out_tex.height;
	const NSUInteger depth = out_tex.arrayLength;
	if (width == 0 || height == 0 || depth == 0)
	{
		if (reason) *reason = "invalid shortcut texture size";
		return false;
	}

	MetalDispatchTexture textures[] = {
		{ (__bridge void *)in_tex },
		{ (__bridge void *)add_tex },
		{ (__bridge void *)out_tex }
	};
	const char *kernel_name = nullptr;
	if (per_channel)
	{
		const int channels = l.out_c;
		if (!weighted_cache.weights_ready || weighted_cache.weights_ptr != l.weights ||
			weighted_cache.weights_channels != channels)
		{
			metal_buffer_free(&weighted_cache.weights_in);
			metal_buffer_free(&weighted_cache.weights_add);
			const size_t bytes = static_cast<size_t>(channels) * sizeof(float);
			if (!metal_buffer_alloc(bytes, &weighted_cache.weights_in) ||
				!metal_buffer_alloc(bytes, &weighted_cache.weights_add))
			{
				if (reason) *reason = "failed to allocate shortcut weights";
				return false;
			}
			if (!metal_buffer_upload(&weighted_cache.weights_in, 0, l.weights, bytes) ||
				!metal_buffer_upload(&weighted_cache.weights_add, 0, l.weights + channels, bytes))
			{
				if (reason) *reason = "failed to upload shortcut weights";
				return false;
			}
			weighted_cache.weights_ptr = l.weights;
			weighted_cache.weights_channels = channels;
			weighted_cache.weights_ready = true;
		}

		if (l.weights_normalization == RELU_NORMALIZATION)
		{
			kernel_name = "shortcut_per_channel_relu_kernel";
		}
		else if (l.weights_normalization == SOFTMAX_NORMALIZATION)
		{
			kernel_name = "shortcut_per_channel_softmax_kernel";
		}
		else
		{
			kernel_name = "shortcut_per_channel_kernel";
		}

		MetalDispatchBuffer buffers[] = {
			{ &weighted_cache.weights_in, 0 },
			{ &weighted_cache.weights_add, 0 }
		};
		const uint32_t channels_u = static_cast<uint32_t>(channels);
		MetalDispatchBytes bytes_args[] = {
			{ &channels_u, sizeof(channels_u) }
		};
		if (!metal_dispatch_2d(kernel_name,
				static_cast<size_t>(width),
				static_cast<size_t>(height),
				8, 8,
				textures, 3,
				buffers, 2,
				bytes_args, 1,
				(__bridge void *)command_buffer))
		{
			if (reason) *reason = "failed to dispatch per-channel shortcut";
			return false;
		}
	}
	else
	{
		if (l.weights_normalization == RELU_NORMALIZATION)
		{
			kernel_name = "shortcut_weighted_relu_kernel";
		}
		else if (l.weights_normalization == SOFTMAX_NORMALIZATION)
		{
			kernel_name = "shortcut_weighted_softmax_kernel";
		}
		else
		{
			kernel_name = "shortcut_weighted_kernel";
		}

		MetalDispatchBytes bytes_args[] = {
			{ &w_in, sizeof(w_in) },
			{ &w_add, sizeof(w_add) }
		};
		if (!metal_dispatch_2d(kernel_name,
				static_cast<size_t>(width),
				static_cast<size_t>(height),
				8, 8,
				textures, 3,
				nullptr, 0,
				bytes_args, 2,
				(__bridge void *)command_buffer))
		{
			if (reason) *reason = "failed to dispatch weighted shortcut";
			return false;
		}
	}

		bool activation_done = false;
		const char *status = "ok";
		if (weighted_cache.neuron)
		{
			[weighted_cache.neuron encodeToCommandBuffer:command_buffer sourceImage:weighted_cache.output_image destinationImage:weighted_cache.output_image];
			activation_done = true;
		}
		else if (l.activation == LINEAR)
		{
			activation_done = true;
		}
		else
		{
			if (encode_mps_activation(command_buffer, weighted_cache.output_image, l.activation, nullptr))
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
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(weighted_cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}

		if (activation_applied)
		{
			*activation_applied = activation_done;
		}

		if (reason) *reason = status;
	}

	return true;
}

bool mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
	float *output, bool defer_readback, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (!output)
	{
		if (reason) *reason = "null output";
		return false;
	}

	if (!net.layers || net.n <= 0)
	{
		if (reason) *reason = "missing network layers";
		return false;
	}

	if (l.n <= 0 || !l.input_layers || !l.input_sizes)
	{
		if (reason) *reason = "missing route inputs";
		return false;
	}

	if (l.groups <= 0 || l.group_id < 0 || l.group_id >= l.groups)
	{
		if (reason) *reason = "invalid route groups";
		return false;
	}

	if (l.out_c <= 0)
	{
		if (reason) *reason = "route channels invalid";
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
			if (reason) *reason = "route input out of range";
			return false;
		}
		const Darknet::Layer & src = net.layers[index];
		if (src.out_w != l.out_w || src.out_h != l.out_h)
		{
			if (reason) *reason = "route input size mismatch";
			return false;
		}
		if (src.batch != l.batch)
		{
			if (reason) *reason = "route input batch mismatch";
			return false;
		}
		if (src.out_c <= 0)
		{
			if (reason) *reason = "route input channels invalid";
			return false;
		}
		if ((src.out_c % l.groups) != 0)
		{
			if (reason) *reason = "route input groups mismatch";
			return false;
		}

		const int part_c = src.out_c / l.groups;
		if (part_c <= 0)
		{
			if (reason) *reason = "route group channels invalid";
			return false;
		}

		MPSImage *image = get_cached_output_image(&src, src.out_w, src.out_h, src.out_c, src.batch);
		if (!image)
		{
			if (reason) *reason = "missing MPS input";
			return false;
		}
		const int input_offset = part_c * l.group_id;
		inputs.push_back({ &src, image, src.out_c, input_offset, part_c, total_channels });
		total_channels += part_c;
	}

	if (total_channels != l.out_c)
	{
		if (reason) *reason = "route channel mismatch";
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
			if (reason) *reason = "failed to allocate route output";
			return false;
		}

		id<MTLTexture> dst_texture = cache.output_image.texture;
		if (!dst_texture)
		{
			if (reason) *reason = "missing output texture";
			return false;
		}

		for (const auto & entry : inputs)
		{
			if (!entry.image || !entry.image.texture)
			{
				if (reason) *reason = "missing input texture";
				return false;
			}
		}

		id<MTLCommandBuffer> command_buffer = create_mps_command_buffer(ctx);
		if (!command_buffer)
		{
			if (reason) *reason = "failed to create command buffer";
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
			if (reason) *reason = "failed to clear route output";
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
				if (reason) *reason = "failed to encode route copy";
				return false;
			}
		}
		[command_buffer commit];
		track_mps_command_buffer(command_buffer);

		if (!defer_readback)
		{
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}

	}

	if (reason) *reason = "ok";
	return true;
}

bool mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason)
{
	auto & ctx = get_mps_context();
	auto & deferred = get_mps_deferred_layers();
	deferred.erase(&l);
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (l.reverse || l.stride <= 0)
	{
		if (reason) *reason = "unsupported upsample";
		return false;
	}

	if (!input || !output)
	{
		if (reason) *reason = "null input/output";
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
			if (reason) *reason = "failed to allocate upsample images";
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
			if (reason) *reason = "failed to create command buffer";
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
			wait_mps_command_buffer_if_needed();
			read_mps_output_image(cache.output_image, l.out_w, l.out_h, l.out_c, l.batch, output);
		}
		else
		{
			deferred.insert(&l);
		}

	}

	if (reason && !*reason) *reason = "ok";
	return true;
}

void Darknet::show_mps_info()
{
	TAT(TATPARMS);

	auto & cfg_and_state = Darknet::CfgAndState::get();
	auto & ctx = get_mps_context();

	if (!ctx.ready)
	{
		*cfg_and_state.output << Darknet::in_colour(Darknet::EColour::kBrightRed, "Apple MPS not available") << std::endl;
		return;
	}

	NSString *device_name = [ctx.device name];
	const char *name = device_name ? [device_name UTF8String] : "unknown";
	*cfg_and_state.output << "Apple MPS device: " << Darknet::in_colour(Darknet::EColour::kBrightGreen, name);

	if ([ctx.device respondsToSelector:@selector(recommendedMaxWorkingSetSize)])
	{
		size_t working_set = static_cast<size_t>([ctx.device recommendedMaxWorkingSetSize]);
		*cfg_and_state.output << ", " << Darknet::in_colour(Darknet::EColour::kYellow, size_to_IEC_string(working_set));
	}

	*cfg_and_state.output << std::endl;

	#error "denizz please remove calls to getenv()"
	const char *self_test = std::getenv("DARKNET_METAL_SELF_TEST");
	if (self_test && self_test[0] == '1')
	{
		const bool ok = metal_self_test();
		*cfg_and_state.output << "Metal backend self-test: "
			<< (ok ? "OK" : "FAILED") << std::endl;
	}
}

#endif // DARKNET_USE_MPS
