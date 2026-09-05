#include "metal_backend.hpp"
#include "apple_backend.hpp"
#include "apple_objc.hpp"

#ifdef DARKNET_USE_MPS

#import <Metal/Metal.h>
#import <CoreFoundation/CoreFoundation.h>

#include <mutex>
#include <unordered_map>
#include <string>
#include <cstring>

namespace Darknet::Apple
{
	namespace
	{
	using Darknet::Apple::Detail::ObjcStrong;
	using Darknet::Apple::Detail::objc_release_if_needed;

	/// Non-owning references to the canonical device/queue shared across threads.
		struct MetalContext
		{
			__unsafe_unretained id<MTLDevice> device = nil;
			__unsafe_unretained id<MTLCommandQueue> queue = nil;
			std::mutex mutex;
		};

		MetalContext & get_context()
		{
			static MetalContext ctx;
			return ctx;
		}

		/// Thread-local command buffer state for batched dispatch.
		struct MetalThreadContext
		{
			id<MTLCommandBuffer> command_buffer = nil;
			bool owns_command_buffer = false;
			std::string error;
		};

		thread_local MetalThreadContext thread_ctx;

		/// Thread-safe cache of the Metal library and pipeline states.
		struct MetalKernelCache
		{
			id<MTLLibrary> library = nil;
			std::unordered_map<std::string, ObjcStrong> pipelines;
			std::mutex mutex;
			bool failed = false;
			std::string error;
		};

		std::string ns_error_string(NSError *error)
		{
			if (not error)
			{
				return {};
			}
			NSString *description = error.localizedDescription;
			const char *text = description.UTF8String;
			return text ? text : "unknown Metal error";
		}

		bool command_buffer_succeeded(id<MTLCommandBuffer> command_buffer)
		{
			if (command_buffer and command_buffer.status == MTLCommandBufferStatusCompleted)
			{
				return true;
			}
			thread_ctx.error = command_buffer
				? "Metal command buffer failed: " + ns_error_string(command_buffer.error)
				: "Metal command buffer is unavailable";
			return false;
		}

		MetalKernelCache & get_kernel_cache()
		{
			static MetalKernelCache cache;
			return cache;
		}

		static const char *kMetalKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline float swish_fn(float x)
{
	return x / (1.0f + exp(-x));
}

inline float softplus_fn(float x)
{
	const float threshold = 20.0f;
	if (x > threshold)
	{
		return x;
	}
	if (x < -threshold)
	{
		return exp(x);
	}
	return log(exp(x) + 1.0f);
}

inline float mish_fn(float x)
{
	return x * tanh(softplus_fn(x));
}

inline float hard_mish_fn(float x)
{
	if (x > 0.0f)
	{
		return x;
	}
	if (x > -2.0f)
	{
		return x * x * 0.5f + x;
	}
	return 0.0f;
}

kernel void swish_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = swish_fn(v.x);
	v.y = swish_fn(v.y);
	v.z = swish_fn(v.z);
	v.w = swish_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void relu_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	tex.write(max(tex.read(uint2(gid.xy), gid.z), 0.0f), uint2(gid.xy), gid.z);
}

kernel void leaky_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v = select(0.1f * v, v, v > 0.0f);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void mish_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = mish_fn(v.x);
	v.y = mish_fn(v.y);
	v.z = mish_fn(v.z);
	v.w = mish_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void hard_mish_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = hard_mish_fn(v.x);
	v.y = hard_mish_fn(v.y);
	v.z = hard_mish_fn(v.z);
	v.w = hard_mish_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void upsample_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::write> out_tex [[texture(1)]],
	constant uint &stride [[buffer(0)]],
	constant float &scale [[buffer(1)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	uint sx = gid.x / stride;
	uint sy = gid.y / stride;
	float4 v = in_tex.read(uint2(sx, sy), gid.z);
	v *= scale;
	out_tex.write(v, uint2(gid.xy), gid.z);
}

inline float logistic_fn(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

inline float loggy_fn(float x)
{
	return 2.0f / (1.0f + exp(-x)) - 1.0f;
}

inline float relu6_fn(float x)
{
	return clamp(x, 0.0f, 6.0f);
}

inline float elu_fn(float x)
{
	return (x >= 0.0f) ? x : (exp(x) - 1.0f);
}

inline float selu_fn(float x)
{
	return (x >= 0.0f) ? (1.0507f * x) : (1.0507f * 1.6732f * (exp(x) - 1.0f));
}

inline float relie_fn(float x)
{
	return (x > 0.0f) ? x : (0.01f * x);
}

inline float ramp_fn(float x)
{
	return (x > 0.0f) ? (x + 0.1f * x) : (0.1f * x);
}

inline float hardtan_fn(float x)
{
	return clamp(x, -1.0f, 1.0f);
}

inline float lhtan_fn(float x)
{
	if (x < 0.0f)
	{
		return 0.001f * x;
	}
	if (x > 1.0f)
	{
		return 0.001f * (x - 1.0f) + 1.0f;
	}
	return x;
}

inline float plse_fn(float x)
{
	if (x < -4.0f)
	{
		return 0.01f * (x + 4.0f);
	}
	if (x > 4.0f)
	{
		return 0.01f * (x - 4.0f) + 1.0f;
	}
	return 0.125f * x + 0.5f;
}

inline float tanh_fn(float x)
{
	return (2.0f / (1.0f + exp(-2.0f * x)) - 1.0f);
}

inline float gelu_fn(float x)
{
	return 0.5f * x * (1.0f + tanh(0.797885f * x + 0.035677f * pow(x, 3.0f)));
}

inline float stair_fn(float x)
{
	int n = (int)floor(x);
	if ((n % 2) == 0)
	{
		return floor(x / 2.0f);
	}
	return (x - (float)n) + floor(x / 2.0f);
}

inline float revleaky_fn(float x)
{
	return (x > 0.0f) ? x : (0.1f * x);
}

kernel void logistic_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = logistic_fn(v.x);
	v.y = logistic_fn(v.y);
	v.z = logistic_fn(v.z);
	v.w = logistic_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void loggy_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = loggy_fn(v.x);
	v.y = loggy_fn(v.y);
	v.z = loggy_fn(v.z);
	v.w = loggy_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void tanh_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = tanh_fn(v.x);
	v.y = tanh_fn(v.y);
	v.z = tanh_fn(v.z);
	v.w = tanh_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void relu6_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = relu6_fn(v.x);
	v.y = relu6_fn(v.y);
	v.z = relu6_fn(v.z);
	v.w = relu6_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void elu_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = elu_fn(v.x);
	v.y = elu_fn(v.y);
	v.z = elu_fn(v.z);
	v.w = elu_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void selu_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = selu_fn(v.x);
	v.y = selu_fn(v.y);
	v.z = selu_fn(v.z);
	v.w = selu_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void gelu_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = gelu_fn(v.x);
	v.y = gelu_fn(v.y);
	v.z = gelu_fn(v.z);
	v.w = gelu_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void relie_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = relie_fn(v.x);
	v.y = relie_fn(v.y);
	v.z = relie_fn(v.z);
	v.w = relie_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void ramp_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = ramp_fn(v.x);
	v.y = ramp_fn(v.y);
	v.z = ramp_fn(v.z);
	v.w = ramp_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void hardtan_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = hardtan_fn(v.x);
	v.y = hardtan_fn(v.y);
	v.z = hardtan_fn(v.z);
	v.w = hardtan_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void lhtan_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = lhtan_fn(v.x);
	v.y = lhtan_fn(v.y);
	v.z = lhtan_fn(v.z);
	v.w = lhtan_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void plse_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = plse_fn(v.x);
	v.y = plse_fn(v.y);
	v.z = plse_fn(v.z);
	v.w = plse_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void stair_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = stair_fn(v.x);
	v.y = stair_fn(v.y);
	v.z = stair_fn(v.z);
	v.w = stair_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void revleaky_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	float4 v = tex.read(uint2(gid.xy), gid.z);
	v.x = revleaky_fn(v.x);
	v.y = revleaky_fn(v.y);
	v.z = revleaky_fn(v.z);
	v.w = revleaky_fn(v.w);
	tex.write(v, uint2(gid.xy), gid.z);
}

kernel void normalize_channels_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	constant uint &channels [[buffer(0)]],
	constant uint &slices_per_image [[buffer(1)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	if (channels == 0 || slices_per_image == 0)
	{
		return;
	}
	const uint image_index = gid.z / slices_per_image;
	const uint slice = gid.z - image_index * slices_per_image;
	// Exactly one thread owns all slices for a pixel.  The previous one-thread-
	// per-slice implementation raced by reading slices while peers overwrote them.
	if (slice != 0u)
	{
		return;
	}
	float sum = 0.0001f;
	for (uint s = 0; s < slices_per_image; ++s)
	{
		const uint slice_id = image_index * slices_per_image + s;
		if (slice_id >= tex.get_array_size())
		{
			break;
		}
		float4 v = tex.read(uint2(gid.xy), slice_id);
		const uint channel_base = s * 4;
		if (channel_base + 0 < channels && v.x > 0.0f) sum += v.x;
		if (channel_base + 1 < channels && v.y > 0.0f) sum += v.y;
		if (channel_base + 2 < channels && v.z > 0.0f) sum += v.z;
		if (channel_base + 3 < channels && v.w > 0.0f) sum += v.w;
	}
	for (uint s = 0; s < slices_per_image; ++s)
	{
		const uint slice_id = image_index * slices_per_image + s;
		if (slice_id >= tex.get_array_size())
		{
			break;
		}
		const uint base_channel = s * 4u;
		float4 v = tex.read(uint2(gid.xy), slice_id);
		v.x = (base_channel + 0u < channels && v.x > 0.0f) ? (v.x / sum) : 0.0f;
		v.y = (base_channel + 1u < channels && v.y > 0.0f) ? (v.y / sum) : 0.0f;
		v.z = (base_channel + 2u < channels && v.z > 0.0f) ? (v.z / sum) : 0.0f;
		v.w = (base_channel + 3u < channels && v.w > 0.0f) ? (v.w / sum) : 0.0f;
		tex.write(v, uint2(gid.xy), slice_id);
	}
}

kernel void normalize_channels_softmax_kernel(texture2d_array<float, access::read_write> tex [[texture(0)]],
	constant uint &channels [[buffer(0)]],
	constant uint &slices_per_image [[buffer(1)]],
	constant uint &use_max_val [[buffer(2)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= tex.get_width() || gid.y >= tex.get_height() || gid.z >= tex.get_array_size())
	{
		return;
	}
	if (channels == 0 || slices_per_image == 0)
	{
		return;
	}
	const uint image_index = gid.z / slices_per_image;
	const uint slice = gid.z - image_index * slices_per_image;
	if (slice != 0u)
	{
		return;
	}
	float max_val = use_max_val ? -INFINITY : 0.0f;
	if (use_max_val)
	{
		for (uint s = 0; s < slices_per_image; ++s)
		{
			const uint slice_id = image_index * slices_per_image + s;
			if (slice_id >= tex.get_array_size())
			{
				break;
			}
			float4 v = tex.read(uint2(gid.xy), slice_id);
			const uint channel_base = s * 4;
			if (channel_base + 0 < channels) max_val = max(max_val, v.x);
			if (channel_base + 1 < channels) max_val = max(max_val, v.y);
			if (channel_base + 2 < channels) max_val = max(max_val, v.z);
			if (channel_base + 3 < channels) max_val = max(max_val, v.w);
		}
	}
	float sum = 0.0001f;
	for (uint s = 0; s < slices_per_image; ++s)
	{
		const uint slice_id = image_index * slices_per_image + s;
		if (slice_id >= tex.get_array_size())
		{
			break;
		}
		float4 v = tex.read(uint2(gid.xy), slice_id);
		const uint channel_base = s * 4;
		if (channel_base + 0 < channels) sum += exp(v.x - max_val);
		if (channel_base + 1 < channels) sum += exp(v.y - max_val);
		if (channel_base + 2 < channels) sum += exp(v.z - max_val);
		if (channel_base + 3 < channels) sum += exp(v.w - max_val);
	}
	for (uint s = 0; s < slices_per_image; ++s)
	{
		const uint slice_id = image_index * slices_per_image + s;
		if (slice_id >= tex.get_array_size())
		{
			break;
		}
		const uint base_channel = s * 4u;
		float4 v = tex.read(uint2(gid.xy), slice_id);
		v.x = (base_channel + 0u < channels) ? exp(v.x - max_val) / sum : 0.0f;
		v.y = (base_channel + 1u < channels) ? exp(v.y - max_val) / sum : 0.0f;
		v.z = (base_channel + 2u < channels) ? exp(v.z - max_val) / sum : 0.0f;
		v.w = (base_channel + 3u < channels) ? exp(v.w - max_val) / sum : 0.0f;
		tex.write(v, uint2(gid.xy), slice_id);
	}
}

kernel void route_clear_kernel(texture2d_array<float, access::read_write> out_tex [[texture(0)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	out_tex.write(float4(0.0f), uint2(gid.xy), gid.z);
}

kernel void route_copy_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::read_write> out_tex [[texture(1)]],
	constant uint &in_channels_total [[buffer(0)]],
	constant uint &out_channels_total [[buffer(1)]],
	constant uint &in_channel_offset [[buffer(2)]],
	constant uint &out_channel_offset [[buffer(3)]],
	constant uint &copy_channels [[buffer(4)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const uint out_slices = (out_channels_total + 3u) / 4u;
	if (out_slices == 0)
	{
		return;
	}
	const uint batch = gid.z / out_slices;
	const uint slice = gid.z - batch * out_slices;
	const uint base_c = slice * 4u;
	const uint in_slices = (in_channels_total + 3u) / 4u;

	float4 out = out_tex.read(uint2(gid.xy), gid.z);
	for (uint i = 0; i < 4u; ++i)
	{
		const uint out_c = base_c + i;
		if (out_c < out_channel_offset || out_c >= (out_channel_offset + copy_channels))
		{
			continue;
		}
		const uint in_c = in_channel_offset + (out_c - out_channel_offset);
		if (in_c >= in_channels_total)
		{
			continue;
		}
		const uint in_slice = (in_c / 4u) + batch * in_slices;
		const uint in_comp = in_c % 4u;
		float4 v = in_tex.read(uint2(gid.xy), in_slice);
		out[i] = v[in_comp];
	}
	out_tex.write(out, uint2(gid.xy), gid.z);
}

kernel void reorg_forward_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::write> out_tex [[texture(1)]],
	constant uint &in_channels [[buffer(0)]],
	constant uint &out_channels [[buffer(1)]],
	constant uint &stride [[buffer(2)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const uint out_slices = (out_channels + 3u) / 4u;
	if (out_slices == 0)
	{
		return;
	}
	const uint batch = gid.z / out_slices;
	const uint slice = gid.z - batch * out_slices;
	const uint base_c = slice * 4u;
	const uint in_slices = (in_channels + 3u) / 4u;

	float4 out = float4(0.0f);
	for (uint i = 0; i < 4u; ++i)
	{
		const uint out_c = base_c + i;
		if (out_c >= out_channels)
		{
			continue;
		}
		const uint offset = out_c / in_channels;
		const uint c2 = out_c - offset * in_channels;
		const uint w2 = gid.x * stride + (offset % stride);
		const uint h2 = gid.y * stride + (offset / stride);
		if (w2 >= in_tex.get_width() || h2 >= in_tex.get_height())
		{
			continue;
		}
		const uint in_slice = (c2 / 4u) + batch * in_slices;
		const uint in_comp = c2 % 4u;
		float4 v = in_tex.read(uint2(w2, h2), in_slice);
		out[i] = v[in_comp];
	}
	out_tex.write(out, uint2(gid.xy), gid.z);
}

kernel void reorg_reverse_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::write> out_tex [[texture(1)]],
	constant uint &in_channels [[buffer(0)]],
	constant uint &out_channels [[buffer(1)]],
	constant uint &stride [[buffer(2)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const uint out_slices = (out_channels + 3u) / 4u;
	if (out_slices == 0)
	{
		return;
	}
	const uint batch = gid.z / out_slices;
	const uint slice = gid.z - batch * out_slices;
	const uint base_c = slice * 4u;
	const uint in_slices = (in_channels + 3u) / 4u;

	float4 out = float4(0.0f);
	for (uint i = 0; i < 4u; ++i)
	{
		const uint out_c = base_c + i;
		if (out_c >= out_channels)
		{
			continue;
		}
		const uint offset = (gid.y % stride) * stride + (gid.x % stride);
		const uint c_in = out_c + out_channels * offset;
		if (c_in >= in_channels)
		{
			continue;
		}
		const uint in_x = gid.x / stride;
		const uint in_y = gid.y / stride;
		if (in_x >= in_tex.get_width() || in_y >= in_tex.get_height())
		{
			continue;
		}
		const uint in_slice = (c_in / 4u) + batch * in_slices;
		const uint in_comp = c_in % 4u;
		float4 v = in_tex.read(uint2(in_x, in_y), in_slice);
		out[i] = v[in_comp];
	}
	out_tex.write(out, uint2(gid.xy), gid.z);
}

kernel void maxpool_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::write> out_tex [[texture(1)]],
	constant uint &channels [[buffer(0)]],
	constant uint &size [[buffer(1)]],
	constant uint &stride_x [[buffer(2)]],
	constant uint &stride_y [[buffer(3)]],
	constant int &pad [[buffer(4)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const uint out_slices = (channels + 3u) / 4u;
	if (out_slices == 0)
	{
		return;
	}
	const uint batch = gid.z / out_slices;
	const uint slice = gid.z - batch * out_slices;
	const uint base_c = slice * 4u;
	const uint in_slices = out_slices;

	const int in_w = (int)in_tex.get_width();
	const int in_h = (int)in_tex.get_height();
	const int start_x = (int)(gid.x * stride_x) - pad;
	const int start_y = (int)(gid.y * stride_y) - pad;

	float4 out = float4(-INFINITY);
	for (uint i = 0; i < 4u; ++i)
	{
		const uint out_c = base_c + i;
		if (out_c >= channels)
		{
			out[i] = 0.0f;
		}
	}

	for (uint ky = 0; ky < size; ++ky)
	{
		for (uint kx = 0; kx < size; ++kx)
		{
			const int ix = start_x + (int)kx;
			const int iy = start_y + (int)ky;
			if (ix < 0 || iy < 0 || ix >= in_w || iy >= in_h)
			{
				continue;
			}
			float4 v = in_tex.read(uint2((uint)ix, (uint)iy), slice + batch * in_slices);
			out = max(out, v);
		}
	}
	for (uint i = 0; i < 4u; ++i)
	{
		const uint out_c = base_c + i;
		if (out_c >= channels)
		{
			out[i] = 0.0f;
		}
	}
	out_tex.write(out, uint2(gid.xy), gid.z);
}

kernel void avgpool_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::write> out_tex [[texture(1)]],
	constant uint &channels [[buffer(0)]],
	constant uint &size [[buffer(1)]],
	constant uint &stride_x [[buffer(2)]],
	constant uint &stride_y [[buffer(3)]],
	constant int &pad [[buffer(4)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const uint out_slices = (channels + 3u) / 4u;
	if (out_slices == 0)
	{
		return;
	}
	const uint batch = gid.z / out_slices;
	const uint slice = gid.z - batch * out_slices;
	const uint base_c = slice * 4u;
	const uint in_slices = out_slices;

	const int in_w = (int)in_tex.get_width();
	const int in_h = (int)in_tex.get_height();
	const int start_x = (int)(gid.x * stride_x) - pad;
	const int start_y = (int)(gid.y * stride_y) - pad;

	float4 sum = float4(0.0f);
	uint count = 0u;

	for (uint ky = 0; ky < size; ++ky)
	{
		for (uint kx = 0; kx < size; ++kx)
		{
			const int ix = start_x + (int)kx;
			const int iy = start_y + (int)ky;
			if (ix < 0 || iy < 0 || ix >= in_w || iy >= in_h)
			{
				continue;
			}
			float4 v = in_tex.read(uint2((uint)ix, (uint)iy), slice + batch * in_slices);
			sum += v;
			count += 1u;
		}
	}

	float4 out = float4(0.0f);
	if (count > 0u)
	{
		out = sum / (float)count;
	}
	for (uint i = 0; i < 4u; ++i)
	{
		const uint out_c = base_c + i;
		if (out_c >= channels)
		{
			out[i] = 0.0f;
		}
	}
	out_tex.write(out, uint2(gid.xy), gid.z);
}

kernel void softmax_kernel(device const float *input [[buffer(0)]],
	device float *output [[buffer(1)]],
	constant uint &n [[buffer(2)]],
	constant uint &batch [[buffer(3)]],
	constant uint &batch_offset [[buffer(4)]],
	constant uint &groups [[buffer(5)]],
	constant uint &group_offset [[buffer(6)]],
	constant uint &stride [[buffer(7)]],
	constant float &temp [[buffer(8)]],
	uint gid [[thread_position_in_grid]])
{
	const uint total = batch * groups * n;
	if (gid >= total)
	{
		return;
	}
	const uint group_size = n;
	const uint batch_group = groups * group_size;
	const uint b = gid / batch_group;
	const uint g = (gid / group_size) % groups;
	const uint i = gid % group_size;
	const uint base = b * batch_offset + g * group_offset;

	float max_val = -INFINITY;
	for (uint j = 0; j < group_size; ++j)
	{
		float v = input[base + j * stride];
		max_val = max(max_val, v);
	}
	float sum = 0.0f;
	for (uint j = 0; j < group_size; ++j)
	{
		float v = input[base + j * stride];
		sum += exp((v - max_val) / temp);
	}
	float val = input[base + i * stride];
	output[base + i * stride] = exp((val - max_val) / temp) / sum;
}

kernel void buffer_fill_float(device float *data [[buffer(0)]],
	constant float &value [[buffer(1)]],
	constant uint &total [[buffer(2)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= total)
	{
		return;
	}
	data[gid] = value;
}

kernel void buffer_copy(device const float *src [[buffer(0)]],
	device float *dst [[buffer(1)]],
	constant uint &total [[buffer(2)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= total)
	{
		return;
	}
	dst[gid] = src[gid];
}

kernel void buffer_axpy(device const float *x [[buffer(0)]],
	device float *y [[buffer(1)]],
	constant float &alpha [[buffer(2)]],
	constant uint &total [[buffer(3)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= total)
	{
		return;
	}
	y[gid] += alpha * x[gid];
}

kernel void buffer_scal(device float *x [[buffer(0)]],
	constant float &alpha [[buffer(1)]],
	constant uint &total [[buffer(2)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= total)
	{
		return;
	}
	x[gid] *= alpha;
}

kernel void sgd_update_weights(device float *parameters [[buffer(0)]],
	device float *updates [[buffer(1)]],
	constant float &rate_over_batch [[buffer(2)]],
	constant float &momentum [[buffer(3)]],
	constant float &decay_term [[buffer(4)]],
	constant uint &total [[buffer(5)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= total)
	{
		return;
	}
	float update = isfinite(updates[gid]) ? updates[gid] : 0.0f;
	float parameter = isfinite(parameters[gid]) ? parameters[gid] : 1.0f / float(gid + 1u);
	update += decay_term * parameter;
	parameter += rate_over_batch * update;
	parameters[gid] = parameter;
	updates[gid] = momentum * update;
}

kernel void sgd_update_parameter(device float *parameters [[buffer(0)]],
	device float *updates [[buffer(1)]],
	constant float &rate_over_batch [[buffer(2)]],
	constant float &momentum [[buffer(3)]],
	constant uint &total [[buffer(4)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= total)
	{
		return;
	}
	const float update = updates[gid];
	parameters[gid] += rate_over_batch * update;
	updates[gid] = momentum * update;
}

inline float apple_logistic_fn(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

struct apple_yolo_plain_params
{
	uint batch;
	uint w;
	uint h;
	uint n;
	uint total;
	uint classes;
	uint outputs;
	uint truths;
	uint truth_size;
	uint max_boxes;
	uint net_w;
	uint net_h;
	uint new_coords;
	uint objectness_smooth;
	float scale_x_y;
	float ignore_thresh;
	float truth_thresh;
	float iou_thresh;
	float obj_normalizer;
	float cls_normalizer;
	float iou_normalizer;
	float max_delta;
};

struct apple_yolo_plain_loss_stats
{
	float total_sq;
	float class_sq;
	float iou_loss;
	float iou_sum;
	float count;
	float rewritten;
};

inline uint apple_yolo_spatial(constant apple_yolo_plain_params &p)
{
	return p.w * p.h;
}

inline uint apple_yolo_entry_index(constant apple_yolo_plain_params &p, uint b, uint n, uint loc, uint entry)
{
	return b * p.outputs + n * apple_yolo_spatial(p) * (p.classes + 5u) + entry * apple_yolo_spatial(p) + loc;
}

inline float4 apple_yolo_box(device const float *output,
	device const float *biases,
	constant apple_yolo_plain_params &p,
	uint anchor,
	uint box_index,
	uint cell_x,
	uint cell_y)
{
	const uint stride = apple_yolo_spatial(p);
	float4 box;
	box.x = (float(cell_x) + output[box_index + 0u * stride]) / float(p.w);
	box.y = (float(cell_y) + output[box_index + 1u * stride]) / float(p.h);
	if (p.new_coords != 0u)
	{
		const float w = output[box_index + 2u * stride];
		const float h = output[box_index + 3u * stride];
		box.z = w * w * 4.0f * biases[2u * anchor] / float(p.net_w);
		box.w = h * h * 4.0f * biases[2u * anchor + 1u] / float(p.net_h);
	}
	else
	{
		box.z = exp(output[box_index + 2u * stride]) * biases[2u * anchor] / float(p.net_w);
		box.w = exp(output[box_index + 3u * stride]) * biases[2u * anchor + 1u] / float(p.net_h);
	}
	return box;
}

inline float apple_yolo_overlap(float x1, float w1, float x2, float w2)
{
	const float l1 = x1 - w1 * 0.5f;
	const float l2 = x2 - w2 * 0.5f;
	const float left = max(l1, l2);
	const float r1 = x1 + w1 * 0.5f;
	const float r2 = x2 + w2 * 0.5f;
	const float right = min(r1, r2);
	return right - left;
}

inline float apple_yolo_intersection(float4 a, float4 b)
{
	const float w = apple_yolo_overlap(a.x, a.z, b.x, b.z);
	if (w <= 0.0f)
	{
		return 0.0f;
	}
	const float h = apple_yolo_overlap(a.y, a.w, b.y, b.w);
	if (h <= 0.0f)
	{
		return 0.0f;
	}
	return w * h;
}

inline float apple_yolo_iou(float4 a, float4 b)
{
	if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(a.z) || !isfinite(a.w) ||
		!isfinite(b.x) || !isfinite(b.y) || !isfinite(b.z) || !isfinite(b.w))
	{
		return 0.0f;
	}
	const float intersection = apple_yolo_intersection(a, b);
	if (!isfinite(intersection) || intersection <= 0.0f)
	{
		return 0.0f;
	}
	const float u = a.z * a.w + b.z * b.w - intersection;
	if (!isfinite(u) || u <= 0.0f)
	{
		return 0.0f;
	}
	const float value = intersection / u;
	return isfinite(value) ? value : 0.0f;
}

inline float4 apple_yolo_ciou_delta(float4 pred, float4 truth)
{
	if (!isfinite(pred.x) || !isfinite(pred.y) || !isfinite(pred.z) || !isfinite(pred.w) ||
		!isfinite(truth.x) || !isfinite(truth.y) || !isfinite(truth.z) || !isfinite(truth.w))
	{
		return float4(0.0f);
	}
	const float pred_t0 = pred.y - pred.w * 0.5f;
	const float pred_b0 = pred.y + pred.w * 0.5f;
	const float pred_l0 = pred.x - pred.z * 0.5f;
	const float pred_r0 = pred.x + pred.z * 0.5f;
	const float pred_t = min(pred_t0, pred_b0);
	const float pred_b = max(pred_t0, pred_b0);
	const float pred_l = min(pred_l0, pred_r0);
	const float pred_r = max(pred_l0, pred_r0);
	const float truth_t = truth.y - truth.w * 0.5f;
	const float truth_b = truth.y + truth.w * 0.5f;
	const float truth_l = truth.x - truth.z * 0.5f;
	const float truth_r = truth.x + truth.z * 0.5f;

	const float X = (pred_b - pred_t) * (pred_r - pred_l);
	const float Xhat = (truth_b - truth_t) * (truth_r - truth_l);
	const float Ih = min(pred_b, truth_b) - max(pred_t, truth_t);
	const float Iw = min(pred_r, truth_r) - max(pred_l, truth_l);
	const float I = Iw * Ih;
	const float U = X + Xhat - I;
	const float S = (pred.x - truth.x) * (pred.x - truth.x) + (pred.y - truth.y) * (pred.y - truth.y);

	const float dX_wrt_t = -1.0f * (pred_r - pred_l);
	const float dX_wrt_b = pred_r - pred_l;
	const float dX_wrt_l = -1.0f * (pred_b - pred_t);
	const float dX_wrt_r = pred_b - pred_t;

	const float dI_wrt_t = pred_t > truth_t ? (-1.0f * Iw) : 0.0f;
	const float dI_wrt_b = pred_b < truth_b ? Iw : 0.0f;
	const float dI_wrt_l = pred_l > truth_l ? (-1.0f * Ih) : 0.0f;
	const float dI_wrt_r = pred_r < truth_r ? Ih : 0.0f;

	const float dU_wrt_t = dX_wrt_t - dI_wrt_t;
	const float dU_wrt_b = dX_wrt_b - dI_wrt_b;
	const float dU_wrt_l = dX_wrt_l - dI_wrt_l;
	const float dU_wrt_r = dX_wrt_r - dI_wrt_r;

	float p_dt = 0.0f;
	float p_db = 0.0f;
	float p_dl = 0.0f;
	float p_dr = 0.0f;
	if (U > 0.0f)
	{
		const float uu = U * U;
		p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / uu;
		p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / uu;
		p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / uu;
		p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / uu;
	}

	const float p_dt0 = p_dt;
	const float p_db0 = p_db;
	const float p_dl0 = p_dl;
	const float p_dr0 = p_dr;
	p_dt = pred_t0 < pred_b0 ? p_dt0 : p_db0;
	p_db = pred_t0 < pred_b0 ? p_db0 : p_dt0;
	p_dl = pred_l0 < pred_r0 ? p_dl0 : p_dr0;
	p_dr = pred_l0 < pred_r0 ? p_dr0 : p_dl0;

	const float Ct = min(pred.y - pred.w * 0.5f, truth.y - truth.w * 0.5f);
	const float Cb = max(pred.y + pred.w * 0.5f, truth.y + truth.w * 0.5f);
	const float Cl = min(pred.x - pred.z * 0.5f, truth.x - truth.z * 0.5f);
	const float Cr = max(pred.x + pred.z * 0.5f, truth.x + truth.z * 0.5f);
	const float Cw = Cr - Cl;
	const float Ch = Cb - Ct;
	const float C = Cw * Cw + Ch * Ch;

	const float dCt_dx = 0.0f;
	const float dCt_dy = pred_t < truth_t ? 1.0f : 0.0f;
	const float dCt_dw = 0.0f;
	const float dCt_dh = pred_t < truth_t ? -0.5f : 0.0f;

	const float dCb_dx = 0.0f;
	const float dCb_dy = pred_b > truth_b ? 1.0f : 0.0f;
	const float dCb_dw = 0.0f;
	const float dCb_dh = pred_b > truth_b ? 0.5f : 0.0f;

	const float dCl_dx = pred_l < truth_l ? 1.0f : 0.0f;
	const float dCl_dy = 0.0f;
	const float dCl_dw = pred_l < truth_l ? -0.5f : 0.0f;
	const float dCl_dh = 0.0f;

	const float dCr_dx = pred_r > truth_r ? 1.0f : 0.0f;
	const float dCr_dy = 0.0f;
	const float dCr_dw = pred_r > truth_r ? 0.5f : 0.0f;
	const float dCr_dh = 0.0f;

	const float dCw_dx = dCr_dx - dCl_dx;
	const float dCw_dy = dCr_dy - dCl_dy;
	const float dCw_dw = dCr_dw - dCl_dw;
	const float dCw_dh = dCr_dh - dCl_dh;

	const float dCh_dx = dCb_dx - dCt_dx;
	const float dCh_dy = dCb_dy - dCt_dy;
	const float dCh_dw = dCb_dw - dCt_dw;
	const float dCh_dh = dCb_dh - dCt_dh;

	float p_dx = p_dl + p_dr;
	float p_dy = p_dt + p_db;
	float p_dw = p_dr - p_dl;
	float p_dh = p_db - p_dt;

	const float ar_gt = truth.z / truth.w;
	const float ar_pred = pred.z / pred.w;
	const float pi = 3.14159265358979323846f;
	const float ar_diff = atan(ar_gt) - atan(ar_pred);
	const float ar_loss = 4.0f / (pi * pi) * ar_diff * ar_diff;
	const float iou_ratio = U != 0.0f ? I / U : 0.0f;
	const float alpha = ar_loss / (1.0f - iou_ratio + ar_loss + 0.000001f);
	const float ar_dw = 8.0f / (pi * pi) * ar_diff * pred.w;
	const float ar_dh = -8.0f / (pi * pi) * ar_diff * pred.z;

	if (C > 0.0f)
	{
		const float cc = C * C;
		p_dx += (2.0f * (truth.x - pred.x) * C - (2.0f * Cw * dCw_dx + 2.0f * Ch * dCh_dx) * S) / cc;
		p_dy += (2.0f * (truth.y - pred.y) * C - (2.0f * Cw * dCw_dy + 2.0f * Ch * dCh_dy) * S) / cc;
		p_dw += (2.0f * Cw * dCw_dw + 2.0f * Ch * dCh_dw) * S / cc + alpha * ar_dw;
		p_dh += (2.0f * Cw * dCw_dh + 2.0f * Ch * dCh_dh) * S / cc + alpha * ar_dh;
	}
	if (Iw <= 0.0f || Ih <= 0.0f)
	{
		const float cc = C * C;
		p_dx = (2.0f * (truth.x - pred.x) * C - (2.0f * Cw * dCw_dx + 2.0f * Ch * dCh_dx) * S) / cc;
		p_dy = (2.0f * (truth.y - pred.y) * C - (2.0f * Cw * dCw_dy + 2.0f * Ch * dCh_dy) * S) / cc;
		p_dw = (2.0f * Cw * dCw_dw + 2.0f * Ch * dCh_dw) * S / cc + alpha * ar_dw;
		p_dh = (2.0f * Cw * dCw_dh + 2.0f * Ch * dCh_dh) * S / cc + alpha * ar_dh;
	}

	return float4(p_dx, p_dy, p_dw, p_dh);
}

inline float apple_yolo_clean_delta(float value, float max_delta)
{
	if (!isfinite(value))
	{
		return 0.0f;
	}
	if (max_delta < 3.4028234663852886e38f)
	{
		value = clamp(value, -max_delta, max_delta);
	}
	return value;
}

inline float apple_yolo_clean_output(float value)
{
	return isfinite(value) ? value : 0.0f;
}

inline float apple_yolo_objectness_delta(float normalizer, float target, float output)
{
	const float value = normalizer * (target - output);
	return isfinite(value) ? value : 0.0f;
}

inline void apple_yolo_assign_class_delta(device const float *output,
	device float *delta,
	constant apple_yolo_plain_params &p,
	uint b,
	uint n,
	uint loc,
	int class_id)
{
	const uint target_class_index = apple_yolo_entry_index(p, b, n, loc, 5u + uint(class_id));
	if (delta[target_class_index] != 0.0f)
	{
		const float result_delta = 1.0f - output[target_class_index];
		delta[target_class_index] = isfinite(result_delta) ? result_delta : delta[target_class_index];
		return;
	}

	for (uint c = 0; c < p.classes; ++c)
	{
		const uint class_index = apple_yolo_entry_index(p, b, n, loc, 5u + c);
		const float y_true = int(c) == class_id ? 1.0f : 0.0f;
		const float result_delta = y_true - output[class_index];
		delta[class_index] = isfinite(result_delta) ? result_delta : delta[class_index];
	}
}

inline void apple_yolo_add_ciou_delta(float4 truth,
	device float *output,
	device float *delta,
	device const float *biases,
	constant apple_yolo_plain_params &p,
	uint anchor,
	uint box_index,
	uint cell_x,
	uint cell_y,
	float scale)
{
	float4 pred = apple_yolo_box(output, biases, p, anchor, box_index, cell_x, cell_y);
	if (pred.z == 0.0f)
	{
		pred.z = 1.0f;
	}
	if (pred.w == 0.0f)
	{
		pred.w = 1.0f;
	}

	float4 d = apple_yolo_ciou_delta(pred, truth);
	const uint stride = apple_yolo_spatial(p);
	if (p.new_coords == 0u)
	{
		d.z *= exp(output[box_index + 2u * stride]);
		d.w *= exp(output[box_index + 3u * stride]);
	}
	d *= p.iou_normalizer;

	delta[box_index + 0u * stride] += apple_yolo_clean_delta(d.x, p.max_delta);
	delta[box_index + 1u * stride] += apple_yolo_clean_delta(d.y, p.max_delta);
	delta[box_index + 2u * stride] += apple_yolo_clean_delta(d.z, p.max_delta);
	delta[box_index + 3u * stride] += apple_yolo_clean_delta(d.w, p.max_delta);
	(void)scale;
}

kernel void yolo_plain_activate_base_delta(
	device const float *input [[buffer(0)]],
	device float *output [[buffer(1)]],
	device float *delta [[buffer(2)]],
	device int *labels [[buffer(3)]],
	device int *class_ids [[buffer(4)]],
	constant apple_yolo_plain_params &p [[buffer(5)]],
	uint gid [[thread_position_in_grid]])
{
	const uint spatial = apple_yolo_spatial(p);
	const uint predictions = p.batch * p.n * spatial;
	if (gid >= predictions)
	{
		return;
	}

	const uint b = gid / (p.n * spatial);
	const uint rem = gid - b * p.n * spatial;
	const uint n = rem / spatial;
	const uint loc = rem - n * spatial;
	const uint box_index = apple_yolo_entry_index(p, b, n, loc, 0u);
	const uint obj_index = apple_yolo_entry_index(p, b, n, loc, 4u);

	for (uint entry = 0; entry < p.classes + 5u; ++entry)
	{
		const uint index = apple_yolo_entry_index(p, b, n, loc, entry);
		output[index] = input[index];
		delta[index] = 0.0f;
	}

	if (p.new_coords != 0u)
	{
		output[box_index + 0u * spatial] = output[box_index + 0u * spatial] * p.scale_x_y - 0.5f * (p.scale_x_y - 1.0f);
		output[box_index + 1u * spatial] = output[box_index + 1u * spatial] * p.scale_x_y - 0.5f * (p.scale_x_y - 1.0f);
	}
	else
	{
		output[box_index + 0u * spatial] = apple_logistic_fn(output[box_index + 0u * spatial]) * p.scale_x_y - 0.5f * (p.scale_x_y - 1.0f);
		output[box_index + 1u * spatial] = apple_logistic_fn(output[box_index + 1u * spatial]) * p.scale_x_y - 0.5f * (p.scale_x_y - 1.0f);
		output[obj_index] = apple_yolo_clean_output(apple_logistic_fn(output[obj_index]));
		for (uint c = 0; c < p.classes; ++c)
		{
			const uint class_index = apple_yolo_entry_index(p, b, n, loc, 5u + c);
			output[class_index] = apple_logistic_fn(output[class_index]);
		}
	}
	delta[obj_index] = apple_yolo_objectness_delta(p.obj_normalizer, 0.0f, output[obj_index]);
	labels[gid] = -1;
	class_ids[gid] = -1;
}

kernel void yolo_plain_ignore_delta(
	device const float *output [[buffer(0)]],
	device float *delta [[buffer(1)]],
	device const float *truth [[buffer(2)]],
	device const float *biases [[buffer(3)]],
	device const int *mask [[buffer(4)]],
	constant apple_yolo_plain_params &p [[buffer(5)]],
	uint gid [[thread_position_in_grid]])
{
	const uint spatial = apple_yolo_spatial(p);
	const uint predictions = p.batch * p.n * spatial;
	if (gid >= predictions)
	{
		return;
	}

	const uint b = gid / (p.n * spatial);
	const uint rem = gid - b * p.n * spatial;
	const uint n = rem / spatial;
	const uint loc = rem - n * spatial;
	const uint cell_y = loc / p.w;
	const uint cell_x = loc - cell_y * p.w;
	const uint box_index = apple_yolo_entry_index(p, b, n, loc, 0u);
	const uint obj_index = apple_yolo_entry_index(p, b, n, loc, 4u);
	const uint anchor = uint(mask[n]);
	const float4 pred = apple_yolo_box(output, biases, p, anchor, box_index, cell_x, cell_y);
	bool class_match = false;
	for (uint c = 0; c < p.classes; ++c)
	{
		if (output[apple_yolo_entry_index(p, b, n, loc, 5u + c)] > 0.25f)
		{
			class_match = true;
			break;
		}
	}

	float best_match_iou = 0.0f;
	for (uint t = 0; t < p.max_boxes; ++t)
	{
		const uint truth_index = b * p.truths + t * p.truth_size;
		const float4 truth_box = float4(truth[truth_index + 0u], truth[truth_index + 1u], truth[truth_index + 2u], truth[truth_index + 3u]);
		if (truth_box.x == 0.0f)
		{
			break;
		}
		const int class_id = int(truth[truth_index + 4u]);
		if (class_id < 0 || class_id >= int(p.classes))
		{
			continue;
		}

		const float iou = apple_yolo_iou(pred, truth_box);
		if (class_match && iou > best_match_iou)
		{
			best_match_iou = iou;
		}
	}

	if (best_match_iou > p.ignore_thresh)
	{
		if (p.objectness_smooth != 0u)
		{
			const float delta_obj = apple_yolo_objectness_delta(p.obj_normalizer, best_match_iou, output[obj_index]);
			if (delta_obj > delta[obj_index])
			{
				delta[obj_index] = delta_obj;
			}
		}
		else
		{
			delta[obj_index] = 0.0f;
		}
	}
}

kernel void yolo_plain_assign_truth_delta(
	device float *output [[buffer(0)]],
	device float *delta [[buffer(1)]],
	device const float *truth [[buffer(2)]],
	device const float *biases [[buffer(3)]],
	device const int *mask [[buffer(4)]],
	device int *labels [[buffer(5)]],
	device int *class_ids [[buffer(6)]],
	device apple_yolo_plain_loss_stats *stats [[buffer(7)]],
	constant apple_yolo_plain_params &p [[buffer(8)]],
	uint b [[thread_position_in_grid]])
{
	if (b >= p.batch)
	{
		return;
	}
	const uint spatial = apple_yolo_spatial(p);
	float iou_loss = 0.0f;
	float iou_sum = 0.0f;
	float count = 0.0f;
	float rewritten = 0.0f;

	for (uint t = 0; t < p.max_boxes; ++t)
	{
		const uint truth_index = b * p.truths + t * p.truth_size;
		const float4 truth_box = float4(truth[truth_index + 0u], truth[truth_index + 1u], truth[truth_index + 2u], truth[truth_index + 3u]);
		if (truth_box.x == 0.0f)
		{
			break;
		}
		const int class_id = int(truth[truth_index + 4u]);
		if (class_id < 0 || class_id >= int(p.classes) ||
			truth_box.x < 0.0f || truth_box.y < 0.0f || truth_box.x > 1.0f || truth_box.y > 1.0f ||
			truth_box.z < 0.0f || truth_box.w < 0.0f)
		{
			continue;
		}

		float best_iou = 0.0f;
		uint best_n = 0u;
		const float4 truth_shift = float4(0.0f, 0.0f, truth_box.z, truth_box.w);
		for (uint n = 0; n < p.total; ++n)
		{
			const float4 pred = float4(0.0f, 0.0f, biases[2u * n] / float(p.net_w), biases[2u * n + 1u] / float(p.net_h));
			const float iou = apple_yolo_iou(pred, truth_shift);
			if (iou > best_iou)
			{
				best_iou = iou;
				best_n = n;
			}
		}

		const uint cell_x = min(uint(truth_box.x * float(p.w)), p.w - 1u);
		const uint cell_y = min(uint(truth_box.y * float(p.h)), p.h - 1u);
		const uint loc = cell_y * p.w + cell_x;

		int mask_index = -1;
		for (uint n = 0; n < p.n; ++n)
		{
			if (uint(mask[n]) == best_n)
			{
				mask_index = int(n);
				break;
			}
		}
		if (mask_index >= 0)
		{
			const uint n = uint(mask_index);
			const uint box_index = apple_yolo_entry_index(p, b, n, loc, 0u);
			const bool box_delta_was_set =
				delta[box_index + 0u * spatial] != 0.0f ||
				delta[box_index + 1u * spatial] != 0.0f ||
				delta[box_index + 2u * spatial] != 0.0f ||
				delta[box_index + 3u * spatial] != 0.0f;
			const float4 pred_box = apple_yolo_box(output, biases, p, best_n, box_index, cell_x, cell_y);
			const float assigned_iou = apple_yolo_iou(pred_box, truth_box);
			apple_yolo_add_ciou_delta(truth_box, output, delta, biases, p, best_n, box_index, cell_x, cell_y, 2.0f - truth_box.z * truth_box.w);
			iou_sum += assigned_iou;
			iou_loss += 1.0f - assigned_iou;
			count += 1.0f;
			rewritten += box_delta_was_set ? 1.0f : 0.0f;

			const uint obj_index = apple_yolo_entry_index(p, b, n, loc, 4u);
			const float delta_obj = apple_yolo_objectness_delta(p.obj_normalizer, 1.0f, output[obj_index]);
			if (p.objectness_smooth != 0u)
			{
				if (delta[obj_index] == 0.0f)
				{
					delta[obj_index] = delta_obj;
				}
			}
			else
			{
				delta[obj_index] = delta_obj;
			}

			apple_yolo_assign_class_delta(output, delta, p, b, n, loc, class_id);

			const uint truth_out_index = b * p.n * spatial + n * spatial + loc;
			labels[truth_out_index] = p.truth_size > 5u ? int(truth[truth_index + 5u]) : 0;
			class_ids[truth_out_index] = class_id;
		}

		if (p.iou_thresh < 1.0f)
		{
			for (uint anchor_candidate = 0; anchor_candidate < p.total; ++anchor_candidate)
			{
				if (anchor_candidate == best_n)
				{
					continue;
				}
				int mask_n = -1;
				for (uint mask_candidate = 0; mask_candidate < p.n; ++mask_candidate)
				{
					if (uint(mask[mask_candidate]) == anchor_candidate)
					{
						mask_n = int(mask_candidate);
						break;
					}
				}
				if (mask_n < 0)
				{
					continue;
				}

				const float4 anchor_box = float4(0.0f, 0.0f,
					biases[2u * anchor_candidate] / float(p.net_w),
					biases[2u * anchor_candidate + 1u] / float(p.net_h));
				const float anchor_iou = apple_yolo_iou(anchor_box, truth_shift);
				if (anchor_iou <= p.iou_thresh)
				{
					continue;
				}

				const uint extra_n = uint(mask_n);
				const uint extra_box_index = apple_yolo_entry_index(p, b, extra_n, loc, 0u);
				const bool extra_box_delta_was_set =
					delta[extra_box_index + 0u * spatial] != 0.0f ||
					delta[extra_box_index + 1u * spatial] != 0.0f ||
					delta[extra_box_index + 2u * spatial] != 0.0f ||
					delta[extra_box_index + 3u * spatial] != 0.0f;
				const float4 extra_pred_box = apple_yolo_box(output, biases, p, anchor_candidate, extra_box_index, cell_x, cell_y);
				const float extra_iou = apple_yolo_iou(extra_pred_box, truth_box);
				apple_yolo_add_ciou_delta(truth_box, output, delta, biases, p, anchor_candidate, extra_box_index, cell_x, cell_y, 2.0f - truth_box.z * truth_box.w);
				iou_sum += extra_iou;
				iou_loss += 1.0f - extra_iou;
				count += 1.0f;
				rewritten += extra_box_delta_was_set ? 1.0f : 0.0f;

				const uint extra_obj_index = apple_yolo_entry_index(p, b, extra_n, loc, 4u);
				const float extra_delta_obj = apple_yolo_objectness_delta(p.obj_normalizer, 1.0f, output[extra_obj_index]);
				if (p.objectness_smooth != 0u)
				{
					if (delta[extra_obj_index] == 0.0f)
					{
						delta[extra_obj_index] = extra_delta_obj;
					}
				}
				else
				{
					delta[extra_obj_index] = extra_delta_obj;
				}

				apple_yolo_assign_class_delta(output, delta, p, b, extra_n, loc, class_id);
			}
		}
	}

	stats[b].iou_loss += iou_loss;
	stats[b].iou_sum += iou_sum;
	stats[b].count += count;
	stats[b].rewritten += rewritten;
}

kernel void yolo_plain_average_thresh_deltas(
	device float *delta [[buffer(0)]],
	constant apple_yolo_plain_params &p [[buffer(1)]],
	uint gid [[thread_position_in_grid]])
{
	if (p.iou_thresh >= 1.0f)
	{
		return;
	}
	const uint spatial = apple_yolo_spatial(p);
	const uint predictions = p.batch * p.n * spatial;
	if (gid >= predictions)
	{
		return;
	}

	const uint b = gid / (p.n * spatial);
	const uint rem = gid - b * p.n * spatial;
	const uint n = rem / spatial;
	const uint loc = rem - n * spatial;
	const uint obj_index = apple_yolo_entry_index(p, b, n, loc, 4u);
	if (delta[obj_index] == 0.0f)
	{
		return;
	}

	uint classes_in_one_box = 0u;
	for (uint c = 0; c < p.classes; ++c)
	{
		const uint class_index = apple_yolo_entry_index(p, b, n, loc, 5u + c);
		if (delta[class_index] > 0.0f)
		{
			++classes_in_one_box;
		}
	}
	if (classes_in_one_box == 0u)
	{
		return;
	}

	const float scale = 1.0f / float(classes_in_one_box);
	const uint box_index = apple_yolo_entry_index(p, b, n, loc, 0u);
	delta[box_index + 0u * spatial] *= scale;
	delta[box_index + 1u * spatial] *= scale;
	delta[box_index + 2u * spatial] *= scale;
	delta[box_index + 3u * spatial] *= scale;
}

kernel void yolo_plain_batch_loss_stats(
	device const float *delta [[buffer(0)]],
	device apple_yolo_plain_loss_stats *stats [[buffer(1)]],
	constant apple_yolo_plain_params &p [[buffer(2)]],
	uint b [[thread_position_in_grid]])
{
	if (b >= p.batch)
	{
		return;
	}
	const uint spatial = apple_yolo_spatial(p);
	float class_sq = 0.0f;
	for (uint n = 0; n < p.n; ++n)
	{
		for (uint entry = 0; entry < p.classes + 5u; ++entry)
		{
			for (uint loc = 0; loc < spatial; ++loc)
			{
				const float raw_delta = delta[apple_yolo_entry_index(p, b, n, loc, entry)];
				const float d = isfinite(raw_delta) ? raw_delta : 0.0f;
				const float sq = d * d;
				if (entry >= 4u)
				{
					class_sq += sq;
				}
			}
		}
	}
	stats[b].total_sq = 0.0f;
	stats[b].class_sq = p.obj_normalizer * class_sq;
}


kernel void bn_rolling_update_kernel(device float *rolling_mean [[buffer(0)]],
	device float *rolling_variance [[buffer(1)]],
	device const float *mean [[buffer(2)]],
	device const float *variance [[buffer(3)]],
	constant float &keep [[buffer(4)]],
	constant float &add [[buffer(5)]],
	constant uint &channels [[buffer(6)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= channels)
	{
		return;
	}
	rolling_mean[gid] = rolling_mean[gid] * keep + mean[gid] * add;
	rolling_variance[gid] = rolling_variance[gid] * keep + variance[gid] * add;
}
)METAL";

		bool ensure_library(id<MTLDevice> device, MetalKernelCache & cache)
		{
			if (cache.library or cache.failed)
			{
				return cache.library != nil;
			}

			MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
			options.languageVersion = MTLLanguageVersion4_0;
			options.mathMode = MTLMathModeRelaxed;
			options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
			NSError *error = nil;
			cache.library = [device newLibraryWithSource:@(kMetalKernelSource) options:options error:&error];
			objc_release_if_needed(options);
			if (not cache.library)
			{
				cache.failed = true;
				cache.error = "Metal library compilation failed: " + ns_error_string(error);
				return false;
			}
			cache.library.label = @"Darknet runtime kernels";
			cache.error.clear();
			return true;
		}

		id<MTLComputePipelineState> get_pipeline(const std::string & kernel_name, id<MTLDevice> device)
		{
			if (kernel_name.empty() or not device)
			{
				return nil;
			}

			auto & cache = get_kernel_cache();
			std::scoped_lock lock(cache.mutex);
			auto it = cache.pipelines.find(kernel_name);
			if (it != cache.pipelines.end())
			{
				return (id<MTLComputePipelineState>)it->second.get();
			}
			if (not ensure_library(device, cache))
			{
				return nil;
			}
			id<MTLFunction> fn = [cache.library newFunctionWithName:@(kernel_name.c_str())];
			if (not fn)
			{
				cache.error = std::string("Metal function not found: ") + kernel_name;
				return nil;
			}
			NSError *error = nil;
			id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
			if (not pipeline)
			{
				cache.error = std::string("Metal pipeline creation failed for ") + kernel_name + ": " + ns_error_string(error);
				return nil;
			}
			cache.pipelines.emplace(kernel_name, ObjcStrong(pipeline));
			cache.error.clear();
			return pipeline;
		}

		// MPS uses a plain 2D texture for a single image with <= 4 channels.
		// Every texture kernel in this library expects a 2D array. A one-slice
		// view shares the original storage, including writes, without a copy.
		id<MTLTexture> array_texture(const MetalDispatchTexture & binding)
		{
			id<MTLTexture> texture = (__bridge id<MTLTexture>)binding.handle;
			if (texture.textureType == MTLTextureType2D)
			{
				texture = [texture newTextureViewWithPixelFormat:texture.pixelFormat
					textureType:MTLTextureType2DArray
					levels:NSMakeRange(0, texture.mipmapLevelCount)
					slices:NSMakeRange(0, 1)];
			}
			if (not texture or texture.textureType != MTLTextureType2DArray)
			{
				thread_ctx.error = "Metal kernel requires a 2D or 2D-array texture";
				return nil;
			}
			return texture;
		}

		/// Acquire a command buffer for a dispatch, auto-committing if created here.
		bool begin_dispatch(__strong id<MTLCommandBuffer> & command_buffer, bool & auto_commit, void *external_buffer)
		{
			auto_commit = false;
			thread_ctx.error.clear();
			if (external_buffer)
			{
				command_buffer = (__bridge id<MTLCommandBuffer>)external_buffer;
				return command_buffer != nil;
			}
			if (thread_ctx.command_buffer)
			{
				command_buffer = thread_ctx.command_buffer;
				return true;
			}
			metal_begin_frame();
			if (not thread_ctx.command_buffer)
			{
				return false;
			}
			command_buffer = thread_ctx.command_buffer;
			auto_commit = true;
			return true;
		}

		/// Commit and wait if this dispatch created the command buffer.
		bool end_dispatch(id<MTLCommandBuffer> command_buffer, const bool auto_commit)
		{
			if (not auto_commit)
			{
				return true;
			}
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
			const bool ok = command_buffer_succeeded(command_buffer);
			thread_ctx.command_buffer = nil;
			thread_ctx.owns_command_buffer = false;
			return ok;
		}
	}

bool metal_is_available()
	{
		auto & ctx = get_context();
		std::scoped_lock lock(ctx.mutex);
		if (ctx.device and ctx.queue)
		{
			return true;
		}
		ctx.device = (__bridge id<MTLDevice>)Darknet::Apple::runtime_metal_device_handle();
		ctx.queue = (__bridge id<MTLCommandQueue>)Darknet::Apple::runtime_metal_command_queue_handle();
		if (not ctx.device or not ctx.queue)
		{
			ctx.device = nil;
			ctx.queue = nil;
			return false;
		}
		return true;
	}

	bool metal_kernel_available(const std::string & kernel_name)
	{
		if (kernel_name.empty() or not metal_is_available())
		{
			return false;
		}
		auto & ctx = get_context();
		@autoreleasepool
		{
			return get_pipeline(kernel_name, ctx.device) != nil;
		}
	}

	std::string metal_last_error()
	{
		if (not thread_ctx.error.empty())
		{
			return thread_ctx.error;
		}
		auto & cache = get_kernel_cache();
		std::scoped_lock lock(cache.mutex);
		return cache.error;
	}

	bool metal_init()
	{
		return metal_is_available();
	}

	/// Begin a thread-local command buffer to batch multiple dispatches.
	void metal_begin_frame()
	{
		if (not metal_is_available())
		{
			return;
		}
		if (thread_ctx.command_buffer)
		{
			return;
		}
		auto & ctx = get_context();
		std::scoped_lock lock(ctx.mutex);
		thread_ctx.command_buffer = [ctx.queue commandBuffer];
		thread_ctx.command_buffer.label = @"Darknet Metal frame";
		thread_ctx.owns_command_buffer = true;
	}

	/// Commit and wait for the thread-local command buffer.
	void metal_end_frame()
	{
		if (not thread_ctx.command_buffer or not thread_ctx.owns_command_buffer)
		{
			return;
		}
		[thread_ctx.command_buffer commit];
		[thread_ctx.command_buffer waitUntilCompleted];
		(void)command_buffer_succeeded(thread_ctx.command_buffer);
		thread_ctx.command_buffer = nil;
		thread_ctx.owns_command_buffer = false;
	}

	/// Commit and wait for outstanding thread-local work, if any.
	void metal_flush()
	{
		if (not thread_ctx.command_buffer or not thread_ctx.owns_command_buffer)
		{
			return;
		}
		[thread_ctx.command_buffer commit];
		[thread_ctx.command_buffer waitUntilCompleted];
		(void)command_buffer_succeeded(thread_ctx.command_buffer);
		thread_ctx.command_buffer = nil;
		thread_ctx.owns_command_buffer = false;
	}

	bool metal_buffer_alloc(const size_t size, MetalBuffer *out)
	{
		if (not out or size == 0)
		{
			return false;
		}
		if (not metal_is_available())
		{
			return false;
		}
		auto & ctx = get_context();
		std::scoped_lock lock(ctx.mutex);
		id<MTLBuffer> buffer = [ctx.device newBufferWithLength:size options:MTLResourceStorageModeShared];
		if (not buffer)
		{
			return false;
		}
#if __has_feature(objc_arc)
		out->handle = (__bridge_retained void *)buffer;
#else
		out->handle = buffer;
	} // namespace Darknet::Apple

#endif
		out->size = size;
		return true;
	}

	void metal_buffer_free(MetalBuffer *buffer)
	{
		if (not buffer or not buffer->handle)
		{
			return;
		}
#if __has_feature(objc_arc)
		__unused id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer->handle;
#else
	id<MTLBuffer> buf = (id<MTLBuffer>)buffer->handle;
	[buf release];
} // namespace Darknet::Apple

#endif
		buffer->handle = nullptr;
		buffer->size = 0;
	}

	bool metal_buffer_upload(const MetalBuffer *buffer, const size_t offset, const void *data, const size_t length)
	{
		if (not buffer or not buffer->handle or not data or length == 0)
		{
			return false;
		}
		if (offset + length > buffer->size)
		{
			return false;
		}
		id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer->handle;
		std::memcpy(static_cast<uint8_t *>([buf contents]) + offset, data, length);
		return true;
	}

	bool metal_buffer_download(const MetalBuffer *buffer, const size_t offset, void *data, const size_t length)
	{
		if (not buffer or not buffer->handle or not data or length == 0)
		{
			return false;
		}
		if (offset + length > buffer->size)
		{
			return false;
		}
		id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer->handle;
		std::memcpy(data, static_cast<uint8_t *>([buf contents]) + offset, length);
		return true;
	}

	bool metal_buffer_fill(const MetalBuffer *buffer, const size_t offset, uint8_t value, const size_t length)
	{
		if (not buffer or not buffer->handle or length == 0)
		{
			return false;
		}
		if (offset + length > buffer->size)
		{
			return false;
		}
		id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer->handle;
		std::memset(static_cast<uint8_t *>([buf contents]) + offset, value, length);
		return true;
	}

	bool metal_dispatch_1d(const std::string & kernel_name,
		const size_t threads, const size_t threads_per_group,
		const MetalDispatchTexture *textures, const size_t texture_count,
		const MetalDispatchBuffer *buffers, const size_t buffer_count,
		const MetalDispatchBytes *bytes, const size_t bytes_count,
		void *command_buffer_handle)
	{
		if (kernel_name.empty() or threads == 0 or threads_per_group == 0)
		{
			return false;
		}
		if (not metal_is_available())
		{
			return false;
		}
		auto & ctx = get_context();
		id<MTLCommandBuffer> command_buffer = nil;
		bool auto_commit = false;
		if (not begin_dispatch(command_buffer, auto_commit, command_buffer_handle))
		{
			return false;
		}
		@autoreleasepool
		{
			id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name, ctx.device);
			if (not pipeline)
			{
				end_dispatch(command_buffer, auto_commit);
				return false;
			}
			const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
			if (threads_per_group > max_threads)
			{
				end_dispatch(command_buffer, auto_commit);
				return false;
			}

			id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
			if (not encoder)
			{
				end_dispatch(command_buffer, auto_commit);
				return false;
			}
			encoder.label = @(kernel_name.c_str());
			[encoder setComputePipelineState:pipeline];

			for (size_t i = 0; i < texture_count; ++i)
			{
				id<MTLTexture> texture = textures ? array_texture(textures[i]) : nil;
				if (not texture)
				{
					[encoder endEncoding];
					end_dispatch(command_buffer, auto_commit);
					return false;
				}
				[encoder setTexture:texture atIndex:i];
			}

			size_t buffer_index = 0;
			for (size_t i = 0; i < buffer_count; ++i)
			{
				if (not buffers or not buffers[i].buffer or not buffers[i].buffer->handle)
				{
					continue;
				}
				id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i].buffer->handle;
				[encoder setBuffer:buffer offset:buffers[i].offset atIndex:buffer_index];
				++buffer_index;
			}

			for (size_t i = 0; i < bytes_count; ++i)
			{
				if (not bytes or not bytes[i].bytes or bytes[i].length == 0)
				{
					continue;
				}
				[encoder setBytes:bytes[i].bytes length:bytes[i].length atIndex:buffer_index];
				++buffer_index;
			}

			const MTLSize grid = MTLSizeMake(threads, 1, 1);
			const MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
			[encoder dispatchThreads:grid threadsPerThreadgroup:group];
			[encoder endEncoding];

			return end_dispatch(command_buffer, auto_commit);
		}
	}

	bool metal_dispatch_1d_with_encoder(const std::string & kernel_name,
		const size_t threads, const size_t threads_per_group,
		const MetalDispatchTexture *textures, const size_t texture_count,
		const MetalDispatchBuffer *buffers, const size_t buffer_count,
		const MetalDispatchBytes *bytes, const size_t bytes_count,
		void *encoder_handle)
	{
		if (kernel_name.empty() or threads == 0 or threads_per_group == 0 or not encoder_handle)
		{
			return false;
		}
		if (not metal_is_available())
		{
			return false;
		}
		auto & ctx = get_context();
		@autoreleasepool
		{
			id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name, ctx.device);
			if (not pipeline)
			{
				return false;
			}
			const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
			if (threads_per_group > max_threads)
			{
				return false;
			}

			id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
			if (not encoder)
			{
				return false;
			}
			encoder.label = @(kernel_name.c_str());
			[encoder setComputePipelineState:pipeline];

			for (size_t i = 0; i < texture_count; ++i)
			{
				id<MTLTexture> texture = textures ? array_texture(textures[i]) : nil;
				if (not texture)
				{
					return false;
				}
				[encoder setTexture:texture atIndex:i];
			}

			size_t buffer_index = 0;
			for (size_t i = 0; i < buffer_count; ++i)
			{
				if (not buffers or not buffers[i].buffer or not buffers[i].buffer->handle)
				{
					continue;
				}
				id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i].buffer->handle;
				[encoder setBuffer:buffer offset:buffers[i].offset atIndex:buffer_index];
				++buffer_index;
			}

			for (size_t i = 0; i < bytes_count; ++i)
			{
				if (not bytes or not bytes[i].bytes or bytes[i].length == 0)
				{
					continue;
				}
				[encoder setBytes:bytes[i].bytes length:bytes[i].length atIndex:buffer_index];
				++buffer_index;
			}

			const MTLSize grid = MTLSizeMake(threads, 1, 1);
			const MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
			[encoder dispatchThreads:grid threadsPerThreadgroup:group];
			return true;
		}
	}

	bool metal_dispatch_2d(const std::string & kernel_name,
		const size_t width, const size_t height,
		const size_t threads_per_group_x, const size_t threads_per_group_y,
		const MetalDispatchTexture *textures, const size_t texture_count,
		const MetalDispatchBuffer *buffers, const size_t buffer_count,
		const MetalDispatchBytes *bytes, const size_t bytes_count,
		void *command_buffer_handle)
	{
		if (kernel_name.empty() or width == 0 or height == 0)
		{
			return false;
		}
		if (not metal_is_available())
		{
			return false;
		}
		auto & ctx = get_context();
		id<MTLCommandBuffer> command_buffer = nil;
		bool auto_commit = false;
		if (not begin_dispatch(command_buffer, auto_commit, command_buffer_handle))
		{
			return false;
		}
		@autoreleasepool
		{
			id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name, ctx.device);
			if (not pipeline)
			{
				end_dispatch(command_buffer, auto_commit);
				return false;
			}
			const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
			const size_t group_threads = threads_per_group_x * threads_per_group_y;
			if (threads_per_group_x == 0 or threads_per_group_y == 0 or group_threads > max_threads)
			{
				end_dispatch(command_buffer, auto_commit);
				return false;
			}

			id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
			if (not encoder)
			{
				end_dispatch(command_buffer, auto_commit);
				return false;
			}
			encoder.label = @(kernel_name.c_str());
			[encoder setComputePipelineState:pipeline];

			NSUInteger depth = 1;
			for (size_t i = 0; i < texture_count; ++i)
			{
				id<MTLTexture> texture = textures ? array_texture(textures[i]) : nil;
				if (not texture)
				{
					[encoder endEncoding];
					end_dispatch(command_buffer, auto_commit);
					return false;
				}
				[encoder setTexture:texture atIndex:i];
				if (texture.arrayLength > depth)
				{
					depth = texture.arrayLength;
				}
			}

			size_t buffer_index = 0;
			for (size_t i = 0; i < buffer_count; ++i)
			{
				if (not buffers or not buffers[i].buffer or not buffers[i].buffer->handle)
				{
					continue;
				}
				id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i].buffer->handle;
				[encoder setBuffer:buffer offset:buffers[i].offset atIndex:buffer_index];
				++buffer_index;
			}

			for (size_t i = 0; i < bytes_count; ++i)
			{
				if (not bytes or not bytes[i].bytes or bytes[i].length == 0)
				{
					continue;
				}
				[encoder setBytes:bytes[i].bytes length:bytes[i].length atIndex:buffer_index];
				++buffer_index;
			}

			const MTLSize grid = MTLSizeMake(width, height, depth);
			const MTLSize group = MTLSizeMake(threads_per_group_x, threads_per_group_y, 1);
			[encoder dispatchThreads:grid threadsPerThreadgroup:group];
			[encoder endEncoding];

			return end_dispatch(command_buffer, auto_commit);
		}
	}

	bool metal_dispatch_texture_kernel(const std::string & kernel_name,
		void *grid_texture_handle,
		const size_t threads_per_group_x, const size_t threads_per_group_y,
		const MetalDispatchTexture *textures, const size_t texture_count,
		const MetalDispatchBuffer *buffers, const size_t buffer_count,
		const MetalDispatchBytes *bytes, const size_t bytes_count,
		void *command_buffer_handle)
	{
		if (not grid_texture_handle)
		{
			return false;
		}
		id<MTLTexture> grid_texture = (__bridge id<MTLTexture>)grid_texture_handle;
		if (not grid_texture)
		{
			return false;
		}
		const size_t width = static_cast<size_t>(grid_texture.width);
		const size_t height = static_cast<size_t>(grid_texture.height);
		if (width == 0 or height == 0)
		{
			return false;
		}
		return metal_dispatch_2d(kernel_name, width, height,
			threads_per_group_x, threads_per_group_y,
			textures, texture_count,
			buffers, buffer_count,
			bytes, bytes_count,
			command_buffer_handle);
	}

} // namespace Darknet::Apple

#endif
