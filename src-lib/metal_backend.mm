#include "metal_backend.hpp"

#ifdef DARKNET_USE_MPS

#import <Metal/Metal.h>

#include <mutex>
#include <unordered_map>
#include <string>
#include <cstring>
#include <cstdio>

namespace
{
	struct MetalContext
	{
		id<MTLDevice> device = nil;
		id<MTLCommandQueue> queue = nil;
		std::mutex mutex;
	};

	MetalContext & get_context()
	{
		static MetalContext ctx;
		return ctx;
	}

	struct MetalThreadContext
	{
		id<MTLCommandBuffer> command_buffer = nil;
		bool owns_command_buffer = false;
	};

	thread_local MetalThreadContext thread_ctx;

	struct MetalKernelCache
	{
		id<MTLLibrary> library = nil;
		std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
		std::mutex mutex;
		bool failed = false;
	};

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

kernel void shortcut_weighted_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::read> add_tex [[texture(1)]],
	texture2d_array<float, access::write> out_tex [[texture(2)]],
	constant float &w_in [[buffer(0)]],
	constant float &w_add [[buffer(1)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	float4 a = in_tex.read(uint2(gid.xy), gid.z);
	float4 b = add_tex.read(uint2(gid.xy), gid.z);
	out_tex.write(a * w_in + b * w_add, uint2(gid.xy), gid.z);
}

kernel void shortcut_weighted_relu_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::read> add_tex [[texture(1)]],
	texture2d_array<float, access::write> out_tex [[texture(2)]],
	constant float &w_in [[buffer(0)]],
	constant float &w_add [[buffer(1)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const float eps = 0.0001f;
	const float w0 = max(w_in, 0.0f);
	const float w1 = max(w_add, 0.0f);
	const float sum = eps + w0 + w1;
	float4 a = in_tex.read(uint2(gid.xy), gid.z);
	float4 b = add_tex.read(uint2(gid.xy), gid.z);
	out_tex.write(a * (w0 / sum) + b * (w1 / sum), uint2(gid.xy), gid.z);
}

kernel void shortcut_weighted_softmax_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::read> add_tex [[texture(1)]],
	texture2d_array<float, access::write> out_tex [[texture(2)]],
	constant float &w_in [[buffer(0)]],
	constant float &w_add [[buffer(1)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const float eps = 0.0001f;
	const float m = max(w_in, w_add);
	const float e0 = exp(w_in - m);
	const float e1 = exp(w_add - m);
	const float sum = eps + e0 + e1;
	float4 a = in_tex.read(uint2(gid.xy), gid.z);
	float4 b = add_tex.read(uint2(gid.xy), gid.z);
	out_tex.write(a * (e0 / sum) + b * (e1 / sum), uint2(gid.xy), gid.z);
}

kernel void shortcut_per_channel_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::read> add_tex [[texture(1)]],
	texture2d_array<float, access::write> out_tex [[texture(2)]],
	device const float *w_in [[buffer(0)]],
	device const float *w_add [[buffer(1)]],
	constant uint &channels [[buffer(2)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const uint base = gid.z * 4;
	float4 a = in_tex.read(uint2(gid.xy), gid.z);
	float4 b = add_tex.read(uint2(gid.xy), gid.z);
	float4 out;
	const uint c0 = base;
	const uint c1 = base + 1;
	const uint c2 = base + 2;
	const uint c3 = base + 3;
	const float w0 = (c0 < channels) ? w_in[c0] : 0.0f;
	const float w1 = (c1 < channels) ? w_in[c1] : 0.0f;
	const float w2 = (c2 < channels) ? w_in[c2] : 0.0f;
	const float w3 = (c3 < channels) ? w_in[c3] : 0.0f;
	const float wa0 = (c0 < channels) ? w_add[c0] : 0.0f;
	const float wa1 = (c1 < channels) ? w_add[c1] : 0.0f;
	const float wa2 = (c2 < channels) ? w_add[c2] : 0.0f;
	const float wa3 = (c3 < channels) ? w_add[c3] : 0.0f;
	out.x = a.x * w0 + b.x * wa0;
	out.y = a.y * w1 + b.y * wa1;
	out.z = a.z * w2 + b.z * wa2;
	out.w = a.w * w3 + b.w * wa3;
	out_tex.write(out, uint2(gid.xy), gid.z);
}

kernel void shortcut_per_channel_relu_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::read> add_tex [[texture(1)]],
	texture2d_array<float, access::write> out_tex [[texture(2)]],
	device const float *w_in [[buffer(0)]],
	device const float *w_add [[buffer(1)]],
	constant uint &channels [[buffer(2)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const float eps = 0.0001f;
	const uint base = gid.z * 4;
	float4 a = in_tex.read(uint2(gid.xy), gid.z);
	float4 b = add_tex.read(uint2(gid.xy), gid.z);
	float4 out;
	const uint c0 = base;
	const uint c1 = base + 1;
	const uint c2 = base + 2;
	const uint c3 = base + 3;
	const float w0 = (c0 < channels) ? max(w_in[c0], 0.0f) : 0.0f;
	const float w1 = (c1 < channels) ? max(w_in[c1], 0.0f) : 0.0f;
	const float w2 = (c2 < channels) ? max(w_in[c2], 0.0f) : 0.0f;
	const float w3 = (c3 < channels) ? max(w_in[c3], 0.0f) : 0.0f;
	const float wa0 = (c0 < channels) ? max(w_add[c0], 0.0f) : 0.0f;
	const float wa1 = (c1 < channels) ? max(w_add[c1], 0.0f) : 0.0f;
	const float wa2 = (c2 < channels) ? max(w_add[c2], 0.0f) : 0.0f;
	const float wa3 = (c3 < channels) ? max(w_add[c3], 0.0f) : 0.0f;
	const float s0 = eps + w0 + wa0;
	const float s1 = eps + w1 + wa1;
	const float s2 = eps + w2 + wa2;
	const float s3 = eps + w3 + wa3;
	out.x = (s0 > 0.0f) ? (a.x * (w0 / s0) + b.x * (wa0 / s0)) : 0.0f;
	out.y = (s1 > 0.0f) ? (a.y * (w1 / s1) + b.y * (wa1 / s1)) : 0.0f;
	out.z = (s2 > 0.0f) ? (a.z * (w2 / s2) + b.z * (wa2 / s2)) : 0.0f;
	out.w = (s3 > 0.0f) ? (a.w * (w3 / s3) + b.w * (wa3 / s3)) : 0.0f;
	out_tex.write(out, uint2(gid.xy), gid.z);
}

kernel void shortcut_per_channel_softmax_kernel(texture2d_array<float, access::read> in_tex [[texture(0)]],
	texture2d_array<float, access::read> add_tex [[texture(1)]],
	texture2d_array<float, access::write> out_tex [[texture(2)]],
	device const float *w_in [[buffer(0)]],
	device const float *w_add [[buffer(1)]],
	constant uint &channels [[buffer(2)]],
	uint3 gid [[thread_position_in_grid]])
{
	if (gid.x >= out_tex.get_width() || gid.y >= out_tex.get_height() || gid.z >= out_tex.get_array_size())
	{
		return;
	}
	const float eps = 0.0001f;
	const uint base = gid.z * 4;
	float4 a = in_tex.read(uint2(gid.xy), gid.z);
	float4 b = add_tex.read(uint2(gid.xy), gid.z);
	float4 out;
	const uint c0 = base;
	const uint c1 = base + 1;
	const uint c2 = base + 2;
	const uint c3 = base + 3;
	const float w0 = (c0 < channels) ? w_in[c0] : 0.0f;
	const float w1 = (c1 < channels) ? w_in[c1] : 0.0f;
	const float w2 = (c2 < channels) ? w_in[c2] : 0.0f;
	const float w3 = (c3 < channels) ? w_in[c3] : 0.0f;
	const float wa0 = (c0 < channels) ? w_add[c0] : 0.0f;
	const float wa1 = (c1 < channels) ? w_add[c1] : 0.0f;
	const float wa2 = (c2 < channels) ? w_add[c2] : 0.0f;
	const float wa3 = (c3 < channels) ? w_add[c3] : 0.0f;
	const float m0 = max(w0, wa0);
	const float m1 = max(w1, wa1);
	const float m2 = max(w2, wa2);
	const float m3 = max(w3, wa3);
	const float e0 = exp(w0 - m0);
	const float e1 = exp(w1 - m1);
	const float e2 = exp(w2 - m2);
	const float e3 = exp(w3 - m3);
	const float ea0 = exp(wa0 - m0);
	const float ea1 = exp(wa1 - m1);
	const float ea2 = exp(wa2 - m2);
	const float ea3 = exp(wa3 - m3);
	const float s0 = eps + e0 + ea0;
	const float s1 = eps + e1 + ea1;
	const float s2 = eps + e2 + ea2;
	const float s3 = eps + e3 + ea3;
	out.x = (s0 > 0.0f) ? (a.x * (e0 / s0) + b.x * (ea0 / s0)) : 0.0f;
	out.y = (s1 > 0.0f) ? (a.y * (e1 / s1) + b.y * (ea1 / s1)) : 0.0f;
	out.z = (s2 > 0.0f) ? (a.z * (e2 / s2) + b.z * (ea2 / s2)) : 0.0f;
	out.w = (s3 > 0.0f) ? (a.w * (e3 / s3) + b.w * (ea3 / s3)) : 0.0f;
	out_tex.write(out, uint2(gid.xy), gid.z);
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

kernel void yolo_activate_kernel(device float *data [[buffer(0)]],
	constant uint &entries [[buffer(1)]],
	constant uint &wh [[buffer(2)]],
	constant float &scale_x_y [[buffer(3)]],
	constant float &bias [[buffer(4)]],
	constant uint &new_coords [[buffer(5)]],
	constant uint &total [[buffer(6)]],
	uint gid [[thread_position_in_grid]])
{
	if (gid >= total)
	{
		return;
	}
	const uint entry = (wh > 0u) ? ((gid / wh) % entries) : 0u;
	float v = data[gid];
	if (new_coords == 0u)
	{
		if (entry == 0u || entry == 1u || entry >= 4u)
		{
			v = 1.0f / (1.0f + exp(-v));
		}
	}
	if (entry == 0u || entry == 1u)
	{
		v = v * scale_x_y + bias;
	}
	data[gid] = v;
}

kernel void yolo_decode_boxes_kernel(device const float *data [[buffer(0)]],
	device float4 *boxes [[buffer(1)]],
	device const float *biases [[buffer(2)]],
	device const int *mask [[buffer(3)]],
	constant uint &w [[buffer(4)]],
	constant uint &h [[buffer(5)]],
	constant uint &entries [[buffer(6)]],
	constant uint &n [[buffer(7)]],
	constant uint &batch [[buffer(8)]],
	constant uint &outputs [[buffer(9)]],
	constant uint &netw [[buffer(10)]],
	constant uint &neth [[buffer(11)]],
	constant uint &new_coords [[buffer(12)]],
	uint gid [[thread_position_in_grid]])
{
	const uint wh = w * h;
	const uint total = wh * n * batch;
	if (gid >= total || wh == 0u || n == 0u || w == 0u || h == 0u)
	{
		return;
	}
	const uint loc = gid % wh;
	const uint anchor = (gid / wh) % n;
	const uint b = gid / (wh * n);
	const uint base = b * outputs + anchor * wh * entries;
	const uint idx = base + loc;

	const uint row = loc / w;
	const uint col = loc - row * w;

	const uint bias_idx = (mask != nullptr) ? (uint)mask[anchor] : anchor;
	const uint bias_base = bias_idx * 2u;
	const float bw = biases[bias_base];
	const float bh = biases[bias_base + 1u];

	const float x = data[idx + 0u * wh];
	const float y = data[idx + 1u * wh];
	const float tw = data[idx + 2u * wh];
	const float th = data[idx + 3u * wh];

	const float bx = (float(col) + x) / float(w);
	const float by = (float(row) + y) / float(h);
	float bw_out = 0.0f;
	float bh_out = 0.0f;

	if (new_coords == 0u)
	{
		bw_out = exp(tw) * bw / float(netw);
		bh_out = exp(th) * bh / float(neth);
	}
	else
	{
		const float tw2 = tw * tw;
		const float th2 = th * th;
		bw_out = tw2 * 4.0f * bw / float(netw);
		bh_out = th2 * 4.0f * bh / float(neth);
	}

	boxes[gid] = float4(bx, by, bw_out, bh_out);
}

kernel void yolo_candidates_kernel(device const float *data [[buffer(0)]],
	device uint *indices [[buffer(1)]],
	device atomic_uint *count [[buffer(2)]],
	constant uint &w [[buffer(3)]],
	constant uint &h [[buffer(4)]],
	constant uint &entries [[buffer(5)]],
	constant uint &n [[buffer(6)]],
	constant uint &batch [[buffer(7)]],
	constant uint &outputs [[buffer(8)]],
	constant float &thresh [[buffer(9)]],
	constant uint &max_candidates [[buffer(10)]],
	uint gid [[thread_position_in_grid]])
{
	const uint wh = w * h;
	const uint total = wh * n * batch;
	if (gid >= total || wh == 0u || n == 0u)
	{
		return;
	}
	const uint loc = gid % wh;
	const uint anchor = (gid / wh) % n;
	const uint b = gid / (wh * n);
	const uint base = b * outputs + anchor * wh * entries;
	const uint obj_index = base + 4u * wh + loc;
	const float objectness = data[obj_index];
	if (objectness > thresh)
	{
		const uint idx = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
		if (idx < max_candidates)
		{
			indices[idx] = gid;
		}
	}
}

kernel void nms_suppress_kernel(device const float4 *boxes [[buffer(0)]],
	device float *scores [[buffer(1)]],
	device const uint *order [[buffer(2)]],
	constant uint &count [[buffer(3)]],
	constant uint &base [[buffer(4)]],
	constant float &thresh [[buffer(5)]],
	uint gid [[thread_position_in_grid]])
{
	const uint j = base + 1u + gid;
	if (j >= count)
	{
		return;
	}
	const uint idx_i = order[base];
	const uint idx_j = order[j];

	const float score_i = scores[idx_i];
	if (score_i <= 0.0f)
	{
		return;
	}
	const float score_j = scores[idx_j];
	if (score_j <= 0.0f)
	{
		return;
	}

	const float4 a = boxes[idx_i];
	const float4 b = boxes[idx_j];

	const float ax1 = a.x - a.z * 0.5f;
	const float ay1 = a.y - a.w * 0.5f;
	const float ax2 = a.x + a.z * 0.5f;
	const float ay2 = a.y + a.w * 0.5f;
	const float bx1 = b.x - b.z * 0.5f;
	const float by1 = b.y - b.w * 0.5f;
	const float bx2 = b.x + b.z * 0.5f;
	const float by2 = b.y + b.w * 0.5f;

	const float iw = max(0.0f, min(ax2, bx2) - max(ax1, bx1));
	const float ih = max(0.0f, min(ay2, by2) - max(ay1, by1));
	const float inter = iw * ih;
	const float area_a = a.z * a.w;
	const float area_b = b.z * b.w;
	const float uni = area_a + area_b - inter;
	if (uni <= 0.0f)
	{
		return;
	}
	const float iou = inter / uni;
	if (iou > thresh)
	{
		scores[idx_j] = 0.0f;
	}
}

kernel void buffer_scale(device float *data [[buffer(0)]],
	constant float &scale [[buffer(1)]],
	uint gid [[thread_position_in_grid]])
{
	data[gid] *= scale;
}
)METAL";

	bool ensure_library(id<MTLDevice> device, MetalKernelCache & cache)
	{
		if (cache.library || cache.failed)
		{
			return cache.library != nil;
		}

		NSError *error = nil;
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		cache.library = [device newLibraryWithSource:@(kMetalKernelSource) options:options error:&error];
		if (!cache.library)
		{
			cache.failed = true;
			if (error)
			{
				std::fprintf(stderr, "[Metal] kernel library compile failed: %s\n",
					error.localizedDescription.UTF8String);
			}
			return false;
		}
		return true;
	}

	id<MTLComputePipelineState> get_pipeline(const char *kernel_name, id<MTLDevice> device)
	{
		if (!kernel_name || !device)
		{
			return nil;
		}

		auto & cache = get_kernel_cache();
		std::scoped_lock lock(cache.mutex);
		auto it = cache.pipelines.find(kernel_name);
		if (it != cache.pipelines.end())
		{
			return it->second;
		}
		if (!ensure_library(device, cache))
		{
			return nil;
		}
		id<MTLFunction> fn = [cache.library newFunctionWithName:@(kernel_name)];
		if (!fn)
		{
			std::fprintf(stderr, "[Metal] kernel function not found: %s\n", kernel_name);
			return nil;
		}
		NSError *error = nil;
		id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
		if (!pipeline)
		{
			if (error)
			{
				std::fprintf(stderr, "[Metal] kernel pipeline failed (%s): %s\n",
					kernel_name, error.localizedDescription.UTF8String);
			}
			return nil;
		}
		cache.pipelines.emplace(kernel_name, pipeline);
		return pipeline;
	}

	bool begin_dispatch(id<MTLCommandBuffer> & command_buffer, bool & auto_commit, void *external_buffer)
	{
		auto_commit = false;
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
		if (!thread_ctx.command_buffer)
		{
			return false;
		}
		command_buffer = thread_ctx.command_buffer;
		auto_commit = true;
		return true;
	}

	void end_dispatch(id<MTLCommandBuffer> command_buffer, bool auto_commit)
	{
		if (!auto_commit)
		{
			return;
		}
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
		thread_ctx.command_buffer = nil;
		thread_ctx.owns_command_buffer = false;
	}
}

bool metal_is_available()
{
	auto & ctx = get_context();
	std::scoped_lock lock(ctx.mutex);
	if (ctx.device)
	{
		return true;
	}
	ctx.device = MTLCreateSystemDefaultDevice();
	if (!ctx.device)
	{
		return false;
	}
	ctx.queue = [ctx.device newCommandQueue];
	if (!ctx.queue)
	{
		ctx.device = nil;
		return false;
	}
	return true;
}

bool metal_init()
{
	return metal_is_available();
}

void metal_shutdown()
{
	auto & ctx = get_context();
	std::scoped_lock lock(ctx.mutex);
	ctx.queue = nil;
	ctx.device = nil;
}

void metal_begin_frame()
{
	if (!metal_is_available())
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
	thread_ctx.owns_command_buffer = true;
}

void metal_end_frame()
{
	if (!thread_ctx.command_buffer || !thread_ctx.owns_command_buffer)
	{
		return;
	}
	[thread_ctx.command_buffer commit];
	thread_ctx.command_buffer = nil;
	thread_ctx.owns_command_buffer = false;
}

void metal_flush()
{
	if (!thread_ctx.command_buffer || !thread_ctx.owns_command_buffer)
	{
		return;
	}
	[thread_ctx.command_buffer commit];
	[thread_ctx.command_buffer waitUntilCompleted];
	thread_ctx.command_buffer = nil;
	thread_ctx.owns_command_buffer = false;
}

bool metal_buffer_alloc(size_t size, MetalBuffer *out)
{
	if (!out || size == 0)
	{
		return false;
	}
	if (!metal_is_available())
	{
		return false;
	}
	auto & ctx = get_context();
	std::scoped_lock lock(ctx.mutex);
	id<MTLBuffer> buffer = [ctx.device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (!buffer)
	{
		return false;
	}
#if __has_feature(objc_arc)
	out->handle = (__bridge_retained void *)buffer;
#else
	out->handle = buffer;
#endif
	out->size = size;
	return true;
}

void metal_buffer_free(MetalBuffer *buffer)
{
	if (!buffer || !buffer->handle)
	{
		return;
	}
#if __has_feature(objc_arc)
	__unused id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer->handle;
#else
	id<MTLBuffer> buf = (id<MTLBuffer>)buffer->handle;
	[buf release];
#endif
	buffer->handle = nullptr;
	buffer->size = 0;
}

bool metal_buffer_upload(const MetalBuffer *buffer, size_t offset, const void *data, size_t length)
{
	if (!buffer || !buffer->handle || !data || length == 0)
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

bool metal_buffer_download(const MetalBuffer *buffer, size_t offset, void *data, size_t length)
{
	if (!buffer || !buffer->handle || !data || length == 0)
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

bool metal_buffer_fill(const MetalBuffer *buffer, size_t offset, uint8_t value, size_t length)
{
	if (!buffer || !buffer->handle || length == 0)
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

bool metal_dispatch_1d(const char *kernel_name,
	size_t threads, size_t threads_per_group,
	const MetalDispatchTexture *textures, size_t texture_count,
	const MetalDispatchBuffer *buffers, size_t buffer_count,
	const MetalDispatchBytes *bytes, size_t bytes_count,
	void *command_buffer_handle)
{
	if (!kernel_name || threads == 0 || threads_per_group == 0)
	{
		return false;
	}
	if (!metal_is_available())
	{
		return false;
	}
	auto & ctx = get_context();
	id<MTLCommandBuffer> command_buffer = nil;
	bool auto_commit = false;
	if (!begin_dispatch(command_buffer, auto_commit, command_buffer_handle))
	{
		return false;
	}
	id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name, ctx.device);
	if (!pipeline)
	{
		end_dispatch(command_buffer, auto_commit);
		return false;
	}

	id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
	if (!encoder)
	{
		end_dispatch(command_buffer, auto_commit);
		return false;
	}
	[encoder setComputePipelineState:pipeline];

	for (size_t i = 0; i < texture_count; ++i)
	{
		id<MTLTexture> texture = textures && textures[i].handle ? (__bridge id<MTLTexture>)textures[i].handle : nil;
		if (texture)
		{
			[encoder setTexture:texture atIndex:i];
		}
	}

	size_t buffer_index = 0;
	for (size_t i = 0; i < buffer_count; ++i)
	{
		if (!buffers || !buffers[i].buffer || !buffers[i].buffer->handle)
		{
			continue;
		}
		id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i].buffer->handle;
		[encoder setBuffer:buffer offset:buffers[i].offset atIndex:buffer_index];
		++buffer_index;
	}

	for (size_t i = 0; i < bytes_count; ++i)
	{
		if (!bytes || !bytes[i].bytes || bytes[i].length == 0)
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

	end_dispatch(command_buffer, auto_commit);
	return true;
}

bool metal_dispatch_2d(const char *kernel_name,
	size_t width, size_t height,
	size_t threads_per_group_x, size_t threads_per_group_y,
	const MetalDispatchTexture *textures, size_t texture_count,
	const MetalDispatchBuffer *buffers, size_t buffer_count,
	const MetalDispatchBytes *bytes, size_t bytes_count,
	void *command_buffer_handle)
{
	if (!kernel_name || width == 0 || height == 0)
	{
		return false;
	}
	if (!metal_is_available())
	{
		return false;
	}
	auto & ctx = get_context();
	id<MTLCommandBuffer> command_buffer = nil;
	bool auto_commit = false;
	if (!begin_dispatch(command_buffer, auto_commit, command_buffer_handle))
	{
		return false;
	}
	id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name, ctx.device);
	if (!pipeline)
	{
		end_dispatch(command_buffer, auto_commit);
		return false;
	}

	id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
	if (!encoder)
	{
		end_dispatch(command_buffer, auto_commit);
		return false;
	}
	[encoder setComputePipelineState:pipeline];

	NSUInteger depth = 1;
	for (size_t i = 0; i < texture_count; ++i)
	{
		id<MTLTexture> texture = textures && textures[i].handle ? (__bridge id<MTLTexture>)textures[i].handle : nil;
		if (texture)
		{
			[encoder setTexture:texture atIndex:i];
			if (texture.arrayLength > depth)
			{
				depth = texture.arrayLength;
			}
		}
	}

	size_t buffer_index = 0;
	for (size_t i = 0; i < buffer_count; ++i)
	{
		if (!buffers || !buffers[i].buffer || !buffers[i].buffer->handle)
		{
			continue;
		}
		id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i].buffer->handle;
		[encoder setBuffer:buffer offset:buffers[i].offset atIndex:buffer_index];
		++buffer_index;
	}

	for (size_t i = 0; i < bytes_count; ++i)
	{
		if (!bytes || !bytes[i].bytes || bytes[i].length == 0)
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

	end_dispatch(command_buffer, auto_commit);
	return true;
}

bool metal_dispatch_texture_kernel(const char *kernel_name,
	void *grid_texture_handle,
	size_t threads_per_group_x, size_t threads_per_group_y,
	const MetalDispatchTexture *textures, size_t texture_count,
	const MetalDispatchBuffer *buffers, size_t buffer_count,
	const MetalDispatchBytes *bytes, size_t bytes_count,
	void *command_buffer_handle)
{
	if (!grid_texture_handle)
	{
		return false;
	}
	id<MTLTexture> grid_texture = (__bridge id<MTLTexture>)grid_texture_handle;
	if (!grid_texture)
	{
		return false;
	}
	const size_t width = static_cast<size_t>(grid_texture.width);
	const size_t height = static_cast<size_t>(grid_texture.height);
	if (width == 0 || height == 0)
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

bool metal_self_test()
{
	if (!metal_is_available())
	{
		return false;
	}

	MetalBuffer buffer;
	if (!metal_buffer_alloc(sizeof(float) * 4, &buffer))
	{
		return false;
	}

	float values[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
	if (!metal_buffer_upload(&buffer, 0, values, sizeof(values)))
	{
		metal_buffer_free(&buffer);
		return false;
	}

	const float scale = 2.0f;
	MetalDispatchBuffer buffers[] = { { &buffer, 0 } };
	MetalDispatchBytes bytes[] = { { &scale, sizeof(scale) } };
	if (!metal_dispatch_1d("buffer_scale", 4, 4, nullptr, 0, buffers, 1, bytes, 1, nullptr))
	{
		metal_buffer_free(&buffer);
		return false;
	}

	float output[4] = {};
	if (!metal_buffer_download(&buffer, 0, output, sizeof(output)))
	{
		metal_buffer_free(&buffer);
		return false;
	}

	metal_buffer_free(&buffer);
	return output[0] == 2.0f && output[1] == 4.0f && output[2] == 6.0f && output[3] == 8.0f;
}

#endif
