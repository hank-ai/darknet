#include "apple_backend.hpp"
#ifdef DARKNET_USE_MPS

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>

namespace Darknet::Apple
{
	namespace
	{
		// One process-wide Metal device/queue backs MPS inference, MPSGraph
		// training, and custom kernels; queue order is the synchronization policy.
		struct RuntimeState
		{
			RuntimeInfo info;
			id<MTLDevice> device = nil;
			id<MTLCommandQueue> command_queue = nil;
			std::once_flag init_once;
		};

		RuntimeState & runtime_state()
		{
			static RuntimeState state;
			return state;
		}

		std::string string_or_unknown(NSString *value)
		{
			const char *cstr = value ? value.UTF8String : nullptr;
			return (cstr and *cstr) ? cstr : "unknown";
		}

		void initialize_runtime_state()
		{
			auto & state = runtime_state();
			state.info = {};

			state.device = MTLCreateSystemDefaultDevice();
			state.info.metal_available = state.device != nil;
			if (not state.device)
			{
				state.info.reason = "MTLCreateSystemDefaultDevice() returned nil";
				return;
			}

			state.info.device_name = string_or_unknown(state.device.name);
			state.info.recommended_max_working_set_size =
				static_cast<size_t>(state.device.recommendedMaxWorkingSetSize);

			if (not state.device.hasUnifiedMemory or
				not [state.device supportsFamily:MTLGPUFamilyApple7])
			{
				state.info.reason = "Apple GPU acceleration requires an Apple7-or-newer unified-memory GPU";
				return;
			}

			state.command_queue = [state.device newCommandQueue];
			state.info.metal_command_queue_available = state.command_queue != nil;
			state.info.apple_gpu_runtime_available = state.info.metal_available and state.info.metal_command_queue_available;
			state.info.reason = state.info.apple_gpu_runtime_available ? "ok" : "failed to create MTLCommandQueue";
		}
	}

	const RuntimeInfo & initialize_runtime()
	{
		auto & state = runtime_state();
		std::call_once(state.init_once, initialize_runtime_state);
		return state.info;
	}

	const RuntimeInfo & runtime_info()
	{
		return initialize_runtime();
	}

	void * runtime_metal_device_handle()
	{
		auto & state = runtime_state();
		(void)initialize_runtime();
		return (__bridge void *)state.device;
	}

	void * runtime_metal_command_queue_handle()
	{
		auto & state = runtime_state();
		(void)initialize_runtime();
		return (__bridge void *)state.command_queue;
	}
}

#else

namespace Darknet::Apple
{
	const RuntimeInfo & initialize_runtime()
	{
		static const RuntimeInfo info{};
		return info;
	}

	const RuntimeInfo & runtime_info()
	{
		return initialize_runtime();
	}

}

#endif
