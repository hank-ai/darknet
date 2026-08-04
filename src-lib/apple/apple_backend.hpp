#pragma once

/** @file
 * Internal Apple backend runtime capability probes.
 */

#include <string>

namespace Darknet
{
	struct Layer;
	struct Network;
}

namespace Darknet::Apple
{
	/** Process-wide Apple GPU capability snapshot. */
	struct RuntimeInfo
	{
		bool metal_available = false;					///< A Metal device was found.
		bool metal_command_queue_available = false;		///< A Metal command queue was created.
		bool apple_gpu_runtime_available = false;		///< All required Apple GPU runtime pieces are ready.
		size_t recommended_max_working_set_size = 0;		///< Device-reported memory working set size, in bytes.
		std::string device_name;							///< Human-readable Metal device name.
		std::string reason;								///< Short status or failure reason.
	};

	/** Concise Apple training capability summary used by logs and CLI output. */
	struct TrainingGraphInfo
	{
		bool runtime_ready = false;			///< The Apple GPU runtime is available.
		bool graph_supported = false;		///< The current network can use Apple graph training.
		std::string reason;				///< Short status or fallback reason.
	};

	/// Return process-wide Apple GPU runtime capability information.
	const RuntimeInfo & runtime_info();

	/// Initialize the Apple GPU runtime snapshot if needed and return it.
	const RuntimeInfo & initialize_runtime();

#ifdef DARKNET_USE_MPS
	/// Return the Objective-C Metal device handle owned by the runtime.
	void * runtime_metal_device_handle();

	/// Return the Objective-C Metal command queue handle owned by the runtime.
	void * runtime_metal_command_queue_handle();
#endif

}
