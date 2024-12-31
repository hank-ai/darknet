/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#pragma once

#include "darknet_internal.hpp"

#if DARKNET_GPU_ROCM


namespace Darknet
{
	/** Display some simple information on AMD ROCm and list the available AMD GPUs.  For example, this is part of the
	 * information which is shown to the user on the console when @p "darknet --version" is called.
	 * @see @ref Darknet::show_version_info()
	 */
	void show_rocm_info();
}

#endif // DARKNET_GPU_ROCM
