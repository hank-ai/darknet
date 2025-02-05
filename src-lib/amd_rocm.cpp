#include "darknet_internal.hpp"

#if DARKNET_GPU_ROCM

#include <rocm-core/rocm_version.h>
#include <rocm_smi/rocm_smi.h>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


void Darknet::show_rocm_info()
{
	TAT(TATPARMS);

	*cfg_and_state.output << "AMD ROCm v" << ROCM_BUILD_INFO << std::endl;

	const auto status1 = rsmi_init(0);

	uint32_t number_of_devices = 0;
	const auto status2 = rsmi_num_monitor_devices(&number_of_devices);

	if (status1 != RSMI_STATUS_SUCCESS or status2 != RSMI_STATUS_SUCCESS)
	{
		const char * msg1 = nullptr;
		const char * msg2 = nullptr;
		rsmi_status_string(status1, &msg1);
		rsmi_status_string(status2, &msg2);

		*cfg_and_state.output
			<< "- status #" << status1 << ": " << (msg1 ? msg1 : "unknown") << std::endl
			<< "- status #" << status2 << ": " << (msg2 ? msg2 : "unknown") << std::endl;
	}

	if (number_of_devices == 0)
	{
		*cfg_and_state.output << Darknet::in_colour(Darknet::EColour::kBrightRed, "AMD GPU not detected!") << std::endl;
	}

	for (uint32_t device_idx = 0; device_idx < number_of_devices; device_idx ++)
	{
		char name[100];
		const size_t len = sizeof(name);
		rsmi_dev_name_get(device_idx, name, len);

		uint64_t memory = 0;
		rsmi_dev_memory_total_get(device_idx, rsmi_memory_type_t::RSMI_MEM_TYPE_VIS_VRAM, &memory);

		*cfg_and_state.output
			<< "=> " << device_idx
			<< ": " << Darknet::in_colour(Darknet::EColour::kBrightGreen, name)
			<< ", " << Darknet::in_colour(Darknet::EColour::kYellow, size_to_IEC_string(memory))
			<< std::endl;
	}

	rsmi_shut_down();
}

#endif // DARKNET_GPU_ROCM
