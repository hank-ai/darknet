#include "darknet_internal.hpp"

namespace
{
	void free_and_clear(uint32_t* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void free_and_clear(float* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void free_and_clear(float** & array)
	{
		TAT(TATPARMS);

		if (array)
		{
			/** @todo Isn't this an array?  Should the array be freed?
			 *
			free_and_clear(*array);
			 */
			free(array);
			array = nullptr;
		}

		return;
	}

	void free_and_clear(int* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void free_and_clear(char* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void free_sublayer(layer* & l)
	{
		TAT(TATPARMS);

		if (l)
		{
			free_layer(*l);
			free(l);
			l = nullptr;
		}

		return;
	}

	#ifdef GPU
	void cuda_free_and_clear(float* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			cuda_free(ptr);
			ptr = nullptr;
		}

		return;
	}
	#endif
}


void free_layer(layer l)
{
	TAT(TATPARMS);

	free_layer_custom(l, 0);
}


void free_layer_custom(layer l, int keep_cudnn_desc)
{
	TAT(TATPARMS);

	if (l.share_layer != nullptr)
	{
		return;	// don't free shared layers
	}

	if (l.antialiasing)
	{
		free_sublayer(l.input_layer);
	}
	if (l.type == CONV_LSTM)
	{
		if (l.peephole)
		{
			free_sublayer(l.vf);
			free_sublayer(l.vi);
			free_sublayer(l.vo);
		}
		else
		{
			free(l.vf);
			free(l.vi);
			free(l.vo);
			l.vf = nullptr;
			l.vi = nullptr;
			l.vo = nullptr;
		}
		free_sublayer(l.wf);
		if (!l.bottleneck)
		{
			free_sublayer(l.wi);
			free_sublayer(l.wg);
			free_sublayer(l.wo);
		}
		free_sublayer(l.uf);
		free_sublayer(l.ui);
		free_sublayer(l.ug);
		free_sublayer(l.uo);
	}

	if (l.type == CRNN)
	{
		free_sublayer(l.input_layer);
		free_sublayer(l.self_layer);
		free_sublayer(l.output_layer);
		l.output		= nullptr;
		l.delta			= nullptr;
#ifdef GPU
		l.output_gpu	= nullptr;
		l.delta_gpu		= nullptr;
#endif // GPU
	}

	if (l.type == DROPOUT)
	{
		if (l.rand)						free_and_clear(l.rand);
#ifdef GPU
		if (l.rand_gpu)					cuda_free_and_clear(l.rand_gpu);
		if (l.drop_blocks_scale)		cuda_free_host(l.drop_blocks_scale);
		l.drop_blocks_scale = nullptr;
		if (l.drop_blocks_scale_gpu)	cuda_free_and_clear(l.drop_blocks_scale_gpu);
#endif
		return;
	}

	if (l.mask)							free_and_clear(l.mask);
	if (l.classes_multipliers)			free_and_clear(l.classes_multipliers);
	if (l.cweights)						free_and_clear(l.cweights);
	if (l.indexes)						free_and_clear(l.indexes);
	if (l.input_layers)					free_and_clear(l.input_layers);
	if (l.input_sizes)					free_and_clear(l.input_sizes);
	if (l.layers_output)				free_and_clear(l.layers_output);
	if (l.layers_delta)					free_and_clear(l.layers_delta);
	if (l.map)							free_and_clear(l.map);
	if (l.rand)							free_and_clear(l.rand);
	if (l.cost)							free_and_clear(l.cost);
	if (l.labels && !l.detection)		free_and_clear(l.labels);
	if (l.class_ids && !l.detection)	free_and_clear(l.class_ids);
	if (l.cos_sim)						free_and_clear(l.cos_sim);
	if (l.exp_cos_sim)					free_and_clear(l.exp_cos_sim);
	if (l.p_constrastive)				free_and_clear(l.p_constrastive);
	if (l.embedding_output)				free_and_clear(l.embedding_output);
	if (l.state)						free_and_clear(l.state);
	if (l.prev_state)					free_and_clear(l.prev_state);
	if (l.forgot_state)					free_and_clear(l.forgot_state);
	if (l.forgot_delta)					free_and_clear(l.forgot_delta);
	if (l.state_delta)					free_and_clear(l.state_delta);
	if (l.concat)						free_and_clear(l.concat);
	if (l.concat_delta)					free_and_clear(l.concat_delta);
	if (l.binary_weights)				free_and_clear(l.binary_weights);
	if (l.biases)						free_and_clear(l.biases);
	if (l.bias_updates)					free_and_clear(l.bias_updates);
	if (l.scales)						free_and_clear(l.scales);
	if (l.scale_updates)				free_and_clear(l.scale_updates);
	if (l.biases_ema)					free_and_clear(l.biases_ema);
	if (l.scales_ema)					free_and_clear(l.scales_ema);
	if (l.weights_ema)					free_and_clear(l.weights_ema);
	if (l.weights)						free_and_clear(l.weights);
	if (l.weight_updates)				free_and_clear(l.weight_updates);
	if (l.align_bit_weights)			free_and_clear(l.align_bit_weights);
	if (l.mean_arr)						free_and_clear(l.mean_arr);

#ifdef GPU
	if (l.delta && l.delta_pinned)
	{
		cudaFreeHost(l.delta);
		l.delta = nullptr;
	}

	if (l.output && l.output_pinned)
	{
		cudaFreeHost(l.output);
		l.output = nullptr;
	}

#endif  // GPU
	if (l.delta)						free_and_clear(l.delta);
	if (l.output)						free_and_clear(l.output);
	if (l.activation_input)				free_and_clear(l.activation_input);
	if (l.squared)						free_and_clear(l.squared);
	if (l.norms)						free_and_clear(l.norms);
	if (l.spatial_mean)					free_and_clear(l.spatial_mean);
	if (l.mean)							free_and_clear(l.mean);
	if (l.variance)						free_and_clear(l.variance);
	if (l.mean_delta)					free_and_clear(l.mean_delta);
	if (l.variance_delta)				free_and_clear(l.variance_delta);
	if (l.rolling_mean)					free_and_clear(l.rolling_mean);
	if (l.rolling_variance)				free_and_clear(l.rolling_variance);
	if (l.x)							free_and_clear(l.x);
	if (l.x_norm)						free_and_clear(l.x_norm);
	if (l.m)							free_and_clear(l.m);
	if (l.v)							free_and_clear(l.v);
	if (l.z_cpu)						free_and_clear(l.z_cpu);
	if (l.r_cpu)						free_and_clear(l.r_cpu);
	if (l.binary_input)					free_and_clear(l.binary_input);
	if (l.bin_re_packed_input)			free_and_clear(l.bin_re_packed_input);
	if (l.t_bit_input)					free_and_clear(l.t_bit_input);
	if (l.loss)							free_and_clear(l.loss);

	// CONV-LSTM
	if (l.f_cpu)						free_and_clear(l.f_cpu);
	if (l.i_cpu)						free_and_clear(l.i_cpu);
	if (l.g_cpu)						free_and_clear(l.g_cpu);
	if (l.o_cpu)						free_and_clear(l.o_cpu);
	if (l.c_cpu)						free_and_clear(l.c_cpu);
	if (l.h_cpu)						free_and_clear(l.h_cpu);
	if (l.temp_cpu)						free_and_clear(l.temp_cpu);
	if (l.temp2_cpu)					free_and_clear(l.temp2_cpu);
	if (l.temp3_cpu)					free_and_clear(l.temp3_cpu);
	if (l.dc_cpu)						free_and_clear(l.dc_cpu);
	if (l.dh_cpu)						free_and_clear(l.dh_cpu);
	if (l.prev_state_cpu)				free_and_clear(l.prev_state_cpu);
	if (l.prev_cell_cpu)				free_and_clear(l.prev_cell_cpu);
	if (l.stored_c_cpu)					free_and_clear(l.stored_c_cpu);
	if (l.stored_h_cpu)					free_and_clear(l.stored_h_cpu);
	if (l.cell_cpu)						free_and_clear(l.cell_cpu);

#ifdef GPU
	if (l.indexes_gpu)					cuda_free((float *)l.indexes_gpu);
	if (l.contrast_p_gpu)				cuda_free((float *)l.contrast_p_gpu);
	l.indexes_gpu = nullptr;
	l.contrast_p_gpu = nullptr;
	if (l.z_gpu)						cuda_free_and_clear(l.z_gpu);
	if (l.r_gpu)						cuda_free_and_clear(l.r_gpu);
	if (l.m_gpu)						cuda_free_and_clear(l.m_gpu);
	if (l.v_gpu)						cuda_free_and_clear(l.v_gpu);
	if (l.forgot_state_gpu)				cuda_free_and_clear(l.forgot_state_gpu);
	if (l.forgot_delta_gpu)				cuda_free_and_clear(l.forgot_delta_gpu);
	if (l.state_gpu)					cuda_free_and_clear(l.state_gpu);
	if (l.state_delta_gpu)				cuda_free_and_clear(l.state_delta_gpu);
	if (l.gate_gpu)						cuda_free_and_clear(l.gate_gpu);
	if (l.gate_delta_gpu)				cuda_free_and_clear(l.gate_delta_gpu);
	if (l.save_gpu)						cuda_free_and_clear(l.save_gpu);
	if (l.save_delta_gpu)				cuda_free_and_clear(l.save_delta_gpu);
	if (l.concat_gpu)					cuda_free_and_clear(l.concat_gpu);
	if (l.concat_delta_gpu)				cuda_free_and_clear(l.concat_delta_gpu);
	if (l.binary_input_gpu)				cuda_free_and_clear(l.binary_input_gpu);
	if (l.binary_weights_gpu)			cuda_free_and_clear(l.binary_weights_gpu);
	if (l.mean_gpu)						cuda_free_and_clear(l.mean_gpu);
	if (l.variance_gpu)					cuda_free_and_clear(l.variance_gpu);
	if (l.m_cbn_avg_gpu)				cuda_free_and_clear(l.m_cbn_avg_gpu);
	if (l.v_cbn_avg_gpu)				cuda_free_and_clear(l.v_cbn_avg_gpu);
	if (l.rolling_mean_gpu)				cuda_free_and_clear(l.rolling_mean_gpu);
	if (l.rolling_variance_gpu)			cuda_free_and_clear(l.rolling_variance_gpu);
	if (l.variance_delta_gpu)			cuda_free_and_clear(l.variance_delta_gpu);
	if (l.mean_delta_gpu)				cuda_free_and_clear(l.mean_delta_gpu);
	if (l.x_norm_gpu)					cuda_free_and_clear(l.x_norm_gpu);

	// assisted excitation
	if (l.gt_gpu)						cuda_free_and_clear(l.gt_gpu);
	if (l.a_avg_gpu)					cuda_free_and_clear(l.a_avg_gpu);

	if (l.align_bit_weights_gpu)		cuda_free((float *)l.align_bit_weights_gpu);
	l.align_bit_weights_gpu = nullptr;

	if (l.mean_arr_gpu)					cuda_free_and_clear(l.mean_arr_gpu);
	if (l.align_workspace_gpu)			cuda_free_and_clear(l.align_workspace_gpu);
	if (l.transposed_align_workspace_gpu) cuda_free_and_clear(l.transposed_align_workspace_gpu);

	if (l.weights_gpu)					cuda_free_and_clear(l.weights_gpu);
	if (l.weight_updates_gpu)			cuda_free_and_clear(l.weight_updates_gpu);
	if (l.weight_deform_gpu)			cuda_free_and_clear(l.weight_deform_gpu);
	if (l.weights_gpu16)				cuda_free_and_clear(l.weights_gpu16);
	if (l.weight_updates_gpu16)			cuda_free_and_clear(l.weight_updates_gpu16);
	if (l.biases_gpu)					cuda_free_and_clear(l.biases_gpu);
	if (l.bias_updates_gpu)				cuda_free_and_clear(l.bias_updates_gpu);
	if (l.scales_gpu)					cuda_free_and_clear(l.scales_gpu);
	if (l.scale_updates_gpu)			cuda_free_and_clear(l.scale_updates_gpu);
	if (l.input_antialiasing_gpu)		cuda_free_and_clear(l.input_antialiasing_gpu);
	if (l.optimized_memory < 2)
	{
		if (l.x_gpu)					cuda_free_and_clear(l.x_gpu);
		if (l.output_gpu)				cuda_free_and_clear(l.output_gpu);
		if (l.output_avg_gpu)			cuda_free_and_clear(l.output_avg_gpu);
		if (l.activation_input_gpu)		cuda_free_and_clear(l.activation_input_gpu);
	}

/// @todo 2023-06-26 For now I'd rather ignore this warning than to try and mess with this old code and risk breaking things.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

	if (l.delta_gpu && (l.optimized_memory < 1 || l.keep_delta_gpu && l.optimized_memory < 3)) cuda_free_and_clear(l.delta_gpu);

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

	if (l.cos_sim_gpu)					cuda_free_and_clear(l.cos_sim_gpu);
	if (l.rand_gpu)						cuda_free_and_clear(l.rand_gpu);
	if (l.squared_gpu)					cuda_free_and_clear(l.squared_gpu);
	if (l.norms_gpu)					cuda_free_and_clear(l.norms_gpu);
	if (l.input_sizes_gpu)				cuda_free((float*)l.input_sizes_gpu);
	if (l.layers_output_gpu)			cuda_free((float*)l.layers_output_gpu);
	if (l.layers_delta_gpu)				cuda_free((float*)l.layers_delta_gpu);
	l.input_sizes_gpu	= nullptr;
	l.layers_output_gpu	= nullptr;
	l.layers_delta_gpu	= nullptr;

	// CONV-LSTM
	if (l.f_gpu)						cuda_free_and_clear(l.f_gpu);
	if (l.i_gpu)						cuda_free_and_clear(l.i_gpu);
	if (l.g_gpu)						cuda_free_and_clear(l.g_gpu);
	if (l.o_gpu)						cuda_free_and_clear(l.o_gpu);
	if (l.c_gpu)						cuda_free_and_clear(l.c_gpu);
	if (l.h_gpu)						cuda_free_and_clear(l.h_gpu);
	if (l.bottelneck_hi_gpu)			cuda_free_and_clear(l.bottelneck_hi_gpu);
	if (l.bottelneck_delta_gpu)			cuda_free_and_clear(l.bottelneck_delta_gpu);
	if (l.temp_gpu)						cuda_free_and_clear(l.temp_gpu);
	if (l.temp2_gpu)					cuda_free_and_clear(l.temp2_gpu);
	if (l.temp3_gpu)					cuda_free_and_clear(l.temp3_gpu);
	if (l.dc_gpu)						cuda_free_and_clear(l.dc_gpu);
	if (l.dh_gpu)						cuda_free_and_clear(l.dh_gpu);
	if (l.prev_state_gpu)				cuda_free_and_clear(l.prev_state_gpu);
	if (l.prev_cell_gpu)				cuda_free_and_clear(l.prev_cell_gpu);
	if (l.stored_c_gpu)					cuda_free_and_clear(l.stored_c_gpu);
	if (l.stored_h_gpu)					cuda_free_and_clear(l.stored_h_gpu);
	if (l.last_prev_state_gpu)			cuda_free_and_clear(l.last_prev_state_gpu);
	if (l.last_prev_cell_gpu)			cuda_free_and_clear(l.last_prev_cell_gpu);
	if (l.cell_gpu)						cuda_free_and_clear(l.cell_gpu);

#ifdef CUDNN   // shouldn't be used for -map
	if (!keep_cudnn_desc)
	{
		if (l.srcTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc));
		if (l.dstTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc));
		if (l.srcTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc16));
		if (l.dstTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc16));
		if (l.dsrcTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc));
		if (l.ddstTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc));
		if (l.dsrcTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc16));
		if (l.ddstTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc16));
		if (l.normTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normTensorDesc));
		if (l.normDstTensorDesc)	CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDesc));
		if (l.normDstTensorDescF16)	CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDescF16));

		if (l.weightDesc)			CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc));
		if (l.weightDesc16)			CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc16));
		if (l.dweightDesc)			CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc));
		if (l.dweightDesc16)		CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc16));

		if (l.convDesc)				CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(l.convDesc));

		if (l.poolingDesc)			CHECK_CUDNN(cudnnDestroyPoolingDescriptor(l.poolingDesc));

		//cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
		//cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
		//cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
	}
#endif  // CUDNN

#endif  // GPU
}
