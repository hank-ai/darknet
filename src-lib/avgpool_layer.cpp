#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


Darknet::Layer make_avgpool_layer(int batch, int w, int h, int c)
{
	TAT(TATPARMS);

	*cfg_and_state.output << "avg                          " << w << " x" << h << " x" << c << " ->   " << c << std::endl;

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::AVGPOOL;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.out_w = 1;
	l.out_h = 1;
	l.out_c = c;
	l.outputs = l.out_c;
	l.inputs = h*w*c;
	int output_size = l.outputs * batch;
	l.output = (float*)xcalloc(output_size, sizeof(float));
	l.delta = (float*)xcalloc(output_size, sizeof(float));
	l.forward = forward_avgpool_layer;
	l.backward = backward_avgpool_layer;
	#ifdef DARKNET_GPU
	l.forward_gpu = forward_avgpool_layer_gpu;
	l.backward_gpu = backward_avgpool_layer_gpu;
	l.output_gpu  = cuda_make_array(l.output, output_size);
	l.delta_gpu   = cuda_make_array(l.delta, output_size);
	#endif

	return l;
}

void resize_avgpool_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	l->w = w;
	l->h = h;
	l->inputs = h*w*l->c;
}

void forward_avgpool_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

#ifdef DARKNET_USE_MPS
	/// @brief MPS global avgpool fast path for inference; falls back to CPU if unsupported.
	if (not state.train)
	{
		const Darknet::Layer *prev = mps_prev_layer(state);
		bool defer_readback = mps_should_defer_readback(state);
		if (mps_global_avgpool_forward(l, prev, state.input, l.output, defer_readback, nullptr))
		{
			mps_coverage_record(l, true);
			return;
		}
		mps_coverage_record(l, false);
		mps_flush_deferred_output(prev);
	}
#endif

	for (int b = 0; b < l.batch; ++b)
	{
		for (int k = 0; k < l.c; ++k)
		{
			int out_index = k + b*l.c;
			l.output[out_index] = 0;
			for (int i = 0; i < l.h*l.w; ++i)
			{
				int in_index = i + l.h*l.w*(k + b*l.c);
				l.output[out_index] += state.input[in_index];
			}
			l.output[out_index] /= l.h*l.w;
		}
	}
}

void backward_avgpool_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	for (int b = 0; b < l.batch; ++b)
	{
		for (int k = 0; k < l.c; ++k)
		{
			int out_index = k + b*l.c;
			for (int i = 0; i < l.h*l.w; ++i)
			{
				int in_index = i + l.h*l.w*(k + b*l.c);
				state.delta[in_index] += l.delta[out_index] / (l.h*l.w);
			}
		}
	}
}
