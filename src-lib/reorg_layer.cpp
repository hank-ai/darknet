#include "darknet_internal.hpp"
#include "apple_mps.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


Darknet::Layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse)
{
	TAT(TATPARMS);

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::REORG;
	l.batch = batch;
	l.stride = stride;
	l.h = h;
	l.w = w;
	l.c = c;
	if(reverse){
		l.out_w = w*stride;
		l.out_h = h*stride;
		l.out_c = c/(stride*stride);
	}else{
		l.out_w = w/stride;
		l.out_h = h/stride;
		l.out_c = c*(stride*stride);
	}
	l.reverse = reverse;

	*cfg_and_state.output
		<< "reorg                    /" << stride
		<< " " << w
		<< " x " << h
		<< " x " << c
		<< " -> " << l.out_w
		<< " x " << l.out_h
		<< " x " << l.out_c
		<< std::endl;

	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = h*w*c;
	int output_size = l.out_h * l.out_w * l.out_c * batch;
	l.output = (float*)xcalloc(output_size, sizeof(float));
	l.delta = (float*)xcalloc(output_size, sizeof(float));

	l.forward = forward_reorg_layer;
	l.backward = backward_reorg_layer;
#ifdef DARKNET_GPU
	l.forward_gpu = forward_reorg_layer_gpu;
	l.backward_gpu = backward_reorg_layer_gpu;

	l.output_gpu  = cuda_make_array(l.output, output_size);
	l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
	return l;
}

void resize_reorg_layer(Darknet::Layer *l, int w, int h)
{
	TAT(TATPARMS);

	int stride = l->stride;
	int c = l->c;

	l->h = h;
	l->w = w;

	if(l->reverse){
		l->out_w = w*stride;
		l->out_h = h*stride;
		l->out_c = c/(stride*stride);
	}else{
		l->out_w = w/stride;
		l->out_h = h/stride;
		l->out_c = c*(stride*stride);
	}

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->outputs;
	int output_size = l->outputs * l->batch;

	l->output = (float*)xrealloc(l->output, output_size * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, output_size * sizeof(float));

#ifdef DARKNET_GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu  = cuda_make_array(l->output, output_size);
	l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}

void forward_reorg_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

#ifdef DARKNET_USE_MPS
	if (!state.train && !l.reverse)
	{
		const Darknet::Layer *prev = mps_prev_layer(state);
		bool defer_readback = mps_should_defer_readback(state);
		if (mps_reorg_forward(l, prev, state.input, l.output, defer_readback, nullptr))
		{
			mps_coverage_record(l, true);
			return;
		}
		mps_coverage_record(l, false);
		mps_flush_deferred_output(prev);
	}
#endif

	if (l.reverse) {
		reorg_cpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, l.output);
	}
	else {
		reorg_cpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, l.output);
	}
}

void backward_reorg_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.reverse) {
		reorg_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, state.delta);
	}
	else {
		reorg_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, state.delta);
	}
}

#ifdef DARKNET_GPU
void forward_reorg_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.reverse) {
		reorg_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, l.output_gpu);
	}
	else {
		reorg_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, l.output_gpu);
	}
}

void backward_reorg_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.reverse) {
		reorg_ongpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, state.delta);
	}
	else {
		reorg_ongpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, state.delta);
	}
}
#endif
