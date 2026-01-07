#include "darknet_internal.hpp"


Darknet::Layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
	TAT(TATPARMS);

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::UPSAMPLE;
	l.batch = batch;
	l.w = w;
	l.h = h;
	l.c = c;
	l.out_w = w*stride;
	l.out_h = h*stride;
	l.out_c = c;

	if (stride < 0)
	{
		// this is important -- if stride was negative, then we're downsampling instead of upsampling

		stride = -stride;
		l.reverse=1;
		l.out_w = w/stride;
		l.out_h = h/stride;
	}

	l.stride = stride;
	l.outputs = l.out_w*l.out_h*l.out_c;
	l.inputs = l.w*l.h*l.c;
	l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
	l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

	l.forward = forward_upsample_layer;
	l.backward = backward_upsample_layer;
	#ifdef DARKNET_GPU
	l.forward_gpu = forward_upsample_layer_gpu;
	l.backward_gpu = backward_upsample_layer_gpu;

	l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
	l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
	#endif

	return l;
}

void resize_upsample_layer(Darknet::Layer *l, int w, int h)
{
	TAT(TATPARMS);

	l->w = w;
	l->h = h;
	l->out_w = w*l->stride;
	l->out_h = h*l->stride;
	if(l->reverse){
		l->out_w = w/l->stride;
		l->out_h = h/l->stride;
	}
	l->outputs = l->out_w*l->out_h*l->out_c;
	l->inputs = l->h*l->w*l->c;
	l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
	l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef DARKNET_GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
	l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif

}

void forward_upsample_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

#ifdef DARKNET_USE_MPS
	if (not state.train and not l.reverse)
	{
		const Darknet::Layer *prev = mps_prev_layer(state);
		bool defer_readback = mps_should_defer_readback(state);
		if (mps_upsample_forward(l, prev, state.input, l.output, defer_readback, nullptr))
		{
			mps_coverage_record(l, true);
			return;
		}
		mps_coverage_record(l, false);
		mps_flush_deferred_output(prev);
	}
#endif

	fill_cpu(l.outputs*l.batch, 0, l.output, 1);
	if(l.reverse){
		upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, state.input);
	}else{
		upsample_cpu( state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
	}
}

void backward_upsample_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if(l.reverse){
		upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, state.delta);
	}else{
		upsample_cpu(state.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
	}
}

#ifdef DARKNET_GPU
void forward_upsample_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	if(l.reverse){
		upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, state.input);
	}else{
		upsample_gpu(state.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
	}
}

void backward_upsample_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if(l.reverse){
		upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, state.delta);
	}else{
		upsample_gpu(state.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu);
	}
}
#endif
