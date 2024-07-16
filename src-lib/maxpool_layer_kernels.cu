#include "darknet_internal.hpp"


__global__ void forward_maxpool_depth_layer_kernel(int n, int w, int h, int c, int out_c, int batch, float *input, float *output, int *indexes)
{
	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	int j = id % w;
	id = id / w;
	int i = id % h;
	id = id / h;
	//int g = id % out_c;
	//id = id / out_c;
	int b = id % batch;

	int k;
	for (int g = 0; g < out_c; ++g)
	{
		int out_index = j + w*(i + h*(g + out_c*b));
		float max = -FLT_MAX;
		int max_i = -1;

		for (k = g; k < c; k += out_c)
		{
			int in_index = j + w*(i + h*(k + c*b));
			float val = input[in_index];

			max_i = (val > max) ? in_index : max_i;
			max = (val > max) ? val : max;
		}
		output[out_index] = max;
		if (indexes) indexes[out_index] = max_i;
	}
}


__global__ void backward_maxpool_depth_layer_kernel(int n, int w, int h, int c, int batch, float *delta, float *prev_delta, int *indexes)
{
	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	int index = indexes[id];
	prev_delta[index] += delta[id];
}


__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *input, float *output, int *indexes)
{
	int h = (in_h + pad - size) / stride_y + 1;
	int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if(id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	int out_index = j + w*(i + h*(k + c*b));
	float max = -INFINITY;
	int max_i = -1;
	int l, m;
	for(l = 0; l < size; ++l){
		for(m = 0; m < size; ++m){
			int cur_h = h_offset + i*stride_y + l;
			int cur_w = w_offset + j*stride_x + m;
			int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
					cur_w >= 0 && cur_w < in_w);
			float val = (valid != 0) ? input[index] : -INFINITY;
			max_i = (val > max) ? index : max_i;
			max   = (val > max) ? val   : max;
		}
	}
	output[out_index] = max;
	if (indexes) indexes[out_index] = max_i;
}

__global__ void forward_zero_nonmax_kernel(int n, float *input, float *output)
{

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	if (input[id] != output[id]) output[id] = 0;
}

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *delta, float *prev_delta, int *indexes)
{
	int h = (in_h + pad - size) / stride_y + 1;
	int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;
	int area_x = (size - 1) / stride_x;
	int area_y = (size - 1) / stride_y;

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if(id >= n) return;

	int index = id;
	int j = id % in_w;
	id /= in_w;
	int i = id % in_h;
	id /= in_h;
	int k = id % in_c;
	id /= in_c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	float d = 0;
	int l, m;
	for(l = -area_y; l < area_y+1; ++l){
		for(m = -area_x; m < area_x+1; ++m){
			int out_w = (j-w_offset)/stride_x + m;
			int out_h = (i-h_offset)/stride_y + l;
			int out_index = out_w + w*(out_h + h*(k + c*b));
			int valid = (out_w >= 0 && out_w < w &&
					out_h >= 0 && out_h < h);
			d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
		}
	}
	prev_delta[index] += d;
}

__global__ void backward_zero_nonmax_kernel(int n, int *indexes, float *prev_delta)
{

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	if (indexes[id] != id) prev_delta[id] = 0;
}

extern "C" void forward_maxpool_layer_gpu(layer l, network_state state)
{
	TAT(TATPARMS);

	if (l.maxpool_depth)
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = 1;// layer.out_c;

		size_t n = h*w*c*l.batch;

		forward_maxpool_depth_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(n, l.w, l.h, l.c, l.out_c, l.batch, state.input, l.output_gpu, l.indexes_gpu);
		CHECK_CUDA(cudaPeekAtLastError());

		return;
	}

#ifdef CUDNN_DISABLED
	if (!state.train && l.stride == l.size)
	{
		// cudnnPoolingBackward
		cudnnStatus_t maxpool_status;

		float alpha = 1, beta = 0;
		maxpool_status = cudnnPoolingForward(
			cudnn_handle(),
			l.poolingDesc,
			&alpha,
			l.srcTensorDesc,
			state.input,
			&beta,
			l.dstTensorDesc,
			l.output_gpu);

		//maxpool_status = cudnnDestroyPoolingDescriptor(poolingDesc);
		//cudnnDestroyTensorDescriptor(l.srcTensorDesc);
		//cudnnDestroyTensorDescriptor(l.dstTensorDesc);
	}
	else
#endif
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = l.out_c;

		size_t n = h * w * c * l.batch;

		forward_maxpool_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (n, l.h, l.w, l.c, l.stride_x, l.stride_y, l.size, l.pad, state.input, l.output_gpu, l.indexes_gpu);
		CHECK_CUDA(cudaPeekAtLastError());

		if (l.maxpool_zero_nonmax)
		{
			forward_zero_nonmax_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (n, state.input, l.output_gpu);
			CHECK_CUDA(cudaPeekAtLastError());
		}
	}

	if (l.antialiasing)
	{
		network_state s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
		s.input = l.output_gpu;
		forward_convolutional_layer_gpu(*(l.input_layer), s);
		simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.input_antialiasing_gpu);
		simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.input_layer->output_gpu, l.output_gpu);
	}
}

extern "C" void backward_maxpool_layer_gpu(layer l, network_state state)
{
	TAT(TATPARMS);

	if (l.antialiasing)
	{
		network_state s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		s.delta = l.delta_gpu;  // s.delta will be returned to l.delta_gpu
		s.input = l.input_antialiasing_gpu;
		//if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
		simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.delta_gpu, l.input_layer->delta_gpu);
		backward_convolutional_layer_gpu(*(l.input_layer), s);

		//simple_copy_ongpu(l.outputs*l.batch, l.input_antialiasing_gpu, l.output_gpu);
	}

	if (l.maxpool_depth)
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = l.out_c;

		size_t n = h * w * c * l.batch;

		backward_maxpool_depth_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(n, l.w, l.h, l.c, l.batch, l.delta_gpu, state.delta, l.indexes_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
		return;
	}

	size_t n = l.h*l.w*l.c*l.batch;

	backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(n, l.h, l.w, l.c, l.stride_x, l.stride_y, l.size, l.pad, l.delta_gpu, state.delta, l.indexes_gpu);
	CHECK_CUDA(cudaPeekAtLastError());

	if (l.maxpool_zero_nonmax)
	{
		backward_zero_nonmax_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (n, l.indexes_gpu, state.delta);
		CHECK_CUDA(cudaPeekAtLastError());
	}
}




__global__ void forward_local_avgpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *input, float *output)
{
	int h = (in_h + pad - size) / stride_y + 1;
	int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	int out_index = j + w*(i + h*(k + c*b));
	float avg = 0;
	int counter = 0;
	int l, m;
	for (l = 0; l < size; ++l) {
		for (m = 0; m < size; ++m) {
			int cur_h = h_offset + i*stride_y + l;
			int cur_w = w_offset + j*stride_x + m;
			int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
				cur_w >= 0 && cur_w < in_w);
			if (valid) {
				counter++;
				avg += input[index];
			}
		}
	}
	output[out_index] = avg / counter;  // as CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
}


__global__ void backward_local_avgpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *delta, float *prev_delta)
{
	int h = (in_h + pad - size) / stride_y + 1;
	int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;
	int area_x = (size - 1) / stride_x;
	int area_y = (size - 1) / stride_y;

	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	int index = id;
	int j = id % in_w;
	id /= in_w;
	int i = id % in_h;
	id /= in_h;
	int k = id % in_c;
	id /= in_c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	int counter = 0;
	float d = 0;
	int l, m;
	for (l = -area_y; l < area_y + 1; ++l) {
		for (m = -area_x; m < area_x + 1; ++m) {
			int out_w = (j - w_offset) / stride_x + m;
			int out_h = (i - h_offset) / stride_y + l;
			int out_index = out_w + w*(out_h + h*(k + c*b));
			int valid = (out_w >= 0 && out_w < w && out_h >= 0 && out_h < h);
			if (valid) {
				counter++;
				d += delta[out_index];
			}
		}
	}
	if(counter > 0) prev_delta[index] += d / counter;
}



extern "C" void forward_local_avgpool_layer_gpu(layer l, network_state state)
{
	TAT(TATPARMS);

#ifdef CUDNN_DISABLED
	if (!state.train && l.stride == l.size)
	{
		// cudnnPoolingBackward
		cudnnStatus_t maxpool_status;

		float alpha = 1, beta = 0;
		maxpool_status = cudnnPoolingForward(
			cudnn_handle(),
			l.poolingDesc,
			&alpha,
			l.srcTensorDesc,
			state.input,
			&beta,
			l.dstTensorDesc,
			l.output_gpu);

		//maxpool_status = cudnnDestroyPoolingDescriptor(poolingDesc);
		//cudnnDestroyTensorDescriptor(l.srcTensorDesc);
		//cudnnDestroyTensorDescriptor(l.dstTensorDesc);
	}
	else
#endif
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = l.out_c;

		size_t n = h*w*c*l.batch;

		forward_local_avgpool_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> > (n, l.h, l.w, l.c, l.stride_x, l.stride_y, l.size, l.pad, state.input, l.output_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}
}

extern "C" void backward_local_avgpool_layer_gpu(layer l, network_state state)
{
	TAT(TATPARMS);

	size_t n = l.h * l.w * l.c * l.batch;

	backward_local_avgpool_layer_kernel << <cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >> >(n, l.h, l.w, l.c, l.stride_x, l.stride_y, l.size, l.pad, l.delta_gpu, state.delta);
	CHECK_CUDA(cudaPeekAtLastError());
}
