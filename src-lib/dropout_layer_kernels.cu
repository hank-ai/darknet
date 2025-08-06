#include "darknet_internal.hpp"


//void fill_ongpu(int N, float ALPHA, float * X, int INCX);
//int64_t get_current_iteration(Darknet::Network net);


__global__ void dropblock_fast_kernel(float *rand, float prob, int w, int h, int spatial, int filters, int batch, int block_size, float *drop_blocks_scale, float *output)
{
	const int threads = BLOCK;
	const int id = threadIdx.x;
	const int f = blockIdx.x % filters;
	const int b = blockIdx.x / filters;

	__shared__ int prob_block;
	__shared__ int index_block;

	if (id == 0) {
		prob_block = 1.0 * 1000000;
		index_block = -1;
	}
	__syncthreads();

	int i;
	for (i = id; i < spatial; i += threads) {
		int index = b*spatial*f + f*spatial + i;

		if (rand[index] < prob) {
			//Chose with the lowest rand[i]
			int new_val = rand[index] * 1000000;
			rand[index] = 1;
			int old_val = atomicMin(&prob_block, new_val);
			if (new_val < old_val)
			{
				index_block = i;
			}
		}

	}
	__syncthreads();
	if (index_block == -1) return;


	int b_x = index_block % w;
	int b_y = index_block / w;

	if (b_x > (w - block_size)) b_x = b_x - (w - block_size);
	if (b_y > (h - block_size)) b_y = b_y - (h - block_size);

	b_x = max(0, min(b_x, w - block_size));
	b_y = max(0, min(b_y, h - block_size));

	int block_square_size = block_size * block_size;

	for (i = id; i < block_square_size; i += threads)
	{
		int i_x = i % block_size;
		int i_y = i / block_size;

		int x = b_x + i_x;
		int y = b_y + i_y;

		if (x >= 0 && x < w && y >= 0 && y < h) {
			int new_index = b*filters*spatial + f*spatial + y*w + x;

			output[new_index] = 0;
			rand[new_index] = 0;
		}
	}

	if (id == 0 && drop_blocks_scale)
	{
		atomicAdd(&drop_blocks_scale[b], block_square_size);
	}

}

__global__ void set_scales_dropblock_kernel(float *drop_blocks_scale, int block_size_w, int block_size_h, int outputs, int batch)
{
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= batch) return;

	const float prob = drop_blocks_scale[index] / (float)outputs;
	const float scale = 1.0f / (1.0f - prob);
	drop_blocks_scale[index] = scale;
}

__global__ void scale_dropblock_kernel(float *output, int size, int outputs, float *drop_blocks_scale)
{
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= size) return;

	const int b = index / outputs;
	output[index] *= drop_blocks_scale[b];
}


__global__ void backward_dropblock_kernel(float *pass, float *delta, int size)
{
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= size) return;

	if (pass[index] == 0) delta[index] = 0;
}


__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}


void forward_dropout_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (!state.train) return;
	int iteration_num = get_current_iteration(state.net); // (*state.net.seen) / (state.net.batch*state.net.subdivisions);
	//if (iteration_num < state.net.burn_in) return;

	// We gradually increase the block size and the probability of dropout - during the first half of the training
	float multiplier = 1.0;
	if(iteration_num < (state.net.max_batches*0.85))
		multiplier = (iteration_num / (float)(state.net.max_batches*0.85));

	// dropblock
	if (l.dropblock)
	{
		//l.probability = 1 / keep_prob
		//const int max_blocks_per_channel = 10;
		const float cur_prob = l.probability * multiplier;
//		const float cur_scale = 1.f / (1.f - cur_prob);

		int block_width = l.dropblock_size_abs *multiplier;
		int block_height = l.dropblock_size_abs *multiplier;

		if (l.dropblock_size_rel) {
			block_width = l.dropblock_size_rel * l.w * multiplier;
			block_height = l.dropblock_size_rel * l.h * multiplier;
		}

		std::clamp(block_width, 1, l.w);
		std::clamp(block_height, 1, l.h);

		const int block_size = std::min(block_width, block_height);
		const float block_prob = cur_prob / (block_size*block_size);
		assert(block_size <= l.w && block_size <= l.h);

		const int size = l.inputs*l.batch;
		cuda_random(l.rand_gpu, size);

		fill_ongpu(l.batch, 0, l.drop_blocks_scale_gpu, 1);

		//fill_ongpu(l.outputs * l.batch, 1, state.input, 1); // remove!!!

		int num_blocks = l.batch * l.c;
		dropblock_fast_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (l.rand_gpu, block_prob, l.w, l.h, l.w*l.h, l.c, l.batch, block_size, l.drop_blocks_scale_gpu, state.input);
		CHECK_CUDA(cudaPeekAtLastError());

		num_blocks = get_number_of_blocks(l.batch, BLOCK);
		set_scales_dropblock_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (l.drop_blocks_scale_gpu, block_size, block_size, l.outputs, l.batch);
		CHECK_CUDA(cudaPeekAtLastError());

		num_blocks = get_number_of_blocks(l.outputs * l.batch, BLOCK);
		scale_dropblock_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (state.input, l.outputs * l.batch, l.outputs, l.drop_blocks_scale_gpu);
		CHECK_CUDA(cudaPeekAtLastError());

	}
	// dropout
	else
	{
		int size = l.inputs*l.batch;
		cuda_random(l.rand_gpu, size);

		yoloswag420blazeit360noscope <<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>> (state.input, size, l.rand_gpu, l.probability, l.scale);
		CHECK_CUDA(cudaPeekAtLastError());
	}
}

void backward_dropout_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if(!state.delta) return;

	const int size = l.inputs*l.batch;

	// dropblock
	if (l.dropblock)
	{
		int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);
		float multiplier = 1.0;
		if (iteration_num < (state.net.max_batches*0.85))
			multiplier = (iteration_num / (float)(state.net.max_batches*0.85));

		int block_width = l.dropblock_size_abs * multiplier;
		int block_height = l.dropblock_size_abs * multiplier;

		if (l.dropblock_size_rel)
		{
			block_width = l.dropblock_size_rel * l.w * multiplier;
			block_height = l.dropblock_size_rel * l.h * multiplier;
		}

		std::clamp(block_width, 1, l.w);
		std::clamp(block_height, 1, l.h);

		int num_blocks = get_number_of_blocks(l.outputs * l.batch, BLOCK);
		backward_dropblock_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>>(l.rand_gpu, state.delta, l.outputs * l.batch);
		CHECK_CUDA(cudaPeekAtLastError());

		scale_dropblock_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (state.delta, l.outputs * l.batch, l.outputs, l.drop_blocks_scale_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	// dropout
	else
	{
		yoloswag420blazeit360noscope <<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>> (state.delta, size, l.rand_gpu, l.probability, l.scale);
		CHECK_CUDA(cudaPeekAtLastError());
	}
}
