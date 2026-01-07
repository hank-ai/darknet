#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


/// @todo what is this?
#define SECRET_NUM -1234

void softmax_tree(float *input, int batch, int inputs, float temp, Darknet::Tree *hierarchy, float *output)
{
	TAT(TATPARMS);

	int b;
	for (b = 0; b < batch; ++b) {
		int i;
		int count = 0;
		for (i = 0; i < hierarchy->groups; ++i) {
			int group_size = hierarchy->group_size[i];
			softmax(input + b*inputs + count, group_size, temp, output + b*inputs + count, 1);
			count += group_size;
		}
	}
}

Darknet::Layer make_softmax_layer(int batch, int inputs, int groups)
{
	TAT(TATPARMS);

	assert(inputs%groups == 0);
	*cfg_and_state.output << "softmax                                        " << inputs << std::endl;
	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::SOFTMAX;
	l.batch = batch;
	l.groups = groups;
	l.inputs = inputs;
	l.outputs = inputs;
	l.loss = (float*)xcalloc(inputs * batch, sizeof(float));
	l.output = (float*)xcalloc(inputs * batch, sizeof(float));
	l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
	l.cost = (float*)xcalloc(1, sizeof(float));

	l.forward = forward_softmax_layer;
	l.backward = backward_softmax_layer;
#ifdef DARKNET_GPU
	l.forward_gpu = forward_softmax_layer_gpu;
	l.backward_gpu = backward_softmax_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, inputs*batch);
	l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
	return l;
}

void forward_softmax_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

#ifdef DARKNET_USE_MPS
	if (not state.train)
	{
		const Darknet::Layer *prev = mps_prev_layer(state);
		if (mps_softmax_forward(l, prev, state.input, l.output, nullptr))
		{
			mps_coverage_record(l, true);
			return;
		}
		mps_coverage_record(l, false);
		mps_flush_deferred_output(prev);
	}
#endif

	if(l.softmax_tree){
		int i;
		int count = 0;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
			int group_size = l.softmax_tree->group_size[i];
			softmax_cpu(state.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
			count += group_size;
		}
	} else {
		softmax_cpu(state.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
	}

	if(state.truth && !l.noloss){
		softmax_x_ent_cpu(l.batch*l.inputs, l.output, state.truth, l.delta, l.loss);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
}

void backward_softmax_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, state.delta, 1);
}

#ifdef DARKNET_GPU

void pull_softmax_layer_output(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_pull_array(l.output_gpu, l.output, l.inputs * l.batch);
}

void forward_softmax_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if(l.softmax_tree){
		softmax_tree_gpu( state.input, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
		/*
		int i;
		int count = 0;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
		int group_size = l.softmax_tree->group_size[i];
		softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
		count += group_size;
		}
		*/
	} else {
		if(l.spatial){
			softmax_gpu_new_api( state.input, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
		}else{
			softmax_gpu_new_api( state.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
		}
	}
	if( state.truth && !l.noloss){
		softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, state.truth, l.delta_gpu, l.loss_gpu);
		if(l.softmax_tree){
			mask_gpu_new_api(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, state.truth, 0);
			mask_gpu_new_api(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, state.truth, 0);
		}
		cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
}

void backward_softmax_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	axpy_ongpu(l.batch * l.inputs, state.net.loss_scale, l.delta_gpu, 1, state.delta, 1);
}

#endif

// -------------------------------------

// Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
Darknet::Layer make_contrastive_layer(int batch, int w, int h, int c, int classes, int inputs, Darknet::Layer *yolo_layer)
{
	TAT(TATPARMS);

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::CONTRASTIVE;
	l.batch = batch;
	l.inputs = inputs;
	l.w = w;
	l.h = h;
	l.c = c;
	l.temperature = 1;

	l.max_boxes = 0;
	if (yolo_layer)
	{
		l.detection = 1;
		l.max_boxes = yolo_layer->max_boxes;
		l.labels = yolo_layer->labels;  // track id
		l.class_ids = yolo_layer->class_ids;  // class_ids
		l.n = yolo_layer->n;            // num of embeddings per cell = num of anchors
		l.classes = yolo_layer->classes;// num of classes
		classes = l.classes;
		l.embedding_size = l.inputs / (l.n*l.h*l.w);
		l.truths = yolo_layer->truths;
		if (l.embedding_size != yolo_layer->embedding_size)
		{
			darknet_fatal_error(DARKNET_LOC, "mismatch between [contrastive] embedding_size=%d and [yolo] embedding_size=%d. They should use the same [convolutional] layer", l.embedding_size, yolo_layer->embedding_size);
		}
		if (l.inputs % (l.n*l.h*l.w) != 0)
		{
			darknet_fatal_error(DARKNET_LOC, "number of filters in the previous (embedding) layer isn't divisable by number of anchors (%d)", l.n);
		}
	}
	else
	{
		l.detection = 0;
		l.labels = (int*)xcalloc(l.batch, sizeof(int)); // labels
		l.n = 1;                                        // num of embeddings per cell
		l.classes = classes;                            // num of classes
		l.embedding_size = l.c;
	}
	l.outputs = inputs;

	l.loss = (float*)xcalloc(1, sizeof(float));
	l.output = (float*)xcalloc(inputs * batch, sizeof(float));
	l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
	l.cost = (float*)xcalloc(1, sizeof(float));

	const size_t step = l.batch*l.n*l.h*l.w;
	l.cos_sim = NULL;
	l.exp_cos_sim = NULL;
	l.p_constrastive = NULL;
	if (!l.detection)
	{
		l.cos_sim = (float*)xcalloc(step*step, sizeof(float));
		l.exp_cos_sim = (float*)xcalloc(step*step, sizeof(float));
		l.p_constrastive = (float*)xcalloc(step*step, sizeof(float));
	}
	//l.p_constrastive = (float*)xcalloc(step*step, sizeof(float));
	//l.contrast_p_size = (int*)xcalloc(1, sizeof(int));
	//*l.contrast_p_size = step;
	//l.contrast_p = (contrastive_params*)xcalloc(*l.contrast_p_size, sizeof(contrastive_params));

	l.forward = forward_contrastive_layer;
	l.backward = backward_contrastive_layer;
#ifdef DARKNET_GPU
	l.forward_gpu = forward_contrastive_layer_gpu;
	l.backward_gpu = backward_contrastive_layer_gpu;

	l.output_gpu = cuda_make_array(l.output, inputs*batch);
	l.delta_gpu = cuda_make_array(l.delta, inputs*batch);

	const int max_contr_size = (l.max_boxes*l.batch)*(l.max_boxes*l.batch) * sizeof(contrastive_params)/4;
	*cfg_and_state.output << "max_contr_size=" << Darknet::size_to_IEC_string(max_contr_size) << std::endl;
	l.contrast_p_gpu = (contrastive_params *)cuda_make_array(NULL, max_contr_size);
#endif

	*cfg_and_state.output
		<< "contrastive "	<< w
		<< " x "			<< h
		<< " x "			<< l.n
		<< " x emb_size "	<< l.embedding_size
		<< " x batch: "		<< batch
		<< " classes="		<< l.classes
		<< ", step="		<< step
		<< std::endl;

	if (l.detection)
	{
		*cfg_and_state.output << "detection" << std::endl;
	}

	return l;
}

static inline float clip_value(float val, const float max_val)
{
	TAT(TATPARMS);

	if (val > max_val)
	{
		val = max_val;
	}
	else if (val < -max_val)
	{
		val = -max_val;
	}
	return val;
}

void forward_contrastive_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (!state.train) return;
	const float truth_thresh = state.net.label_smooth_eps;

	const int mini_batch = l.batch / l.steps;

	fill_cpu(l.batch*l.inputs, 0, l.delta, 1);

	if (!l.detection)
	{

		for (int b = 0; b < l.batch; ++b)
		{
			if (state.net.adversarial)
			{
				l.labels[b] = b % 2;
			}
			else
			{
				l.labels[b] = b / 2;
			}
		}

		// set labels
		for (int b = 0; b < l.batch; ++b)
		{
			for (int h = 0; h < l.h; ++h)
			{
				for (int w = 0; w < l.w; ++w)
				{
					// find truth with max prob (only 1 label even if mosaic is used)
					//float max_truth = 0;
					for (int n = 0; n < l.classes; ++n)
					{
						const float truth_prob = state.truth[b*l.classes + n];
						if (truth_prob > truth_thresh)
						{
							l.labels[b] = n;
						}
					}
				}
			}
		}

	}

	// set pointers to features
	float **z = (float**)xcalloc(l.batch*l.n*l.h*l.w, sizeof(float*));

	for (int b = 0; b < l.batch; ++b)
	{
		for (int n = 0; n < l.n; ++n)
		{
			for (int h = 0; h < l.h; ++h)
			{
				for (int w = 0; w < l.w; ++w)
				{
					const int z_index = b*l.n*l.h*l.w + n*l.h*l.w + h*l.w + w;
					if (l.labels[z_index] < 0) continue;

					z[z_index] = (float*)xcalloc(l.embedding_size, sizeof(float));
					get_embedding(state.input, l.w, l.h, l.c, l.embedding_size, w, h, n, b, z[z_index]);
				}
			}
		}
	}

	int contrast_p_index = 0;

	const size_t step = l.batch*l.n*l.h*l.w;
	size_t contrast_p_size = step;
	if (!l.detection)
	{
		contrast_p_size = l.batch*l.batch;
	}
	contrastive_params *contrast_p = (contrastive_params*)xcalloc(contrast_p_size, sizeof(contrastive_params));

	float *max_sim_same = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
	float *max_sim_diff = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
	fill_cpu(l.batch*l.inputs, -10, max_sim_same, 1);
	fill_cpu(l.batch*l.inputs, -10, max_sim_diff, 1);

	// precalculate cosine similiraty
	for (int b = 0; b < l.batch; ++b)
	{
		for (int n = 0; n < l.n; ++n)
		{
			for (int h = 0; h < l.h; ++h)
			{
				for (int w = 0; w < l.w; ++w)
				{
					const int z_index = b*l.n*l.h*l.w + n*l.h*l.w + h*l.w + w;
					if (l.labels[z_index] < 0)
					{
						continue;
					}

					for (int b2 = 0; b2 < l.batch; ++b2)
					{
						for (int n2 = 0; n2 < l.n; ++n2)
						{
							for (int h2 = 0; h2 < l.h; ++h2)
							{
								for (int w2 = 0; w2 < l.w; ++w2)
								{
									const int z_index2 = b2*l.n*l.h*l.w + n2*l.h*l.w + h2*l.w + w2;
									if (l.labels[z_index2] < 0) continue;
									if (z_index == z_index2) continue;
									if (l.detection)
										if (l.class_ids[z_index] != l.class_ids[z_index2]) continue;

									const int time_step_i = b / mini_batch;
									const int time_step_j = b2 / mini_batch;
									if (time_step_i != time_step_j) continue;

//									const size_t step = l.batch*l.n*l.h*l.w;

									const float sim = cosine_similarity(z[z_index], z[z_index2], l.embedding_size);
									const float exp_sim = expf(sim / l.temperature);
									if (!l.detection)
									{
										l.cos_sim[z_index*step + z_index2] = sim;
										l.exp_cos_sim[z_index*step + z_index2] = exp_sim;
									}

									// calc good sim
									if (l.labels[z_index] == l.labels[z_index2] && max_sim_same[z_index] < sim) max_sim_same[z_index] = sim;
									if (l.labels[z_index] != l.labels[z_index2] && max_sim_diff[z_index] < sim) max_sim_diff[z_index] = sim;

									contrast_p[contrast_p_index].sim = sim;
									contrast_p[contrast_p_index].exp_sim = exp_sim;
									contrast_p[contrast_p_index].i = z_index;
									contrast_p[contrast_p_index].j = z_index2;
									contrast_p[contrast_p_index].time_step_i = time_step_i;
									contrast_p[contrast_p_index].time_step_j = time_step_j;
									contrast_p_index++;
									if ((contrast_p_index+1) >= contrast_p_size)
									{
										contrast_p_size = contrast_p_index + 1;
										contrast_p = (contrastive_params*)xrealloc(contrast_p, contrast_p_size * sizeof(contrastive_params));
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// calc contrastive accuracy
	int i;
	int good_sims = 0, all_sims = 0, same_sim = 0, diff_sim = 0;
	for (i = 0; i < l.batch*l.inputs; ++i) {
		if (max_sim_same[i] >= -1 && max_sim_diff[i] >= -1) {
			if (max_sim_same[i] >= -1) same_sim++;
			if (max_sim_diff[i] >= -1) diff_sim++;
			++all_sims;
			if (max_sim_diff[i] < max_sim_same[i]) good_sims++;
		}
	}
	if (all_sims > 0)
	{
		*l.loss = 100.0f * good_sims / all_sims;
	}
	else
	{
		*l.loss = -1.0f;
	}

	*cfg_and_state.output
		<< "Contrast"
		<< " accuracy=" << *l.loss << "%"
		<< ", all=" << all_sims
		<< ", good=" << good_sims
		<< ", same=" << same_sim
		<< ", diff=" << diff_sim
		<< std::endl;

	free(max_sim_same);
	free(max_sim_diff);

	const size_t contr_size = contrast_p_index;

	if (l.detection) {
#ifdef DARKNET_GPU
		const int max_contr_size = (l.max_boxes*l.batch)*(l.max_boxes*l.batch);
		if (max_contr_size < contr_size)
		{
			darknet_fatal_error(DARKNET_LOC, "too large number of bboxes: contr_size=%d > max_contr_size=%d", contr_size, max_contr_size);
		}
		int *labels = NULL;
		if (contr_size > 2) {
			cuda_push_array((float *)l.contrast_p_gpu, (float *)contrast_p, contr_size * sizeof(contrastive_params) / 4);
			P_constrastive_f_det_gpu(labels, l.embedding_size, l.temperature, l.contrast_p_gpu, contr_size);
			cuda_pull_array((float *)l.contrast_p_gpu, (float *)contrast_p, contr_size * sizeof(contrastive_params) / 4);
		}
#else   // DARKNET_GPU
		int k;
		//#pragma omp parallel for
		for (k = 0; k < contr_size; ++k) {
			contrast_p[k].P = P_constrastive_f_det(k, l.labels, z, l.embedding_size, l.temperature, contrast_p, contr_size);
		}
#endif  // DARKNET_GPU
	}
	else {
		// precalculate P-contrastive
		for (int b = 0; b < l.batch; ++b)
		{
			for (int n = 0; n < l.n; ++n)
			{
				for (int h = 0; h < l.h; ++h)
				{
					for (int w = 0; w < l.w; ++w)
					{
						const int z_index = b*l.n*l.h*l.w + n*l.h*l.w + h*l.w + w;
						if (l.labels[z_index] < 0)
						{
							continue;
						}

						for (int b2 = 0; b2 < l.batch; ++b2)
						{
							for (int n2 = 0; n2 < l.n; ++n2)
							{
								for (int h2 = 0; h2 < l.h; ++h2)
								{
									for (int w2 = 0; w2 < l.w; ++w2)
									{
										const int z_index2 = b2*l.n*l.h*l.w + n2*l.h*l.w + h2*l.w + w2;
										if (l.labels[z_index2] < 0)
										{
											continue;
										}
										if (z_index == z_index2)
										{
											continue;
										}
										if (l.detection)
										{
											if (l.class_ids[z_index] != l.class_ids[z_index2])
											{
												continue;
											}
										}

										const int time_step_i = b / mini_batch;
										const int time_step_j = b2 / mini_batch;
										if (time_step_i != time_step_j)
										{
											continue;
										}

										float P = -10;
										if (l.detection)
										{
											P = P_constrastive_f(z_index, z_index2, l.labels, z, l.embedding_size, l.temperature, contrast_p, contr_size);
										}
										else
										{
											P = P_constrastive(z_index, z_index2, l.labels, step, z, l.embedding_size, l.temperature, l.cos_sim, l.exp_cos_sim);
											l.p_constrastive[z_index*step + z_index2] = P;
										}

										for (int q = 0; q < contr_size; ++q)
										{
											if (contrast_p[q].i == z_index && contrast_p[q].j == z_index2)
											{
												contrast_p[q].P = P;
												break;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}


	// calc deltas
	int bd = 0;
	#pragma omp parallel for
	for (bd = 0; bd < l.batch; ++bd)
	{
		for (int nd = 0; nd < l.n; ++nd)
		{
			for (int hd = 0; hd < l.h; ++hd)
			{
				for (int wd = 0; wd < l.w; ++wd)
				{
					const int z_index = bd*l.n*l.h*l.w + nd*l.h*l.w + hd*l.w + wd;
					if (l.labels[z_index] < 0)
					{
						continue;
					}

					const int delta_index = bd*l.embedding_size*l.n*l.h*l.w + nd*l.embedding_size*l.h*l.w + hd*l.w + wd;
					const int wh = l.w*l.h;

					if (l.detection)
					{
						// detector

						// positive
						grad_contrastive_loss_positive_f(z_index, l.class_ids, l.labels, step, z, l.embedding_size, l.temperature, l.delta + delta_index, wh, contrast_p, contr_size);

						// negative
						grad_contrastive_loss_negative_f(z_index, l.class_ids, l.labels, step, z, l.embedding_size, l.temperature, l.delta + delta_index, wh, contrast_p, contr_size, l.contrastive_neg_max);
					}
					else
					{
						/// @todo does this still apply now that classifier code has been removed?
						// classifier

						// positive
						grad_contrastive_loss_positive(z_index, l.labels, step, z, l.embedding_size, l.temperature, l.cos_sim, l.p_constrastive, l.delta + delta_index, wh);

						// negative
						grad_contrastive_loss_negative(z_index, l.labels, step, z, l.embedding_size, l.temperature, l.cos_sim, l.p_constrastive, l.delta + delta_index, wh);
					}
				}
			}
		}
	}

	scal_cpu(l.inputs * l.batch, l.cls_normalizer, l.delta, 1);

	for (i = 0; i < l.inputs * l.batch; ++i)
	{
		l.delta[i] = clip_value(l.delta[i], l.max_delta);
	}

	*(l.cost) = pow(mag_array(l.delta, l.inputs * l.batch), 2);
	if (state.net.adversarial)
	{
		*cfg_and_state.output << "adversarial contrastive loss=" << *(l.cost) << std::endl << std::endl;
	}
	else
	{
		*cfg_and_state.output << "contrastive loss=" << *(l.cost) << std::endl << std::endl;
	}

	for (int b = 0; b < l.batch; ++b)
	{
		for (int n = 0; n < l.n; ++n)
		{
			for (int h = 0; h < l.h; ++h)
			{
				for (int w = 0; w < l.w; ++w)
				{
					const int z_index = b*l.n*l.h*l.w + n*l.h*l.w + h*l.w + w;
					//if (l.labels[z_index] < 0) continue;
					if (z[z_index])
					{
						free(z[z_index]);
					}
				}
			}
		}
	}

	free(contrast_p);
	free(z);
}

void backward_contrastive_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, state.delta, 1);
}


#ifdef DARKNET_GPU

void pull_contrastive_layer_output(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_pull_array(l.output_gpu, l.output, l.inputs*l.batch);
}

void push_contrastive_layer_output(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_push_array(l.delta_gpu, l.delta, l.inputs*l.batch);
}


void forward_contrastive_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
	if (!state.train) return;

	float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
	cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
	memcpy(in_cpu, l.output, l.batch*l.outputs * sizeof(float));
	float *truth_cpu = 0;
	if (state.truth) {
		int num_truth = l.batch*l.classes;
		if (l.detection) num_truth = l.batch*l.truths;
		truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);
	}
	Darknet::NetworkState cpu_state = state;
	cpu_state.truth = truth_cpu;
	cpu_state.input = in_cpu;

	forward_contrastive_layer(l, cpu_state);
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);

	free(in_cpu);
	if (cpu_state.truth) free(cpu_state.truth);
}

void backward_contrastive_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	axpy_ongpu(l.batch * l.inputs, state.net.loss_scale, l.delta_gpu, 1, state.delta, 1);
}

#endif
