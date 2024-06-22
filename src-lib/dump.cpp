#include "dump.hpp"


std::string dump(uint64_t * ui)
{
	std::stringstream ss;

	if (ui == nullptr)
	{
		ss << (void*)ui;
	}
	else
	{
		ss << "[ui64=" << *ui << "]";
	}

	return ss.str();
}


std::string dump(float * f, size_t count=1)
{
	std::stringstream ss;

	if (f == nullptr)
	{
		ss << (void*)f;
	}
	else
	{
		ss << std::fixed << std::setprecision(6);

		for (auto idx = 0; idx < count; idx ++)
		{
			ss << (idx == 0 ? "[f=" : ", ") << f[idx];
		}
		ss << "]";
	}

	return ss.str();
}


std::string dump(int * i, int count=1)
{
	std::stringstream ss;

	if (i == nullptr)
	{
		ss << (void*)i;
	}
	else
	{
		for (auto idx = 0; idx < count; idx ++)
		{
			ss << (idx == 0 ? "[i=" : ", ") << i[idx];
		}
		ss << "]";
	}

	return ss.str();
}


void dump(network * net, const Darknet::CfgFile::CommonParms & parms)
{
	std::ofstream ofs("dump.txt");

	if (net == nullptr)
	{
		ofs << "net=nullptr" << std::endl;
		return;
	}

	ofs	<< std::fixed << std::setprecision(6)
		<< "---------------------------------"										<< std::endl
		<< "dump of network at "			<< net									<< std::endl
		<< "n="								<< net->n								<< std::endl
		<< "batch="							<< net->batch							<< std::endl
		<< "seen="							<< dump(net->seen)						<< std::endl
		<< "badlabels_reject_threshold="	<< dump(net->badlabels_reject_threshold)<< std::endl
		<< "delta_rolling_max="				<< dump(net->delta_rolling_max)			<< std::endl
		<< "delta_rolling_avg="				<< dump(net->delta_rolling_avg)			<< std::endl
		<< "delta_rolling_std="				<< dump(net->delta_rolling_std)			<< std::endl
		<< "weights_reject_freq="			<< net->weights_reject_freq				<< std::endl
		<< "equidistant_point="				<< net->equidistant_point				<< std::endl
		<< "badlabels_rejection_percentage="<< net->badlabels_rejection_percentage	<< std::endl
		<< "num_sigmas_reject_badlabels="	<< net->num_sigmas_reject_badlabels		<< std::endl
		<< "ema_alpha="						<< net->num_sigmas_reject_badlabels		<< std::endl
		<< "cur_iteration="					<< dump(net->cur_iteration)				<< std::endl
		<< "loss_scale="					<< net->loss_scale						<< std::endl
		<< "t="								<< dump(net->t)							<< std::endl
		<< "epoch="							<< net->epoch							<< std::endl
		<< "subdivisions="					<< net->subdivisions					<< std::endl
		<< "layers="						<< net->layers							<< std::endl
		<< "output="						<< dump(net->output)					<< std::endl
		<< "policy="						<< Darknet::to_string(static_cast<Darknet::ELearningRatePolicy>(net->policy)) << std::endl
		<< "benchmark_layers="				<< net->benchmark_layers				<< std::endl
		<< "total_bbox="					<< dump(net->total_bbox)				<< std::endl
		<< "rewritten_bbox="				<< dump(net->rewritten_bbox)			<< std::endl
		<< "learning_rate="					<< net->learning_rate					<< std::endl
		<< "learning_rate_min="				<< net->learning_rate_min				<< std::endl
		<< "learning_rate_max="				<< net->learning_rate_max				<< std::endl
		<< "batches_per_cycle="				<< net->batches_per_cycle				<< std::endl
		<< "batches_cycle_mult="			<< net->batches_cycle_mult				<< std::endl
		<< "momentum="						<< net->momentum						<< std::endl
		<< "decay="							<< net->decay							<< std::endl
		<< "gamma="							<< net->gamma							<< std::endl
		<< "scale="							<< net->scale							<< std::endl
		<< "power="							<< net->power							<< std::endl
		<< "time_steps="					<< net->time_steps						<< std::endl
		<< "step="							<< net->step							<< std::endl
		<< "max_batches="					<< net->max_batches						<< std::endl
		<< "num_boxes="						<< net->num_boxes						<< std::endl
		<< "train_images_num="				<< net->train_images_num				<< std::endl
		<< "seq_scales="					<< dump(net->seq_scales, net->num_steps)<< std::endl
		<< "scales="						<< dump(net->scales, net->num_steps)	<< std::endl
		<< "steps="							<< dump(net->steps, net->num_steps)		<< std::endl
		<< "num_steps="						<< net->num_steps						<< std::endl
		<< "burn_in="						<< net->burn_in							<< std::endl
		<< "cudnn_half="					<< net->cudnn_half						<< std::endl
		<< "adam="							<< net->adam							<< std::endl
		<< "B1="							<< net->B1								<< std::endl
		<< "B2="							<< net->B2								<< std::endl
		<< "eps="							<< net->eps								<< std::endl
		<< "inputs="						<< net->inputs							<< std::endl
		<< "outputs="						<< net->outputs							<< std::endl
		<< "truths="						<< net->truths							<< std::endl
		<< "notruth="						<< net->notruth							<< std::endl
		<< "h="								<< net->h								<< std::endl
		<< "w="								<< net->w								<< std::endl
		<< "c="								<< net->c								<< std::endl
		<< "max_crop="						<< net->max_crop						<< std::endl
		<< "min_crop="						<< net->min_crop						<< std::endl
		<< "max_ratio="						<< net->max_ratio						<< std::endl
		<< "min_ratio="						<< net->min_ratio						<< std::endl
		<< "center="						<< net->center							<< std::endl
		<< "flip="							<< net->flip							<< std::endl
		<< "gaussian_noise="				<< net->gaussian_noise					<< std::endl
		<< "blur="							<< net->blur							<< std::endl
		<< "mixup="							<< net->mixup							<< std::endl
		<< "label_smooth_eps="				<< net->label_smooth_eps				<< std::endl
		<< "resize_step="					<< net->resize_step						<< std::endl
		<< "attention="						<< net->attention						<< std::endl
		<< "adversarial="					<< net->adversarial						<< std::endl
		<< "adversarial_lr="				<< net->adversarial_lr					<< std::endl
		<< "max_chart_loss="				<< net->max_chart_loss					<< std::endl
		<< "letter_box="					<< net->letter_box						<< std::endl
		<< "mosaic_bound="					<< net->mosaic_bound					<< std::endl
		<< "contrastive="					<< net->contrastive						<< std::endl
		<< "contrastive_jit_flip="			<< net->contrastive_jit_flip			<< std::endl
		<< "contrastive_color="				<< net->contrastive_color				<< std::endl
		<< "unsupervised="					<< net->unsupervised					<< std::endl
		<< "angle="							<< net->angle							<< std::endl
		<< "aspect="						<< net->aspect							<< std::endl
		<< "exposure="						<< net->exposure						<< std::endl
		<< "saturation="					<< net->saturation						<< std::endl
		<< "hue="							<< net->hue								<< std::endl
		<< "random="						<< net->random							<< std::endl
		<< "track="							<< net->track							<< std::endl
		<< "augment_speed="					<< net->augment_speed					<< std::endl
		<< "sequential_subdivisions="		<< net->sequential_subdivisions			<< std::endl
		<< "init_sequential_subdivisions="	<< net->init_sequential_subdivisions	<< std::endl
		<< "current_subdivision="			<< net->current_subdivision				<< std::endl
		<< "try_fix_nan="					<< net->try_fix_nan						<< std::endl
		<< "gpu_index="						<< net->gpu_index						<< std::endl

/// @todo dump(tree) not implemented
//		tree *hierarchy;

		<< "input="							<< dump(net->input)						<< std::endl
		<< "truth="							<< dump(net->truth)						<< std::endl
		<< "delta="							<< dump(net->delta)						<< std::endl
//		<< "workspace="						<< dump(net->workspace)					<< std::endl -- causes segfaults
		<< "train="							<< net->train							<< std::endl
		<< "index="							<< net->index							<< std::endl
		<< "cost="							<< dump(net->cost)						<< std::endl
		<< "clip="							<< net->clip							<< std::endl
		<< "delta_gpu="						<< dump(net->delta_gpu)					<< std::endl
		<< "output_gpu="					<< dump(net->output_gpu)				<< std::endl
//		<< "input_state_gpu="				<< dump(net->input_state_gpu)			<< std::endl -- causes segfaults
		<< "input_pinned_cpu="				<< dump(net->input_pinned_cpu)			<< std::endl
		<< "input_pinned_cpu_flag="			<< net->input_pinned_cpu_flag			<< std::endl
		<< "input_gpu="						<< net->input_gpu						<< std::endl
		<< "truth_gpu="						<< net->truth_gpu						<< std::endl
		<< "input16_gpu="					<< net->input16_gpu						<< std::endl
		<< "output16_gpu="					<< net->output16_gpu					<< std::endl
		<< "max_input16_size="				<< dump(net->max_input16_size)			<< std::endl
		<< "max_output16_size="				<< dump(net->max_output16_size)			<< std::endl
		<< "wait_stream="					<< net->wait_stream						<< std::endl
		<< "cuda_graph="					<< net->cuda_graph						<< std::endl
		<< "cuda_graph_exec="				<< net->cuda_graph_exec					<< std::endl
		<< "use_cuda_graph="				<< net->use_cuda_graph					<< std::endl
		<< "cuda_graph_ready="				<< net->cuda_graph_ready				<< std::endl
		<< "global_delta_gpu="				<< dump(net->global_delta_gpu)			<< std::endl
		<< "state_delta_gpu="				<< dump(net->state_delta_gpu)			<< std::endl
		<< "max_delta_gpu_size="			<< net->max_delta_gpu_size				<< std::endl
		<< "optimized_memory="				<< net->optimized_memory				<< std::endl
		<< "dynamic_minibatch="				<< net->dynamic_minibatch				<< std::endl
		<< "workspace_size_limit="			<< net->workspace_size_limit			<< std::endl
		;

	for (size_t n = 0; n < net->n; n ++)
	{
		layer & l = net->layers[n];
		ofs	<< "--------------------------"							<< std::endl
			<< "n="						<< n << " of " << net->n	<< std::endl
			<< "type="					<< get_layer_string(l.type)	<< std::endl
			<< "activation="			<< Darknet::to_string(static_cast<Darknet::EActivation>(l.activation))		<< std::endl
			<< "lstm_activation="		<< Darknet::to_string(static_cast<Darknet::EActivation>(l.lstm_activation))	<< std::endl
			<< "cost_type="				<< l.cost_type				<< std::endl
#if 0
			<< "forward="				<< (void*)l.forward			<< std::endl
			<< "backward="				<< (void*)l.backward		<< std::endl
			<< "update="				<< (void*)l.update			<< std::endl
			<< "forward_gpu="			<< (void*)l.forward_gpu		<< std::endl
			<< "backward_gpu="			<< (void*)l.backward_gpu	<< std::endl
			<< "update_gpu="			<< (void*)l.update_gpu		<< std::endl
			<< "share_layer="			<< (void*)l.share_layer		<< std::endl
#endif
			<< "train="					<< l.train					<< std::endl
			<< "avgpool="				<< l.avgpool				<< std::endl
			<< "batch_normalize="		<< l.batch_normalize		<< std::endl
			<< "shortcut="				<< l.shortcut				<< std::endl
			<< "batch="					<< l.batch					<< std::endl
			<< "dynamic_minibatch="		<< l.dynamic_minibatch		<< std::endl
			<< "forced="				<< l.forced					<< std::endl
			<< "flipped="				<< l.flipped				<< std::endl
			<< "inputs="				<< l.inputs					<< std::endl
			<< "outputs="				<< l.outputs				<< std::endl
			<< "mean_alpha="			<< l.mean_alpha				<< std::endl
			<< "nweights="				<< l.nweights				<< std::endl
			<< "nbiases="				<< l.nbiases				<< std::endl
			<< "extra="					<< l.extra					<< std::endl
			<< "truths="				<< l.truths					<< std::endl
			<< "h="						<< l.h						<< std::endl
			<< "w="						<< l.w						<< std::endl
			<< "c="						<< l.c						<< std::endl
			<< "out_h="					<< l.out_h					<< std::endl
			<< "out_w="					<< l.out_w					<< std::endl
			<< "out_c="					<< l.out_c					<< std::endl
			<< "n="						<< l.n						<< std::endl
			<< "max_boxes="				<< l.max_boxes				<< std::endl
			<< "truth_size="			<< l.truth_size				<< std::endl
			<< "groups="				<< l.groups					<< std::endl
			<< "group_id="				<< l.group_id				<< std::endl
			<< "size="					<< l.size					<< std::endl
			<< "side="					<< l.side					<< std::endl
			<< "stride="				<< l.stride					<< std::endl
			<< "stride_x="				<< l.stride_x				<< std::endl
			<< "stride_y="				<< l.stride_y				<< std::endl
			<< "dilation="				<< l.dilation				<< std::endl
			<< "antialiasing="			<< l.antialiasing			<< std::endl
			<< "maxpool_depth="			<< l.maxpool_depth			<< std::endl
			<< "maxpool_zero_nonmax="	<< l.maxpool_zero_nonmax	<< std::endl
			<< "out_channels="			<< l.out_channels			<< std::endl
			<< "reverse="				<< l.reverse				<< std::endl
			<< "coordconv="				<< l.coordconv				<< std::endl
			<< "flatten="				<< l.flatten				<< std::endl
			<< "spatial="				<< l.spatial				<< std::endl
			<< "pad="					<< l.pad					<< std::endl
			<< "sqrt="					<< l.sqrt					<< std::endl
			<< "flip="					<< l.flip					<< std::endl
			<< "index="					<< l.index					<< std::endl
			<< "scale_wh="				<< l.scale_wh				<< std::endl
			<< "binary="				<< l.binary					<< std::endl
			<< "xnor="					<< l.xnor					<< std::endl
			<< "peephole="				<< l.peephole				<< std::endl
			<< "use_bin_output="		<< l.use_bin_output			<< std::endl
			<< "keep_delta_gpu="		<< l.keep_delta_gpu			<< std::endl
			<< "optimized_memory="		<< l.optimized_memory		<< std::endl
			<< "steps="					<< l.steps					<< std::endl
			<< "history_size="			<< l.history_size			<< std::endl
			<< "bottleneck="			<< l.bottleneck				<< std::endl
			<< "time_normalizer="		<< l.time_normalizer		<< std::endl
			<< "state_constrain="		<< l.state_constrain		<< std::endl
			<< "hidden="				<< l.hidden					<< std::endl
			<< "truth="					<< l.truth					<< std::endl
			<< "smooth="				<< l.smooth					<< std::endl
			<< "dot="					<< l.dot					<< std::endl
			<< "deform="				<< l.deform					<< std::endl
			<< "grad_centr="			<< l.grad_centr				<< std::endl
			<< "sway="					<< l.sway					<< std::endl
			<< "rotate="				<< l.rotate					<< std::endl
			<< "stretch="				<< l.stretch				<< std::endl
			<< "stretch_sway="			<< l.stretch_sway			<< std::endl
			<< "angle="					<< l.angle					<< std::endl
			<< "jitter="				<< l.jitter					<< std::endl
			<< "resize="				<< l.resize					<< std::endl
			<< "saturation="			<< l.saturation				<< std::endl
			<< "exposure="				<< l.exposure				<< std::endl
			<< "shift="					<< l.shift					<< std::endl
			<< "ratio="					<< l.ratio					<< std::endl
			<< "learning_rate_scale="	<< l.learning_rate_scale	<< std::endl
			<< "clip="					<< l.clip					<< std::endl
			<< "focal_loss="			<< l.focal_loss				<< std::endl
			<< "classes_multipliers="	<< dump(l.classes_multipliers) << std::endl
			<< "label_smooth_eps="		<< l.label_smooth_eps		<< std::endl
			<< "noloss="				<< l.noloss					<< std::endl
			<< "softmax="				<< l.softmax				<< std::endl
			<< "classes="				<< l.classes				<< std::endl
			<< "detection="				<< l.detection				<< std::endl
			<< "embedding_layer_id="	<< l.embedding_layer_id		<< std::endl
			<< "embedding_output="		<< dump(l.embedding_output)	<< std::endl
			<< "embedding_size="		<< l.embedding_size			<< std::endl
			<< "sim_thresh="			<< l.sim_thresh				<< std::endl
			<< "track_history_size="	<< l.track_history_size		<< std::endl
			<< "dets_for_track="		<< l.dets_for_track			<< std::endl
			<< "dets_for_show="			<< l.dets_for_show			<< std::endl
			<< "track_ciou_norm="		<< l.track_ciou_norm		<< std::endl
			<< "coords="				<< l.coords					<< std::endl
			<< "background="			<< l.background				<< std::endl
			<< "rescore="				<< l.rescore				<< std::endl
			<< "objectness="			<< l.objectness				<< std::endl
			<< "does_cost="				<< l.does_cost				<< std::endl
			<< "joint="					<< l.joint					<< std::endl
			<< "noadjust="				<< l.noadjust				<< std::endl
			<< "reorg="					<< l.reorg					<< std::endl
			<< "log="					<< l.log					<< std::endl
			<< "tanh="					<< l.tanh					<< std::endl
			<< "mask="					<< dump(l.mask, l.n)		<< std::endl
			<< "total="					<< l.total					<< std::endl
			<< "bflops="				<< l.bflops					<< std::endl
			<< "adam="					<< l.adam					<< std::endl
			<< "B1="					<< l.B1						<< std::endl
			<< "B2="					<< l.B2						<< std::endl
			<< "eps="					<< l.eps					<< std::endl
			<< "t="						<< l.t						<< std::endl
			<< "alpha="					<< l.alpha					<< std::endl
			<< "beta="					<< l.beta					<< std::endl
			<< "kappa="					<< l.kappa					<< std::endl
			<< "coord_scale="			<< l.coord_scale			<< std::endl
			<< "object_scale="			<< l.object_scale			<< std::endl
			<< "noobject_scale="		<< l.noobject_scale			<< std::endl
			<< "mask_scale="			<< l.mask_scale				<< std::endl
			<< "class_scale="			<< l.class_scale			<< std::endl
			<< "bias_match="			<< l.bias_match				<< std::endl
			<< "random="				<< l.random					<< std::endl
			<< "ignore_thresh="			<< l.ignore_thresh			<< std::endl
			<< "truth_thresh="			<< l.truth_thresh			<< std::endl
			<< "iou_thresh="			<< l.iou_thresh				<< std::endl
			<< "thresh="				<< l.thresh					<< std::endl
			<< "focus="					<< l.focus					<< std::endl
			<< "classfix="				<< l.classfix				<< std::endl
			<< "absolute="				<< l.absolute				<< std::endl
			<< "assisted_excitation="	<< l.assisted_excitation	<< std::endl
			<< "onlyforward="			<< l.onlyforward			<< std::endl
			<< "stopbackward="			<< l.stopbackward			<< std::endl
			<< "train_only_bn="			<< l.train_only_bn			<< std::endl
			<< "dont_update="			<< l.dont_update			<< std::endl
			<< "burnin_update="			<< l.burnin_update			<< std::endl
			<< "dontload="				<< l.dontload				<< std::endl
			<< "dontsave="				<< l.dontsave				<< std::endl
			<< "dontloadscales="		<< l.dontloadscales			<< std::endl
			<< "numload="				<< l.numload				<< std::endl
			<< "temperature="			<< l.temperature			<< std::endl
			<< "probability="			<< l.probability			<< std::endl
			<< "dropblock_size_rel="	<< l.dropblock_size_rel		<< std::endl
			<< "dropblock_size_abs="	<< l.dropblock_size_abs		<< std::endl
			<< "dropblock="				<< l.dropblock				<< std::endl
			<< "scale="					<< l.scale					<< std::endl
			<< "receptive_w="			<< l.receptive_w			<< std::endl
			<< "receptive_h="			<< l.receptive_h			<< std::endl
			<< "receptive_w_scale="		<< l.receptive_w_scale		<< std::endl
			<< "receptive_h_scale="		<< l.receptive_h_scale		<< std::endl
//			<< "cweights="				<< l.cweights				<< std::endl
			<< "indexes="				<< dump(l.indexes)			<< std::endl
			<< "input_layers="			<< dump(l.input_layers)		<< std::endl
			<< "input_sizes="			<< dump(l.input_sizes)		<< std::endl
			<< "layers_output="			<< (void*)l.layers_output	<< std::endl
			<< "layers_delta="			<< (void*)l.layers_delta	<< std::endl
			<< "weights_type="			<< l.weights_type			<< std::endl
			<< "weights_normalization="	<< l.weights_normalization	<< std::endl
#if 0
			int   * map;
			int   * counts;
			float ** sums;
			float * rand;
			float * cost;
			int *labels;
			int *class_ids;
			int contrastive_neg_max;
			float *cos_sim;
			float *exp_cos_sim;
			float *p_constrastive;
			contrastive_params *contrast_p_gpu;
			float * state;
			float * prev_state;
			float * forgot_state;
			float * forgot_delta;
			float * state_delta;
			float * combine_cpu;
			float * combine_delta_cpu;
			float *concat;
			float *concat_delta;
			float *binary_weights;
#endif
			<< "biases="				<< dump(l.biases, l.nbiases)	<< std::endl
			<< "bias_updates="			<< dump(l.bias_updates)			<< std::endl
			<< "scales="				<< dump(l.scales)				<< std::endl
			<< "scale_updates="			<< dump(l.scale_updates)		<< std::endl
			<< "weights_ema="			<< dump(l.weights_ema)			<< std::endl
			<< "biases_ema="			<< dump(l.biases_ema)			<< std::endl
			<< "scales_ema="			<< dump(l.scales_ema)			<< std::endl
//			<< "weights="				<< dump(l.weights)				<< std::endl // prior to loading weights this will be garbage
			<< "weight_updates="		<< dump(l.weight_updates)		<< std::endl
			<< "scale_x_y="				<< l.scale_x_y					<< std::endl
			<< "objectness_smooth="		<< l.objectness_smooth			<< std::endl
			<< "new_coords="			<< l.new_coords					<< std::endl
			<< "show_details="			<< l.show_details					<< std::endl
			<< "max_delta="				<< l.max_delta					<< std::endl
			<< "uc_normalizer="			<< l.uc_normalizer				<< std::endl
			<< "iou_normalizer="		<< l.iou_normalizer				<< std::endl
			<< "obj_normalizer="		<< l.obj_normalizer				<< std::endl
			<< "cls_normalizer="		<< l.cls_normalizer				<< std::endl
			<< "delta_normalizer="		<< l.delta_normalizer				<< std::endl
			<< "iou_loss="				<< Darknet::to_string(static_cast<Darknet::EIoULoss>(l.iou_loss)) << std::endl
			<< "iou_thresh_kind="		<< Darknet::to_string(static_cast<Darknet::EIoULoss>(l.iou_thresh_kind)) << std::endl
			<< "nms_kind="				<< Darknet::to_string(static_cast<Darknet::ENMSKind>(l.nms_kind)) << std::endl
			<< "beta_nms="				<< l.beta_nms						<< std::endl
			<< "yolo_point="			<< l.yolo_point					<< std::endl
#if 0
			char *align_bit_weights_gpu;
			float *mean_arr_gpu;
			float *align_workspace_gpu;
			float *transposed_align_workspace_gpu;
			int align_workspace_size;
			char *align_bit_weights;
			float *mean_arr;
			int align_bit_weights_size;
			int lda_align;
			int new_lda;
			int bit_align;
			float *col_image;
			float * delta;
			float * output;
			float * activation_input;
			int delta_pinned;
			int output_pinned;
			float * loss;
			float * squared;
			float * norms;
			float * spatial_mean;
			float * mean;
			float * variance;
			float * mean_delta;
			float * variance_delta;
			float * rolling_mean;
			float * rolling_variance;
			float * x;
			float * x_norm;
			float * m;
			float * v;
			float * bias_m;
			float * bias_v;
			float * scale_m;
			float * scale_v;
			float *z_cpu;
			float *r_cpu;
			float *h_cpu;
			float *stored_h_cpu;
			float * prev_state_cpu;
			float *temp_cpu;
			float *temp2_cpu;
			float *temp3_cpu;
			float *dh_cpu;
			float *hh_cpu;
			float *prev_cell_cpu;
			float *cell_cpu;
			float *f_cpu;
			float *i_cpu;
			float *g_cpu;
			float *o_cpu;
			float *c_cpu;
			float *stored_c_cpu;
			float *dc_cpu;
			float *binary_input;
			uint32_t *bin_re_packed_input;
			char *t_bit_input;
			struct layer *input_layer;
			struct layer *self_layer;
			struct layer *output_layer;
			struct layer *reset_layer;
			struct layer *update_layer;
			struct layer *state_layer;
			struct layer *input_gate_layer;
			struct layer *state_gate_layer;
			struct layer *input_save_layer;
			struct layer *state_save_layer;
			struct layer *input_state_layer;
			struct layer *state_state_layer;
			struct layer *input_z_layer;
			struct layer *state_z_layer;
			struct layer *input_r_layer;
			struct layer *state_r_layer;
			struct layer *input_h_layer;
			struct layer *state_h_layer;

			struct layer *wz;
			struct layer *uz;
			struct layer *wr;
			struct layer *ur;
			struct layer *wh;
			struct layer *uh;
			struct layer *uo;
			struct layer *wo;
			struct layer *vo;
			struct layer *uf;
			struct layer *wf;
			struct layer *vf;
			struct layer *ui;
			struct layer *wi;
			struct layer *vi;
			struct layer *ug;
			struct layer *wg;

			tree *softmax_tree;

			size_t workspace_size;

			//#ifdef GPU
			int *indexes_gpu;

			int stream;
			int wait_stream_id;

			float *z_gpu;
			float *r_gpu;
			float *h_gpu;
			float *stored_h_gpu;
			float *bottelneck_hi_gpu;
			float *bottelneck_delta_gpu;

			float *temp_gpu;
			float *temp2_gpu;
			float *temp3_gpu;

			float *dh_gpu;
			float *hh_gpu;
			float *prev_cell_gpu;
			float *prev_state_gpu;
			float *last_prev_state_gpu;
			float *last_prev_cell_gpu;
			float *cell_gpu;
			float *f_gpu;
			float *i_gpu;
			float *g_gpu;
			float *o_gpu;
			float *c_gpu;
			float *stored_c_gpu;
			float *dc_gpu;

			// adam
			float *m_gpu;
			float *v_gpu;
			float *bias_m_gpu;
			float *scale_m_gpu;
			float *bias_v_gpu;
			float *scale_v_gpu;

			float * combine_gpu;
			float * combine_delta_gpu;

			float * forgot_state_gpu;
			float * forgot_delta_gpu;
			float * state_gpu;
			float * state_delta_gpu;
			float * gate_gpu;
			float * gate_delta_gpu;
			float * save_gpu;
			float * save_delta_gpu;
			float * concat_gpu;
			float * concat_delta_gpu;

			float *binary_input_gpu;
			float *binary_weights_gpu;
			float *bin_conv_shortcut_in_gpu;
			float *bin_conv_shortcut_out_gpu;

			float * mean_gpu;
			float * variance_gpu;
			float * m_cbn_avg_gpu;
			float * v_cbn_avg_gpu;

			float * rolling_mean_gpu;
			float * rolling_variance_gpu;

			float * variance_delta_gpu;
			float * mean_delta_gpu;

			float * col_image_gpu;

			float * x_gpu;
			float * x_norm_gpu;
			float * weights_gpu;
			float * weight_updates_gpu;
			float * weight_deform_gpu;
			float * weight_change_gpu;

			float * weights_gpu16;
			float * weight_updates_gpu16;

			float * biases_gpu;
			float * bias_updates_gpu;
			float * bias_change_gpu;

			float * scales_gpu;
			float * scale_updates_gpu;
			float * scale_change_gpu;

			float * input_antialiasing_gpu;
			float * output_gpu;
			float * output_avg_gpu;
			float * activation_input_gpu;
			float * loss_gpu;
			float * delta_gpu;
			float * cos_sim_gpu;
			float * rand_gpu;
			float * drop_blocks_scale;
			float * drop_blocks_scale_gpu;
			float * squared_gpu;
			float * norms_gpu;

			float *gt_gpu;
			float *a_avg_gpu;

			int *input_sizes_gpu;
			float **layers_output_gpu;
			float **layers_delta_gpu;
			//#ifdef CUDNN
			cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
			cudnnTensorDescriptor_t srcTensorDesc16, dstTensorDesc16;
			cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
			cudnnTensorDescriptor_t dsrcTensorDesc16, ddstTensorDesc16;
			cudnnTensorDescriptor_t normTensorDesc, normDstTensorDesc, normDstTensorDescF16;
			cudnnFilterDescriptor_t weightDesc, weightDesc16;
			cudnnFilterDescriptor_t dweightDesc, dweightDesc16;
			cudnnConvolutionDescriptor_t convDesc;
			cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
			cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
			cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
			cudnnPoolingDescriptor_t poolingDesc;
			//#else   // CUDNN
			void* srcTensorDesc, *dstTensorDesc;
			void* srcTensorDesc16, *dstTensorDesc16;
			void* dsrcTensorDesc, *ddstTensorDesc;
			void* dsrcTensorDesc16, *ddstTensorDesc16;
			void* normTensorDesc, *normDstTensorDesc, *normDstTensorDescF16;
			void* weightDesc, *weightDesc16;
			void* dweightDesc, *dweightDesc16;
			void* convDesc;
			UNUSED_ENUM_TYPE fw_algo, fw_algo16;
			UNUSED_ENUM_TYPE bd_algo, bd_algo16;
			UNUSED_ENUM_TYPE bf_algo, bf_algo16;
			void* poolingDesc;
#endif
			;
	} // for() layers

	ofs	<< "----------------------------"							<< std::endl
		<< "batch="					<< parms.batch					<< std::endl
		<< "inputs="				<< parms.inputs					<< std::endl
		<< "h="						<< parms.h						<< std::endl
		<< "w="						<< parms.w						<< std::endl
		<< "c="						<< parms.c						<< std::endl
		<< "index="					<< parms.index					<< std::endl
		<< "time_steps="			<< parms.time_steps				<< std::endl
		<< "train="					<< parms.train					<< std::endl
		<< "last_stop_backward="	<< parms.last_stop_backward		<< std::endl
		<< "avg_outputs="			<< parms.avg_outputs			<< std::endl
		<< "avg_counter="			<< parms.avg_counter			<< std::endl
		<< "bflops="				<< parms.bflops					<< std::endl
		<< "workspace_size="		<< parms.workspace_size			<< std::endl
		<< "max_inputs="			<< parms.max_inputs				<< std::endl
		<< "max_outputs="			<< parms.max_outputs			<< std::endl
		<< "receptive_w="			<< parms.receptive_w			<< std::endl
		<< "receptive_h="			<< parms.receptive_h			<< std::endl
		<< "receptive_w_scale="		<< parms.receptive_w_scale		<< std::endl
		<< "receptive_h_scale="		<< parms.receptive_h_scale		<< std::endl
		<< "show_receptive_field="	<< parms.show_receptive_field	<< std::endl
		;

	return;
}


void Darknet::dump(network * net, const Darknet::CfgFile::CommonParms & parms)
{
	::dump(net, parms);

	return;
}


void Darknet::dump(Darknet::CfgFile & cfg)
{
	::dump(cfg.net, cfg.parms);

	return;
}
