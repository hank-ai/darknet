// Oh boy, why am I about to do this....

// ^ That comment was the very first commit to the original Darknet repo by Joseph Redmon on Nov 4, 2013.

#pragma once

#include "darknet.h"

/*
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;
*/

namespace Darknet
{
	struct NetworkState
	{
		float *truth;
		float *input;
		float *delta;
		float *workspace;
		int train;
		int index;
		network net;
	};
}

extern "C"
{

#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);
void forward_network_gpu(network net, Darknet::NetworkState state);
void backward_network_gpu(network net, Darknet::NetworkState state);
void update_network_gpu(network net);
void forward_backward_network_gpu(network net, float *x, float *y);
#endif

float get_current_seq_subdivisions(network net);
int get_sequence_value(network net);
float get_current_rate(network net);
int get_current_batch(network net);
int64_t get_current_iteration(network net);
//void free_network(network net); // darknet.h
void compare_networks(network n1, network n2, data d);

network make_network(int n);
void forward_network(network net, Darknet::NetworkState state);
void backward_network(network net, Darknet::NetworkState state);
void update_network(network net);

float train_network(network net, data d);
float train_network_waitkey(network net, data d, int wait_key);
float train_network_batch(network net, data d, int n);
float train_network_datum(network net, float *x, float *y);

matrix network_predict_data(network net, data test);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
Darknet::Image get_network_image(network net);
Darknet::Image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);

int get_network_nuisance(network net);
int get_network_background(network net);
network combine_train_valid_networks(network net_train, network net_map);
void copy_weights_net(network net_train, network *net_map);
void free_network_recurrent_state(network net);
void randomize_network_recurrent_state(network net);
void remember_network_recurrent_state(network net);
void restore_network_recurrent_state(network net);
int is_ema_initialized(network net);
void ema_update(network net, float ema_alpha);
void ema_apply(network net);
void reject_similar_weights(network net, float sim_threshold);

}
