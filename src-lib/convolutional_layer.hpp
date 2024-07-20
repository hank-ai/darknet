#pragma once

#include "dark_cuda.hpp"
//#include "image.hpp"
#include "activations.hpp"
#include "layer.hpp"
//#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef GPU
void forward_convolutional_layer_gpu(Darknet::Layer & l, network_state state);
void backward_convolutional_layer_gpu(Darknet::Layer & l, network_state state);
void update_convolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

void push_convolutional_layer(Darknet::Layer & l);
void pull_convolutional_layer(Darknet::Layer & l);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#ifdef CUDNN
void cudnn_convolutional_setup(Darknet::Layer *l, int cudnn_preference, size_t workspace_size_specify);
void create_convolutional_cudnn_tensors(Darknet::Layer *l);
void cuda_convert_f32_to_f16(float* input_f32, size_t size, float *output_f16);
#endif
#endif
void free_convolutional_batchnorm(Darknet::Layer *l);

size_t get_convolutional_workspace_size(const Darknet::Layer & l);
Darknet::Layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, Darknet::Layer * share_layer, int assisted_excitation, int deform, int train);
void denormalize_convolutional_layer(Darknet::Layer & l);
void set_specified_workspace_limit(Darknet::Layer *l, size_t workspace_size_limit);
void resize_convolutional_layer(Darknet::Layer * l, int w, int h);
void forward_convolutional_layer(Darknet::Layer & l, network_state state);
void update_convolutional_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
image *visualize_convolutional_layer(const Darknet::Layer & l, const char * window, image * prev_weights);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(Darknet::Layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void binary_align_weights(Darknet::Layer *l);

void backward_convolutional_layer(Darknet::Layer & l, network_state state);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(const Darknet::Layer & l);
image get_convolutional_delta(const Darknet::Layer & l);
image get_convolutional_weight(const Darknet::Layer & l, int i);


int convolutional_out_height(const Darknet::Layer & l);
int convolutional_out_width(const Darknet::Layer & l);
void rescale_weights(Darknet::Layer & l, float scale, float trans);
void rgbgr_weights(const Darknet::Layer & l);
void assisted_excitation_forward(Darknet::Layer & l, network_state state);
void assisted_excitation_forward_gpu(Darknet::Layer & l, network_state state);

#ifdef __cplusplus
}
#endif
