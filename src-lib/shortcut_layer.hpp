#pragma once

#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_shortcut_layer(int batch, int n, int *input_layers, int* input_sizes, int w, int h, int c, float **layers_output, float **layers_delta, float **layers_output_gpu, float **layers_delta_gpu, WEIGHTS_TYPE_T weights_type, WEIGHTS_NORMALIZATION_T weights_normalization, ACTIVATION activation, int train);
void forward_shortcut_layer(Darknet::Layer & l, network_state state);
void backward_shortcut_layer(Darknet::Layer & l, network_state state);
void update_shortcut_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay);
void resize_shortcut_layer(Darknet::Layer *l, int w, int h, network *net);

#ifdef GPU
void forward_shortcut_layer_gpu(Darknet::Layer & l, network_state state);
void backward_shortcut_layer_gpu(Darknet::Layer & l, network_state state);
void update_shortcut_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale);
void pull_shortcut_layer(Darknet::Layer & l);
void push_shortcut_layer(Darknet::Layer & l);
#endif

#ifdef __cplusplus
}
#endif
