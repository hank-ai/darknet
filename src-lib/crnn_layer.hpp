#pragma once

#include "activations.hpp"
#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int xnor, int train);
void resize_crnn_layer(Darknet::Layer *l, int w, int h);
void free_state_crnn(Darknet::Layer & l);

void forward_crnn_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_crnn_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_crnn_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_crnn_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_crnn_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_crnn_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
void push_crnn_layer(Darknet::Layer & l);
void pull_crnn_layer(Darknet::Layer & l);
#endif

#ifdef __cplusplus
}
#endif
