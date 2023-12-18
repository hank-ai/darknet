#pragma once

#include "dark_cuda.hpp"
#include "image.hpp"
#include "activations.hpp"
#include "layer.hpp"
#include "network.hpp"

typedef layer local_layer;

#ifdef __cplusplus
extern "C" {
#endif
#ifdef GPU
void forward_local_layer_gpu(local_layer layer, network_state state);
void backward_local_layer_gpu(local_layer layer, network_state state);
void update_local_layer_gpu(local_layer layer, int batch, float learning_rate, float momentum, float decay, float loss_scale);

void push_local_layer(local_layer layer);
void pull_local_layer(local_layer layer);
#endif

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);

void forward_local_layer(const local_layer layer, network_state state);
void backward_local_layer(local_layer layer, network_state state);
void update_local_layer(local_layer layer, int batch, float learning_rate, float momentum, float decay);

void bias_output(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

#ifdef __cplusplus
}
#endif
