#pragma once

#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
void softmax_array(float *input, int n, float temp, float *output);
layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const layer l, network_state state);
void backward_softmax_layer(const layer l, network_state state);

#ifdef GPU
void pull_softmax_layer_output(const layer l);
void forward_softmax_layer_gpu(const layer l, network_state state);
void backward_softmax_layer_gpu(const layer l, network_state state);
#endif

//-----------------------

layer make_contrastive_layer(int batch, int w, int h, int n, int classes, int inputs, layer *yolo_layer);
void forward_contrastive_layer(layer l, network_state state);
void backward_contrastive_layer(layer l, network_state net);

#ifdef GPU
void pull_contrastive_layer_output(const layer l);
void push_contrastive_layer_output(const layer l);
void forward_contrastive_layer_gpu(layer l, network_state state);
void backward_contrastive_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
