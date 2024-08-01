#pragma once

#include "darknet_internal.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_dropout_layer(int batch, int inputs, float probability, int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c);

void forward_dropout_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_dropout_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_dropout_layer(Darknet::Layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_dropout_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
#ifdef __cplusplus
}
#endif
