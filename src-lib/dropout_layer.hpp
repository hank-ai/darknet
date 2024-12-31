#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_dropout_layer(int batch, int inputs, float probability, int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c);

void forward_dropout_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_dropout_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_dropout_layer(Darknet::Layer *l, int inputs);

#ifdef DARKNET_GPU
void forward_dropout_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_dropout_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
