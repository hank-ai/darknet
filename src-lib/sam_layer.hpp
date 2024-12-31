#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_sam_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_sam_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_sam_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_sam_layer(Darknet::Layer *l, int w, int h);

#ifdef DARKNET_GPU
void forward_sam_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_sam_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
