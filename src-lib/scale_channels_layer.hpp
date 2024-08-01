#pragma once

#include "darknet_internal.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_scale_channels_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2, int scale_wh);
void forward_scale_channels_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_scale_channels_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_scale_channels_layer(Darknet::Layer *l, network *net);

#ifdef GPU
void forward_scale_channels_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_scale_channels_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif

#ifdef __cplusplus
}
#endif
