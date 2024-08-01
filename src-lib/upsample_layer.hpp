#pragma once

#include "darknet_internal.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_upsample_layer(int batch, int w, int h, int c, int stride);
void forward_upsample_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_upsample_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_upsample_layer(Darknet::Layer *l, int w, int h);

#ifdef GPU
void forward_upsample_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_upsample_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif

#ifdef __cplusplus
}
#endif
