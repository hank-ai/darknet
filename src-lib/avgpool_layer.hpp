#pragma once

#include "darknet_internal.hpp"

#ifdef __cplusplus
extern "C" {
#endif

image get_avgpool_image(Darknet::Layer & l);
Darknet::Layer make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(Darknet::Layer *l, int w, int h);
void forward_avgpool_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_avgpool_layer(Darknet::Layer & l, Darknet::NetworkState state);

#ifdef GPU
void forward_avgpool_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_avgpool_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif

#ifdef __cplusplus
}
#endif
