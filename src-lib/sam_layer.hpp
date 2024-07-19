#pragma once

#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_sam_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_sam_layer(Darknet::Layer & l, network_state state);
void backward_sam_layer(Darknet::Layer & l, network_state state);
void resize_sam_layer(Darknet::Layer *l, int w, int h);

#ifdef GPU
void forward_sam_layer_gpu(Darknet::Layer & l, network_state state);
void backward_sam_layer_gpu(Darknet::Layer & l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
