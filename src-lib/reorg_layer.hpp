#pragma once

#include "image.hpp"
#include "dark_cuda.hpp"
#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse);
void resize_reorg_layer(Darknet::Layer *l, int w, int h);
void forward_reorg_layer(Darknet::Layer & l, network_state state);
void backward_reorg_layer(Darknet::Layer & l, network_state state);

#ifdef GPU
void forward_reorg_layer_gpu(Darknet::Layer & l, network_state state);
void backward_reorg_layer_gpu(Darknet::Layer & l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
