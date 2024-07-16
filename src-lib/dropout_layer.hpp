#pragma once

#include "layer.hpp"
//#include "network.hpp"
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_dropout_layer(int batch, int inputs, float probability, int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c);

void forward_dropout_layer(layer l, network_state state);
void backward_dropout_layer(layer l, network_state state);
void resize_dropout_layer(layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(layer l, network_state state);
void backward_dropout_layer_gpu(layer l, network_state state);
#endif
#ifdef __cplusplus
}
#endif
