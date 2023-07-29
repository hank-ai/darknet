#pragma once

#include "activations.hpp"
#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

void forward_activation_layer(layer l, network_state state);
void backward_activation_layer(layer l, network_state state);

#ifdef GPU
void forward_activation_layer_gpu(layer l, network_state state);
void backward_activation_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
