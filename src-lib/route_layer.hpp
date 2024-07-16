#pragma once

#include "network.hpp"
#include "layer.hpp"

#ifdef __cplusplus
extern "C" {
#endif
layer make_route_layer(int batch, int n, int *input_layers, int *input_size, int groups, int group_id);
void forward_route_layer(const layer l, network_state state);
void backward_route_layer(const layer l, network_state state);
void resize_route_layer(layer *l, network *net);

#ifdef GPU
void forward_route_layer_gpu(const layer l, network_state state);
void backward_route_layer_gpu(const layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
