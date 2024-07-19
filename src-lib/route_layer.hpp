#pragma once

#include "network.hpp"
#include "layer.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_route_layer(int batch, int n, int *input_layers, int *input_size, int groups, int group_id);
void forward_route_layer(Darknet::Layer & l, network_state state);
void backward_route_layer(Darknet::Layer & l, network_state state);
void resize_route_layer(Darknet::Layer *l, network *net);

#ifdef GPU
void forward_route_layer_gpu(Darknet::Layer & l, network_state state);
void backward_route_layer_gpu(Darknet::Layer & l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
