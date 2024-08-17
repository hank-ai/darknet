#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_route_layer(int batch, int n, int *input_layers, int *input_size, int groups, int group_id);
void forward_route_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_route_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_route_layer(Darknet::Layer *l, Darknet::Network *net);

#ifdef GPU
void forward_route_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_route_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
