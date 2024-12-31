#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize);
size_t get_connected_workspace_size(const Darknet::Layer & l);

void forward_connected_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_connected_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_connected_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
void denormalize_connected_layer(Darknet::Layer & l);
void statistics_connected_layer(Darknet::Layer & l);

#ifdef DARKNET_GPU
void forward_connected_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_connected_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_connected_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
void push_connected_layer(Darknet::Layer & l);
void pull_connected_layer(Darknet::Layer & l);
#endif
