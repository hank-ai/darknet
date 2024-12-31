#pragma once

#include "darknet_internal.hpp"

/// @todo what is this?
#define USET

Darknet::Layer make_rnn_layer(int batch, int inputs, int hidden, int outputs, int steps, ACTIVATION activation, int batch_normalize, int log);

void forward_rnn_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_rnn_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_rnn_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

#ifdef DARKNET_GPU
void forward_rnn_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_rnn_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_rnn_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
void push_rnn_layer(Darknet::Layer & l);
void pull_rnn_layer(Darknet::Layer & l);
#endif
