#pragma once

#include "darknet_internal.hpp"

/// @todo what is this?
#define USET

Darknet::Layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);

void forward_lstm_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_lstm_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_lstm_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

#ifdef DARKNET_GPU
void forward_lstm_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_lstm_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_lstm_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
#endif
