#pragma once

//#include "image.hpp"
#include "layer.hpp"
//#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_batchnorm_layer(int batch, int w, int h, int c, int train);
void forward_batchnorm_layer(Darknet::Layer & l, network_state state);
void backward_batchnorm_layer(Darknet::Layer & l, network_state state);
void update_batchnorm_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

void resize_batchnorm_layer(Darknet::Layer *l, int w, int h);

#ifdef GPU
void forward_batchnorm_layer_gpu(Darknet::Layer & l, network_state state);
void backward_batchnorm_layer_gpu(Darknet::Layer & l, network_state state);
void update_batchnorm_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale);
void pull_batchnorm_layer(Darknet::Layer /*&*/ l);
void push_batchnorm_layer(Darknet::Layer /*&*/ l);
#endif

#ifdef __cplusplus
}
#endif
