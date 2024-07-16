#pragma once

#include "image.hpp"
#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(layer *l, int w, int h);
void forward_normalization_layer(const layer l, network_state state);
void backward_normalization_layer(const layer l, network_state state);
void visualize_normalization_layer(layer l, char *window);

#ifdef GPU
void forward_normalization_layer_gpu(const layer l, network_state state);
void backward_normalization_layer_gpu(const layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
