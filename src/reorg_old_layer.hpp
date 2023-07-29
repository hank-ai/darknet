#pragma once

#include "image.hpp"
#include "dark_cuda.hpp"
#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
layer make_reorg_old_layer(int batch, int w, int h, int c, int stride, int reverse);
void resize_reorg_old_layer(layer *l, int w, int h);
void forward_reorg_old_layer(const layer l, network_state state);
void backward_reorg_old_layer(const layer l, network_state state);

#ifdef GPU
void forward_reorg_old_layer_gpu(layer l, network_state state);
void backward_reorg_old_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
