#pragma once

//#include "image.hpp"
#include "dark_cuda.hpp"
//#include "layer.hpp"
//#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
image get_maxpool_image(Darknet::Layer /*&*/ l);
Darknet::Layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x, int stride_y, int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train);
void resize_maxpool_layer(Darknet::Layer *l, int w, int h);
void forward_maxpool_layer(Darknet::Layer & l, network_state state);
void backward_maxpool_layer(Darknet::Layer & l, network_state state);

void forward_local_avgpool_layer(Darknet::Layer & l, network_state state);
void backward_local_avgpool_layer(Darknet::Layer & l, network_state state);

#ifdef GPU
void forward_maxpool_layer_gpu(Darknet::Layer & l, network_state state);
void backward_maxpool_layer_gpu(Darknet::Layer & l, network_state state);
void cudnn_maxpool_setup(Darknet::Layer *l);

void forward_local_avgpool_layer_gpu(Darknet::Layer & l, network_state state);
void backward_local_avgpool_layer_gpu(Darknet::Layer & l, network_state state);
#endif // GPU

#ifdef __cplusplus
}
#endif
