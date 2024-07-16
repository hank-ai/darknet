#pragma once

//#include "image.hpp"
#include "layer.hpp"
//#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
image get_crop_image(layer l);
layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer(const layer l, network_state state);
void resize_crop_layer(layer *l, int w, int h);

#ifdef GPU
void forward_crop_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
