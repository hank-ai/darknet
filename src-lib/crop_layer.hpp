#pragma once

//#include "image.hpp"
#include "layer.hpp"
//#include "network.hpp"

typedef layer crop_layer;

#ifdef __cplusplus
extern "C" {
#endif
image get_crop_image(crop_layer l);
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer(const crop_layer l, network_state state);
void resize_crop_layer(layer *l, int w, int h);

#ifdef GPU
void forward_crop_layer_gpu(crop_layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
