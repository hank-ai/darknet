#pragma once

#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes);
void forward_region_layer(const layer l, network_state state);
void backward_region_layer(const layer l, network_state state);
void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void resize_region_layer(layer *l, int w, int h);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
void zero_objectness(layer l);

#ifdef GPU
void forward_region_layer_gpu(const layer l, network_state state);
void backward_region_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
