#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes);
void forward_region_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_region_layer(Darknet::Layer & l, Darknet::NetworkState state);
void get_region_boxes(const Darknet::Layer & l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void resize_region_layer(Darknet::Layer *l, int w, int h);
void get_region_detections(Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
void zero_objectness(Darknet::Layer & l);

#ifdef GPU
void forward_region_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_region_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
