#pragma once

//#include "darknet.h"
#include "layer.hpp"
#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
Darknet::Layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_yolo_layer(Darknet::Layer & l, network_state state);
void backward_yolo_layer(Darknet::Layer & l, network_state state);
void resize_yolo_layer(Darknet::Layer *l, int w, int h);
int yolo_num_detections(const Darknet::Layer & l, float thresh);
int yolo_num_detections_batch(const Darknet::Layer & l, float thresh, int batch);
int get_yolo_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
int get_yolo_detections_batch(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch);
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_yolo_layer_gpu(Darknet::Layer & l, network_state state);
void backward_yolo_layer_gpu(Darknet::Layer & l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
