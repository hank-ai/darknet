#pragma once

//Gaussian YOLOv3 implementation

#include "darknet.h"
#include "layer.hpp"
#include "network.hpp"

Darknet::Layer make_gaussian_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_gaussian_yolo_layer(Darknet::Layer & l, network_state state);
void backward_gaussian_yolo_layer(Darknet::Layer & l, network_state state);
void resize_gaussian_yolo_layer(Darknet::Layer *l, int w, int h);
int gaussian_yolo_num_detections(const Darknet::Layer & l, float thresh);
int get_gaussian_yolo_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
void correct_gaussian_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_gaussian_yolo_layer_gpu(Darknet::Layer & l, network_state state);
void backward_gaussian_yolo_layer_gpu(Darknet::Layer & l, network_state state);
#endif
