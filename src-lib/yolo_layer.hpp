#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
Darknet::Layer make_yolo_layer_bdp(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_yolo_layer(Darknet::Layer *l, int w, int h);
int yolo_num_detections(const Darknet::Layer & l, float thresh);
int yolo_num_detections_batch(const Darknet::Layer & l, float thresh, int batch);
int get_yolo_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter);
int get_yolo_detections_batch(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter, int batch);
void correct_yolo_boxes(Darknet::Detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

/// @implement OBB: Forward pass functions for oriented bounding boxes (BDP representation)
void forward_yolo_layer_bdp(Darknet::Layer & l, Darknet::NetworkState state);
int yolo_num_detections_bdp(const Darknet::Layer & l, float thresh);
int get_yolo_detections_bdp(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, DarknetDetectionOBB *dets, int letter);
void correct_yolo_boxes_bdp(DarknetDetectionOBB *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef DARKNET_GPU
void forward_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
