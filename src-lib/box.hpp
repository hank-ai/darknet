#pragma once

#include "darknet_internal.hpp"


struct dbox
{
	float dx;
	float dy;
	float dw;
	float dh;
};


struct detection_with_class
{
	Darknet::Detection det;
	/// The most probable class id.  The best class index in this->prob.
	/// Is filled temporary when processing results, otherwise not initialized.
	int best_class;
};


Darknet::Box float_to_box(const float * f);
Darknet::Box float_to_box_stride(const float * f, const int stride);
float box_iou_kind(const Darknet::Box & a, const Darknet::Box & b, const IOU_LOSS iou_kind);
float box_iou(const Darknet::Box & a, const Darknet::Box & b);
float box_giou(const Darknet::Box & a, const Darknet::Box & b);
float box_diou(const Darknet::Box & a, const Darknet::Box & b);
float box_ciou(const Darknet::Box & a, const Darknet::Box & b);
dxrep dx_box_iou(const Darknet::Box & a, const Darknet::Box & b, const IOU_LOSS iou_loss);
dbox diou(const Darknet::Box & a, const Darknet::Box & b);
float box_rmse(const Darknet::Box & a, const Darknet::Box & b);
void do_nms(Darknet::Box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(Darknet::Box *boxes, float **probs, int total, int classes, float thresh);
Darknet::Box decode_box(const Darknet::Box & b, const Darknet::Box & anchor);
Darknet::Box encode_box(const Darknet::Box & b, const Darknet::Box & anchor);

/// Creates array of detections with prob > thresh and fills best_class for them
/// Return number of selected detections in *selected_detections_num
detection_with_class * get_actual_detections(const Darknet::Detection *dets, int dets_num, float thresh, int* selected_detections_num, const Darknet::VStr & names);
