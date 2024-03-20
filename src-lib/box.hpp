#pragma once

#include "darknet.h"


struct dbox
{
	float dx;
	float dy;
	float dw;
	float dh;
};


struct detection_with_class
{
	detection det;
	/// The most probable class id.  The best class index in this->prob.
	/// Is filled temporary when processing results, otherwise not initialized.
	int best_class;
};


box float_to_box(const float * f);
box float_to_box_stride(const float * f, const int stride);
float box_iou_kind(const box & a, const box & b, const IOU_LOSS iou_kind);
float box_iou(const box & a, const box & b);
float box_giou(const box & a, const box & b);
float box_diou(const box & a, const box & b);
float box_ciou(const box & a, const box & b);
dxrep dx_box_iou(const box & a, const box & b, const IOU_LOSS iou_loss);
boxabs to_tblr(const box & a);
dbox diou(const box & a, const box & b);
float box_rmse(const box & a, const box & b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh);
box decode_box(const box & b, const box & anchor);
box encode_box(const box & b, const box & anchor);

/// Creates array of detections with prob > thresh and fills best_class for them
/// Return number of selected detections in *selected_detections_num
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num, char **names);
