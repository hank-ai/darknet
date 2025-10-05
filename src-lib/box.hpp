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

// BDP-specific IoU functions - RIOU only (rotated IoU with angular correction)
float box_iou_bdp(const DarknetBoxBDP & a, const DarknetBoxBDP & b);

// Rotated IoU functions - true polygon intersection for oriented bounding boxes
// These use fx,fy to determine box orientation and compute true rotated rectangle IoU
float box_riou(const DarknetBoxBDP & a, const DarknetBoxBDP & b);
dxrep_bdp dx_box_riou(const DarknetBoxBDP & a, const DarknetBoxBDP & b, const IOU_LOSS iou_loss);
float box_rmse(const Darknet::Box & a, const Darknet::Box & b);

// BDP to pixel coordinate conversion structures for RIOU loss calculation
// These convert normalized BDP parameters (x,y,w,h,fx,fy) to actual pixel corners
// for computing true rotated IoU loss during training
struct Point2D {
    float x, y;
    Point2D() : x(0.0f), y(0.0f) {}
    Point2D(float x_, float y_) : x(x_), y(y_) {}
};

struct RectParams {
    float x, y;      // Center (normalized [0,1])
    float w, h;      // Dimensions (normalized [0,1])
    float fx, fy;    // Front point (normalized [0,1])

    bool isValid() const {
        return std::isfinite(x) && std::isfinite(y) &&
               std::isfinite(w) && std::isfinite(h) &&
               std::isfinite(fx) && std::isfinite(fy) &&
               x > 0.0f && w > 0.0f && h > 0.0f && fx > 0.0f && fy > 0.0f && y > 0.0f;
    }
};

struct RectCorners {
    Point2D p1, p2, p3, p4;  // Top-left, top-right, bottom-right, bottom-left
};

// Transform between BDP normalized params and pixel corner coordinates
// Used for RIOU loss calculation during training
class RotatedRectTransform {
public:
    static RectCorners forward(const RectParams& params, int imageWidth, int imageHeight);
    static std::optional<RectParams> inverse(const RectCorners& corners, int imageWidth, int imageHeight);
};

// Angular correction: computes cos(angle/2) between predicted and target orientation vectors
// Used for BDP loss calculation to penalize orientation misalignment in rotated IoU
// Returns value in [0,1] where 1=perfect alignment, 0=opposite directions (180°)
float computeFrontPointCosine(const RectParams& pred, const RectParams& target);

void do_nms(Darknet::Box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(Darknet::Box *boxes, float **probs, int total, int classes, float thresh);
Darknet::Box decode_box(const Darknet::Box & b, const Darknet::Box & anchor);
Darknet::Box encode_box(const Darknet::Box & b, const Darknet::Box & anchor);

/// Creates array of detections with prob > thresh and fills best_class for them
/// Return number of selected detections in *selected_detections_num
detection_with_class * get_actual_detections(const Darknet::Detection *dets, int dets_num, float thresh, int* selected_detections_num, const Darknet::VStr & names);
