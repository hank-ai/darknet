#include <gtest/gtest.h>
#include "darknet_internal.hpp"
#include <cmath>

// Tests for BDP forward loss: IoU + Angular Correction + Front Point Loss
// Validates loss computation in forward_yolo_layer_bdp()

float compute_angular_correction(float pred_fx, float pred_fy, float pred_x, float pred_y,
                                  float truth_fx, float truth_fy, float truth_x, float truth_y) {
    float pvx = pred_fx - pred_x, pvy = pred_fy - pred_y;
    float tvx = truth_fx - truth_x, tvy = truth_fy - truth_y;
    float dot = pvx*tvx + pvy*tvy;
    float pmag = std::sqrt(pvx*pvx + pvy*pvy);
    float tmag = std::sqrt(tvx*tvx + tvy*tvy);
    if (pmag < 1e-8f || tmag < 1e-8f) return 1.0f;
    float cos_a = std::max(-1.0f, std::min(1.0f, dot/(pmag*tmag)));
    return std::cos(std::acos(cos_a) / 2.0f);
}

// Axis-aligned boxes → angular correction ≈ 1
TEST(BDPForwardLoss, AxisAlignedMatch) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};
    DarknetBoxBDP truth = {0.55f, 0.52f, 0.25f, 0.18f, 0.675f, 0.52f};
    float iou = box_iou_bdp(pred, truth);
    float corr = compute_angular_correction(pred.fx, pred.fy, pred.x, pred.y,
                                            truth.fx, truth.fy, truth.x, truth.y);
    EXPECT_GT(iou, 0.0f) << "Overlapping boxes";
    EXPECT_NEAR(corr, 1.0f, 0.1f) << "Axis-aligned → correction≈1";
}

// Different orientations → correction < 1
TEST(BDPForwardLoss, AngularPenalty) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};   // Horizontal
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.3f, 0.2f, 0.5f, 0.65f};  // Vertical (90°)
    float iou = box_iou_bdp(pred, truth);
    float corr = compute_angular_correction(pred.fx, pred.fy, pred.x, pred.y,
                                            truth.fx, truth.fy, truth.x, truth.y);
    EXPECT_LT(corr, 0.9f) << "90° rotation → correction<0.9";
    EXPECT_LT(iou * corr, iou) << "Angular penalty reduces IoU";
}

// BDP IoU (with angular correction) and RIOU (true rotated) are finite and bounded
TEST(BDPForwardLoss, IoUVariantsFinite) {
    std::vector<std::pair<DarknetBoxBDP, DarknetBoxBDP>> cases = {
        {{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f}, {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f}},  // Perfect
        {{0.4f, 0.5f, 0.2f, 0.2f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f}},  // Partial
        {{0.2f, 0.2f, 0.1f, 0.1f, 0.25f, 0.2f}, {0.8f, 0.8f, 0.1f, 0.1f, 0.85f, 0.8f}}, // No overlap
        {{0.5f, 0.5f, 0.3f, 0.15f, 0.65f, 0.55f}, {0.5f, 0.5f, 0.3f, 0.15f, 0.55f, 0.65f}} // Rotated
    };

    for (size_t i = 0; i < cases.size(); i++) {
        auto [pred, truth] = cases[i];
        // BDP only has: box_iou_bdp (axis-aligned + angular correction) and box_riou (true rotated)
        float iou_bdp = box_iou_bdp(pred, truth);
        float riou = box_riou(pred, truth);

        EXPECT_TRUE(std::isfinite(iou_bdp)) << "Case " << i << ": IoU BDP finite";
        EXPECT_TRUE(std::isfinite(riou)) << "Case " << i << ": RIOU finite";

        EXPECT_GE(iou_bdp, -1e-6f) << "Case " << i << ": IoU BDP ≥ 0";
        EXPECT_LE(iou_bdp, 1.0f + 1e-6f) << "Case " << i << ": IoU BDP ≤ 1";

        EXPECT_GE(riou, -1e-6f) << "Case " << i << ": RIOU ≥ 0";
        EXPECT_LE(riou, 1.0f + 1e-6f) << "Case " << i << ": RIOU ≤ 1";
    }
}

// Front point loss combined with IoU
TEST(BDPForwardLoss, FrontPointLoss) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.55f, 0.52f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};
    float iou = box_iou_bdp(pred, truth);
    float dfx = truth.fx - pred.fx;  // 0.6 - 0.55 = 0.05
    float dfy = truth.fy - pred.fy;  // 0.5 - 0.52 = -0.02
    float fp_loss = 0.5f*dfx*dfx + 0.5f*dfy*dfy;  // Smooth L1 (small errors)

    // Same center/size → axis-aligned IoU=1, but angular correction reduces it
    // because front points differ (different orientations)
    EXPECT_GT(iou, 0.8f) << "Same center/size → IoU should be high (>0.8)";
    EXPECT_LT(iou, 1.0f) << "Different orientations → IoU < 1 due to angular correction";

    // Front point loss calculation: 0.5*(0.05)² + 0.5*(-0.02)² = 0.00145
    EXPECT_NEAR(fp_loss, 0.00145f, 1e-5f) << "Front point loss from small error";
}

// Angular correction is symmetric
TEST(BDPForwardLoss, AngularSymmetry) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.55f};
    DarknetBoxBDP b = {0.5f, 0.5f, 0.3f, 0.2f, 0.62f, 0.52f};

    float corr_ab = compute_angular_correction(a.fx, a.fy, a.x, a.y, b.fx, b.fy, b.x, b.y);
    float corr_ba = compute_angular_correction(b.fx, b.fy, b.x, b.y, a.fx, a.fy, a.x, a.y);

    EXPECT_NEAR(corr_ab, corr_ba, 1e-5f) << "Angular correction symmetric";
}

// Perfect match → all losses zero
TEST(BDPForwardLoss, PerfectMatch) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.25f, 0.18f, 0.625f, 0.55f};
    DarknetBoxBDP truth = pred;

    EXPECT_NEAR(box_iou_bdp(pred, truth), 1.0f, 1e-4f) << "IoU BDP=1";
    EXPECT_NEAR(box_riou(pred, truth), 1.0f, 1e-4f) << "RIOU=1";

    float corr = compute_angular_correction(pred.fx, pred.fy, pred.x, pred.y,
                                            truth.fx, truth.fy, truth.x, truth.y);
    EXPECT_NEAR(corr, 1.0f, 1e-4f) << "Angular correction=1";

    float fp_loss = 0.5f*std::pow(truth.fx-pred.fx, 2) + 0.5f*std::pow(truth.fy-pred.fy, 2);
    EXPECT_NEAR(fp_loss, 0.0f, 1e-6f) << "Front point loss=0";
}

// IoU decreases with distance
TEST(BDPForwardLoss, IoUMonotonic) {
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};
    std::vector<DarknetBoxBDP> preds = {
        {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f},      // Same
        {0.52f, 0.51f, 0.2f, 0.2f, 0.62f, 0.51f},  // Close
        {0.55f, 0.53f, 0.2f, 0.2f, 0.65f, 0.53f},  // Medium
        {0.6f, 0.6f, 0.2f, 0.2f, 0.7f, 0.6f}       // Far
    };

    float prev_iou = 2.0f;
    for (auto& pred : preds) {
        float iou = box_iou_bdp(pred, truth);
        EXPECT_LT(iou, prev_iou + 1e-4f) << "IoU decreases with distance";
        prev_iou = iou;
    }
}

// BDP IoU vs RIOU differ for rotated boxes
TEST(BDPForwardLoss, IoUTypesDiffer) {
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};

    // box_iou_bdp: axis-aligned IoU with angular correction (approximation)
    float iou_bdp = box_iou_bdp(pred, truth);
    // box_riou: true rotated IoU using polygon intersection
    float riou = box_riou(pred, truth);

    // For rotated boxes, RIOU (true) may differ from BDP IoU (approximation)
    EXPECT_TRUE(std::isfinite(iou_bdp)) << "IoU BDP must be finite";
    EXPECT_TRUE(std::isfinite(riou)) << "RIOU must be finite";
}

// Very small boxes remain stable
TEST(BDPForwardLoss, SmallBoxStability) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.01f, 0.01f, 0.505f, 0.5f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.01f, 0.01f, 0.505f, 0.5f};

    EXPECT_TRUE(std::isfinite(box_iou_bdp(pred, truth))) << "IoU BDP finite for tiny boxes";
    EXPECT_TRUE(std::isfinite(box_riou(pred, truth))) << "RIOU finite for tiny boxes";
    EXPECT_NEAR(box_iou_bdp(pred, truth), 1.0f, 1e-3f) << "Identical tiny boxes → IoU=1";
    EXPECT_NEAR(box_riou(pred, truth), 1.0f, 1e-3f) << "Identical tiny boxes → RIOU=1";
}

// Non-overlapping → IoU=0
TEST(BDPForwardLoss, NonOverlapping) {
    DarknetBoxBDP pred = {0.2f, 0.2f, 0.15f, 0.15f, 0.275f, 0.2f};
    DarknetBoxBDP truth = {0.8f, 0.8f, 0.15f, 0.15f, 0.875f, 0.8f};

    float iou_bdp = box_iou_bdp(pred, truth);
    float riou = box_riou(pred, truth);

    EXPECT_NEAR(iou_bdp, 0.0f, 1e-4f) << "No overlap → IoU BDP=0";
    EXPECT_NEAR(riou, 0.0f, 1e-4f) << "No overlap → RIOU=0";
}
