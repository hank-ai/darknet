#include <gtest/gtest.h>
#include "darknet_internal.hpp"

// Tests for box_iou_bdp: Axis-aligned IoU for BDP boxes
// Computes IoU using only (x,y,w,h), ignoring front point (fx,fy)

// Perfect match → IoU = 1.0
TEST(BoxIouBDP, IdenticalBoxes) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};
    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};
    EXPECT_NEAR(box_iou_bdp(a, b), 1.0f, 1e-5f) << "Identical boxes → IoU=1.0";
}

// No overlap → IoU = 0.0
TEST(BoxIouBDP, NoOverlap) {
    DarknetBoxBDP a = {0.2f, 0.2f, 0.1f, 0.1f, 0.25f, 0.2f};
    DarknetBoxBDP b = {0.8f, 0.8f, 0.1f, 0.1f, 0.85f, 0.8f};
    EXPECT_NEAR(box_iou_bdp(a, b), 0.0f, 1e-5f) << "No overlap → IoU=0.0";
}

// Partial overlap → 0 < IoU < 1
TEST(BoxIouBDP, PartialOverlap) {
    DarknetBoxBDP a = {0.4f, 0.5f, 0.2f, 0.2f, 0.5f, 0.5f};
    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};
    float iou = box_iou_bdp(a, b);
    EXPECT_GT(iou, 0.0f) << "Partial overlap → IoU>0";
    EXPECT_LT(iou, 1.0f) << "Partial overlap → IoU<1";
}

// Front point affects IoU via angular correction: IoU_BDP = RIOU * cos(angle/2)
TEST(BoxIouBDP, IncludesAngularCorrection) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};  // fp right (0°)
    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.6f};  // fp up (90°)
    // Same position/size → RIOU=1.0, but different orientation (90°)
    // Angular correction: cos(90°/2) = cos(45°) ≈ 0.707
    float expected = std::cos(M_PI / 4.0f);  // cos(45°) = √2/2
    EXPECT_NEAR(box_iou_bdp(a, b), expected, 1e-5f) << "Same x,y,w,h but 90° rotation → IoU≈0.707";
}

// IoU(a,b) = IoU(b,a)
TEST(BoxIouBDP, Symmetry) {
    DarknetBoxBDP a = {0.4f, 0.5f, 0.25f, 0.15f, 0.5f, 0.5f};
    DarknetBoxBDP b = {0.6f, 0.5f, 0.20f, 0.12f, 0.7f, 0.5f};
    EXPECT_NEAR(box_iou_bdp(a, b), box_iou_bdp(b, a), 1e-5f) << "IoU symmetric";
}

// IoU ∈ [0,1]
TEST(BoxIouBDP, ValidRange) {
    DarknetBoxBDP a = {0.3f, 0.4f, 0.15f, 0.12f, 0.375f, 0.4f};
    DarknetBoxBDP b = {0.7f, 0.6f, 0.18f, 0.14f, 0.79f, 0.6f};
    float iou = box_iou_bdp(a, b);
    EXPECT_GE(iou, 0.0f) << "IoU ≥ 0";
    EXPECT_LE(iou, 1.0f) << "IoU ≤ 1";
    EXPECT_TRUE(std::isfinite(iou)) << "IoU finite";
}

// Small box inside large box
TEST(BoxIouBDP, ContainedBox) {
    DarknetBoxBDP large = {0.5f, 0.5f, 0.6f, 0.6f, 0.8f, 0.5f};
    DarknetBoxBDP small = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};
    float iou = box_iou_bdp(small, large);
    EXPECT_GT(iou, 0.0f) << "Contained → IoU>0";
    EXPECT_LT(iou, 1.0f) << "Contained → IoU<1";
}

// Numerical stability for tiny boxes
TEST(BoxIouBDP, VerySmallBoxes) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.01f, 0.01f, 0.505f, 0.5f};
    DarknetBoxBDP b = a;
    EXPECT_NEAR(box_iou_bdp(a, b), 1.0f, 1e-4f) << "Tiny identical boxes → IoU=1.0";
    EXPECT_TRUE(std::isfinite(box_iou_bdp(a, b))) << "Tiny boxes → finite IoU";
}
