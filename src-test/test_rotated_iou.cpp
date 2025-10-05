#include <gtest/gtest.h>
#include "darknet_internal.hpp"

// Tests for box_riou: True rotated IoU using Sutherland-Hodgman polygon clipping
// Unlike box_iou_bdp (axis-aligned), box_riou accounts for box orientation via front point

// Identical boxes → RIoU = 1.0
TEST(RotatedIoU, PerfectOverlap) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};
    EXPECT_NEAR(box_riou(a, a), 1.0f, 1e-4f) << "Identical → RIoU=1.0";
}

// No overlap → RIoU = 0.0
TEST(RotatedIoU, NoOverlap) {
    DarknetBoxBDP a = {0.3f, 0.3f, 0.1f, 0.1f, 0.35f, 0.3f};
    DarknetBoxBDP b = {0.7f, 0.7f, 0.1f, 0.1f, 0.75f, 0.7f};
    EXPECT_NEAR(box_riou(a, b), 0.0f, 1e-4f) << "No overlap → RIoU=0.0";
}

// Axis-aligned boxes
TEST(RotatedIoU, AxisAlignedBoxes) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.4f, 0.2f, 0.7f, 0.5f};
    DarknetBoxBDP b = {0.55f, 0.52f, 0.3f, 0.15f, 0.7f, 0.52f};
    float iou = box_riou(a, b);
    EXPECT_GT(iou, 0.0f) << "Overlapping → RIoU>0";
    EXPECT_LT(iou, 1.0f) << "Partial → RIoU<1";
    EXPECT_TRUE(std::isfinite(iou)) << "Finite RIoU";
}

// 45° rotated square
TEST(RotatedIoU, Rotated45Degrees) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.2f, 0.2f, 0.641f, 0.641f};  // 45°
    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.2f, 0.7f, 0.5f};      // 0°
    float iou = box_riou(a, b);
    // 45° rotated square: intersection area / union area ≈ 0.707 (√2/2)
    EXPECT_GT(iou, 0.65f) << "45° rotation → significant overlap";
    EXPECT_LT(iou, 0.75f) << "45° rotation → RIoU≈0.707";
}

// Perpendicular boxes (90°)
TEST(RotatedIoU, PerpendicularBoxes) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.4f, 0.1f, 0.7f, 0.5f};   // Horizontal (fp to right)
    DarknetBoxBDP b = {0.5f, 0.5f, 0.4f, 0.1f, 0.5f, 0.7f};   // Vertical (fp up) - swapped w/h
    float iou = box_riou(a, b);
    EXPECT_GE(iou, 0.0f) << "RIoU ≥ 0";
    EXPECT_LT(iou, 0.35f) << "Perpendicular → moderate overlap (shared center)";
}

// Partial overlap at angle
TEST(RotatedIoU, PartialOverlapAtAngle) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.3f, 0.15f, 0.65f, 0.55f};
    DarknetBoxBDP b = {0.52f, 0.48f, 0.25f, 0.12f, 0.62f, 0.52f};
    float iou = box_riou(a, b);
    EXPECT_GT(iou, 0.2f) << "Partial overlap significant";
    EXPECT_LT(iou, 0.8f) << "Not near-perfect";
    EXPECT_TRUE(std::isfinite(iou)) << "Finite";
}

// 180° rotation → same box (front edge symmetry)
TEST(RotatedIoU, Rotated180Degrees) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};  // fp right
    DarknetBoxBDP b = {0.5f, 0.5f, 0.3f, 0.2f, 0.35f, 0.5f};  // fp left (180°)
    EXPECT_NEAR(box_riou(a, b), 1.0f, 1e-3f) << "180° rotation → identical box";
}

// RIoU(a,b) = RIoU(b,a)
TEST(RotatedIoU, Symmetry) {
    DarknetBoxBDP a = {0.4f, 0.5f, 0.25f, 0.15f, 0.52f, 0.54f};
    DarknetBoxBDP b = {0.6f, 0.5f, 0.20f, 0.12f, 0.68f, 0.48f};
    EXPECT_NEAR(box_riou(a, b), box_riou(b, a), 1e-5f) << "Symmetric";
}

// Numerical stability for tiny boxes
TEST(RotatedIoU, VerySmallBoxes) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.005f, 0.005f, 0.5025f, 0.5025f};
    EXPECT_NEAR(box_riou(a, a), 1.0f, 1e-3f) << "Tiny identical → RIoU=1.0";
    EXPECT_TRUE(std::isfinite(box_riou(a, a))) << "Tiny → finite";
}

// Small box fully inside large box
TEST(RotatedIoU, FullContainmentRotated) {
    DarknetBoxBDP large = {0.5f, 0.5f, 0.6f, 0.6f, 0.8f, 0.5f};
    DarknetBoxBDP small = {0.5f, 0.5f, 0.1f, 0.1f, 0.55f, 0.55f};  // Rotated small
    float iou = box_riou(large, small);
    EXPECT_GT(iou, 0.0f) << "Contained → RIoU>0";
    EXPECT_LT(iou, 0.05f) << "Small/large ratio → low RIoU";
}

// Gradual rotation continuity
TEST(RotatedIoU, GradualRotation) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};      // 0°
    DarknetBoxBDP b = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.513f};    // ~5°
    float iou = box_riou(a, b);
    EXPECT_GT(iou, 0.90f) << "Small rotation → high overlap";
    EXPECT_LT(iou, 1.0f) << "Non-zero rotation → RIoU<1";
}

// Corner-only intersection
TEST(RotatedIoU, CornerIntersection) {
    DarknetBoxBDP a = {0.4f, 0.4f, 0.25f, 0.25f, 0.525f, 0.4f};
    DarknetBoxBDP b = {0.6f, 0.6f, 0.25f, 0.25f, 0.725f, 0.6f};
    float iou = box_riou(a, b);
    EXPECT_GE(iou, 0.0f) << "RIoU ≥ 0";
    EXPECT_LT(iou, 0.15f) << "Corner-only → small RIoU";
}

// Thin rotated crossing thick box
TEST(RotatedIoU, ThinRotatedCrossing) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.4f, 0.05f, 0.7f, 0.5f};    // Thin horizontal
    DarknetBoxBDP b = {0.5f, 0.5f, 0.3f, 0.3f, 0.62f, 0.62f};   // Thick 45°
    float iou = box_riou(a, b);
    EXPECT_GT(iou, 0.0f) << "Crossing → overlap";
    EXPECT_LT(iou, 0.3f) << "Thin/thick → limited overlap";
    EXPECT_TRUE(std::isfinite(iou)) << "Finite";
}

// Same dimensions, different rotation at center
TEST(RotatedIoU, SameDimensionsDifferentRotation) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.3f, 0.15f, 0.65f, 0.5f};    // 0°
    DarknetBoxBDP b = {0.5f, 0.5f, 0.3f, 0.15f, 0.61f, 0.58f};   // ~30°
    float iou = box_riou(a, b);
    EXPECT_GT(iou, 0.5f) << "Same center/size → significant overlap";
    EXPECT_LT(iou, 0.95f) << "Different rotation → reduced overlap";
}

// Translation test
TEST(RotatedIoU, TranslatedBox) {
    DarknetBoxBDP a = {0.3f, 0.3f, 0.2f, 0.2f, 0.4f, 0.3f};
    DarknetBoxBDP b = {0.35f, 0.35f, 0.2f, 0.2f, 0.45f, 0.35f};  // Diagonal shift
    float iou = box_riou(a, b);
    EXPECT_GT(iou, 0.3f) << "Slight translation → still overlap";
    EXPECT_LT(iou, 1.0f) << "Translated → not identical";
}

// Extreme aspect ratio
TEST(RotatedIoU, ExtremeAspectRatio) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.6f, 0.02f, 0.8f, 0.5f};  // Very wide/thin
    EXPECT_NEAR(box_riou(a, a), 1.0f, 1e-3f) << "Extreme aspect → still works";
}

// Batch validation test
TEST(RotatedIoU, RandomBoxesValid) {
    std::vector<DarknetBoxBDP> boxes = {
        {0.25f, 0.25f, 0.15f, 0.10f, 0.32f, 0.26f},
        {0.75f, 0.75f, 0.20f, 0.15f, 0.85f, 0.75f},
        {0.50f, 0.50f, 0.30f, 0.25f, 0.65f, 0.60f},
        {0.60f, 0.40f, 0.18f, 0.12f, 0.69f, 0.44f},
    };

    for (size_t i = 0; i < boxes.size(); i++) {
        for (size_t j = 0; j < boxes.size(); j++) {
            float iou = box_riou(boxes[i], boxes[j]);
            EXPECT_GE(iou, 0.0f) << "RIoU ≥ 0 for boxes " << i << "," << j;
            EXPECT_LE(iou, 1.0f) << "RIoU ≤ 1 for boxes " << i << "," << j;
            EXPECT_TRUE(std::isfinite(iou)) << "Finite for boxes " << i << "," << j;
            if (i == j) {
                EXPECT_NEAR(iou, 1.0f, 1e-3f) << "Self → RIoU=1";
            }
        }
    }
}

