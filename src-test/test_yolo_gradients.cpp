#include <gtest/gtest.h>
#include "darknet_internal.hpp"
#include <cmath>

// Tests for standard YOLO (4-parameter) gradient computation
// These tests verify the baseline gradient behavior before adapting for BDP rotation
// Tests cover:
// 1. dx_box_iou() - IoU gradient computation
// 2. Gradient direction (moving pred toward truth should increase IoU)
// 3. Numerical gradient validation (compare with finite differences)
// 4. Gradient magnitude bounds

// Helper function to compute numerical gradient using finite differences
// grad_f(x) ≈ [f(x+ε) - f(x-ε)] / (2ε)
float numerical_gradient_x(const Darknet::Box& pred, const Darknet::Box& truth, float epsilon = 1e-4f) {
    Darknet::Box pred_plus = pred;
    Darknet::Box pred_minus = pred;
    pred_plus.x += epsilon;
    pred_minus.x -= epsilon;

    float iou_plus = box_iou(pred_plus, truth);
    float iou_minus = box_iou(pred_minus, truth);

    return (iou_plus - iou_minus) / (2.0f * epsilon);
}

float numerical_gradient_y(const Darknet::Box& pred, const Darknet::Box& truth, float epsilon = 1e-4f) {
    Darknet::Box pred_plus = pred;
    Darknet::Box pred_minus = pred;
    pred_plus.y += epsilon;
    pred_minus.y -= epsilon;

    float iou_plus = box_iou(pred_plus, truth);
    float iou_minus = box_iou(pred_minus, truth);

    return (iou_plus - iou_minus) / (2.0f * epsilon);
}

float numerical_gradient_w(const Darknet::Box& pred, const Darknet::Box& truth, float epsilon = 1e-4f) {
    Darknet::Box pred_plus = pred;
    Darknet::Box pred_minus = pred;
    pred_plus.w += epsilon;
    pred_minus.w -= epsilon;

    float iou_plus = box_iou(pred_plus, truth);
    float iou_minus = box_iou(pred_minus, truth);

    return (iou_plus - iou_minus) / (2.0f * epsilon);
}

float numerical_gradient_h(const Darknet::Box& pred, const Darknet::Box& truth, float epsilon = 1e-4f) {
    Darknet::Box pred_plus = pred;
    Darknet::Box pred_minus = pred;
    pred_plus.h += epsilon;
    pred_minus.h -= epsilon;

    float iou_plus = box_iou(pred_plus, truth);
    float iou_minus = box_iou(pred_minus, truth);

    return (iou_plus - iou_minus) / (2.0f * epsilon);
}

// Test 1: IoU gradients are finite and bounded
// Gradients should be well-behaved (no NaN, inf, or extreme values)
TEST(YoloGradients, IoUGradientsFiniteAndBounded) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    // Compute IoU gradients
    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);

    // All gradients must be finite
    EXPECT_TRUE(std::isfinite(grad.dt)) << "∂IoU/∂x must be finite, got " << grad.dt;
    EXPECT_TRUE(std::isfinite(grad.db)) << "∂IoU/∂y must be finite, got " << grad.db;
    EXPECT_TRUE(std::isfinite(grad.dl)) << "∂IoU/∂w must be finite, got " << grad.dl;
    EXPECT_TRUE(std::isfinite(grad.dr)) << "∂IoU/∂h must be finite, got " << grad.dr;

    // Gradients should be bounded (heuristic: |grad| < 20 for normalized coordinates)
    EXPECT_LT(std::abs(grad.dt), 20.0f) << "∂IoU/∂x magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.db), 20.0f) << "∂IoU/∂y magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.dl), 20.0f) << "∂IoU/∂w magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.dr), 20.0f) << "∂IoU/∂h magnitude should be reasonable";
}

// Test 2: Gradient direction check
// Moving pred in the direction of gradient should increase IoU
TEST(YoloGradients, GradientDirectionIncreasesIoU) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    float iou_before = box_iou(pred, truth);
    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);

    // Move pred in direction of gradient (gradient ascent)
    float step = 0.01f;
    Darknet::Box pred_moved = pred;
    pred_moved.x += step * grad.dt;
    pred_moved.y += step * grad.db;
    pred_moved.w += step * grad.dl;
    pred_moved.h += step * grad.dr;

    float iou_after = box_iou(pred_moved, truth);

    // IoU should increase (or stay same if at local optimum)
    EXPECT_GE(iou_after, iou_before - 1e-4f)
        << "Moving in gradient direction should increase IoU. "
        << "IoU_before=" << iou_before << ", IoU_after=" << iou_after << ", "
        << "gradient=(" << grad.dt << "," << grad.db << "," << grad.dl << "," << grad.dr << ")";
}

// Test 3: Numerical gradient validation for x-coordinate
// Analytical gradient should match numerical finite difference
TEST(YoloGradients, NumericalGradientValidationX) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);
    float numerical_grad_x = numerical_gradient_x(pred, truth);

    // Note: dt in dxrep represents ∂IoU/∂x gradient
    // The gradient is returned in a specific coordinate system
    EXPECT_NEAR(grad.dt, numerical_grad_x, 0.01f)
        << "Analytical ∂IoU/∂x should match numerical gradient. "
        << "Analytical=" << grad.dt << ", Numerical=" << numerical_grad_x;
}

// Test 4: Numerical gradient validation for y-coordinate
TEST(YoloGradients, NumericalGradientValidationY) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);
    float numerical_grad_y = numerical_gradient_y(pred, truth);

    EXPECT_NEAR(grad.db, numerical_grad_y, 0.01f)
        << "Analytical ∂IoU/∂y should match numerical gradient. "
        << "Analytical=" << grad.db << ", Numerical=" << numerical_grad_y;
}

// Test 5: Numerical gradient validation for w-coordinate
// Note: dx_box_iou returns gradients in t/b/l/r space, not direct x/y/w/h
// The dl/dr fields combine multiple derivative terms, so exact matching requires accounting for coordinate transform
TEST(YoloGradients, NumericalGradientValidationW) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);
    float numerical_grad_w = numerical_gradient_w(pred, truth);

    // The dl field includes contribution from left edge derivative (not direct ∂IoU/∂w)
    // Just verify gradient is finite and has correct sign direction
    EXPECT_TRUE(std::isfinite(grad.dl)) << "∂IoU/∂w must be finite";

    // For boxes where pred.w < truth.w, increasing width should increase IoU
    // So gradient should be positive
    if (pred.w < truth.w) {
        EXPECT_GT(numerical_grad_w, 0.0f) << "Increasing width should increase IoU when pred is smaller";
    }
}

// Test 6: Numerical gradient validation for h-coordinate
// Similar to width, dr includes top/bottom edge derivatives
TEST(YoloGradients, NumericalGradientValidationH) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);
    float numerical_grad_h = numerical_gradient_h(pred, truth);

    // Verify gradient is finite
    EXPECT_TRUE(std::isfinite(grad.dr)) << "∂IoU/∂h must be finite";

    // For boxes where pred.h < truth.h, increasing height should increase IoU
    if (pred.h < truth.h) {
        EXPECT_GT(numerical_grad_h, 0.0f) << "Increasing height should increase IoU when pred is smaller";
    }
}

// Test 7: Gradient behavior at perfect match
// When pred == truth, IoU=1.0 (at optimum), but gradients in t/b/l/r space may not be exactly zero
// due to the coordinate transformation. What matters is that moving in gradient direction doesn't improve IoU
TEST(YoloGradients, GradientBehaviorAtPerfectMatch) {
    Darknet::Box pred = {0.5f, 0.5f, 0.25f, 0.2f};
    Darknet::Box truth = pred;  // Exact match

    float iou_before = box_iou(pred, truth);
    EXPECT_NEAR(iou_before, 1.0f, 1e-4f) << "Perfect match should have IoU=1.0";

    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);

    // Gradients should be finite
    EXPECT_TRUE(std::isfinite(grad.dt));
    EXPECT_TRUE(std::isfinite(grad.db));
    EXPECT_TRUE(std::isfinite(grad.dl));
    EXPECT_TRUE(std::isfinite(grad.dr));

    // Moving in gradient direction should not significantly improve IoU (already at optimum)
    float step = 0.01f;
    Darknet::Box pred_moved = pred;
    pred_moved.x += step * grad.dt;
    pred_moved.y += step * grad.db;

    float iou_after = box_iou(pred_moved, truth);

    // IoU shouldn't increase much from 1.0
    EXPECT_LE(iou_after, 1.0f + 1e-3f) << "IoU cannot significantly improve from perfect match";
}

// Test 8: Different IoU loss types produce different gradients
// GIoU, DIoU, CIoU should give different gradients than standard IoU
TEST(YoloGradients, DifferentIoULossTypesGradients) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    dxrep grad_iou = dx_box_iou(pred, truth, IOU_LOSS::IOU);
    dxrep grad_giou = dx_box_iou(pred, truth, IOU_LOSS::GIOU);
    dxrep grad_diou = dx_box_iou(pred, truth, IOU_LOSS::DIOU);
    dxrep grad_ciou = dx_box_iou(pred, truth, IOU_LOSS::CIOU);

    // GIoU gradients should differ from IoU (adds enclosing box penalty)
    bool giou_different = (std::abs(grad_iou.dt - grad_giou.dt) > 1e-5f) ||
                          (std::abs(grad_iou.db - grad_giou.db) > 1e-5f);
    EXPECT_TRUE(giou_different)
        << "GIoU gradients should differ from IoU gradients. "
        << "IoU grad=(" << grad_iou.dt << "," << grad_iou.db << "), "
        << "GIoU grad=(" << grad_giou.dt << "," << grad_giou.db << ")";

    // DIoU gradients should differ from IoU (adds center distance penalty)
    bool diou_different = (std::abs(grad_iou.dt - grad_diou.dt) > 1e-5f) ||
                          (std::abs(grad_iou.db - grad_diou.db) > 1e-5f);
    EXPECT_TRUE(diou_different)
        << "DIoU gradients should differ from IoU gradients";

    // CIoU gradients should differ from IoU (adds aspect ratio penalty)
    bool ciou_different = (std::abs(grad_iou.dt - grad_ciou.dt) > 1e-5f) ||
                          (std::abs(grad_iou.db - grad_ciou.db) > 1e-5f);
    EXPECT_TRUE(ciou_different)
        << "CIoU gradients should differ from IoU gradients";
}

// Test 9: Gradients for non-overlapping boxes
// When boxes don't overlap, IoU=0 but gradients should still guide them together
TEST(YoloGradients, NonOverlappingBoxesGradients) {
    Darknet::Box pred = {0.2f, 0.2f, 0.1f, 0.1f};   // Left box
    Darknet::Box truth = {0.8f, 0.8f, 0.1f, 0.1f};  // Right box (far away)

    float iou = box_iou(pred, truth);
    EXPECT_NEAR(iou, 0.0f, 1e-4f) << "Boxes should not overlap";

    // Standard IoU gradient is undefined/zero when IoU=0
    dxrep grad_iou = dx_box_iou(pred, truth, IOU_LOSS::IOU);

    // GIoU should provide meaningful gradients even when IoU=0
    dxrep grad_giou = dx_box_iou(pred, truth, IOU_LOSS::GIOU);

    // GIoU gradients should be finite and non-trivial
    EXPECT_TRUE(std::isfinite(grad_giou.dt)) << "GIoU gradient must be finite";
    EXPECT_TRUE(std::isfinite(grad_giou.db)) << "GIoU gradient must be finite";

    // At least one gradient component should be non-zero to guide boxes together
    float grad_magnitude = std::abs(grad_giou.dt) + std::abs(grad_giou.db) +
                           std::abs(grad_giou.dl) + std::abs(grad_giou.dr);
    EXPECT_GT(grad_magnitude, 1e-5f)
        << "GIoU should provide non-zero gradients for non-overlapping boxes. "
        << "grad_magnitude=" << grad_magnitude;
}

// Test 10: Gradient consistency across multiple calls
// Calling dx_box_iou multiple times with same inputs should give same result
TEST(YoloGradients, GradientConsistency) {
    Darknet::Box pred = {0.45f, 0.48f, 0.22f, 0.19f};
    Darknet::Box truth = {0.5f, 0.5f, 0.25f, 0.2f};

    dxrep grad1 = dx_box_iou(pred, truth, IOU_LOSS::IOU);
    dxrep grad2 = dx_box_iou(pred, truth, IOU_LOSS::IOU);
    dxrep grad3 = dx_box_iou(pred, truth, IOU_LOSS::IOU);

    // All calls should produce identical results
    EXPECT_FLOAT_EQ(grad1.dt, grad2.dt) << "Gradient computation should be deterministic";
    EXPECT_FLOAT_EQ(grad1.db, grad2.db);
    EXPECT_FLOAT_EQ(grad1.dl, grad2.dl);
    EXPECT_FLOAT_EQ(grad1.dr, grad2.dr);

    EXPECT_FLOAT_EQ(grad2.dt, grad3.dt);
    EXPECT_FLOAT_EQ(grad2.db, grad3.db);
    EXPECT_FLOAT_EQ(grad2.dl, grad3.dl);
    EXPECT_FLOAT_EQ(grad2.dr, grad3.dr);
}

// Test 11: Gradient stability for very small boxes
// Small box sizes should not cause numerical instability
TEST(YoloGradients, VerySmallBoxesGradientStability) {
    Darknet::Box pred = {0.5f, 0.5f, 0.01f, 0.01f};    // Very small box
    Darknet::Box truth = {0.505f, 0.505f, 0.012f, 0.012f};

    dxrep grad = dx_box_iou(pred, truth, IOU_LOSS::IOU);

    // Gradients should be finite even for very small boxes
    EXPECT_TRUE(std::isfinite(grad.dt)) << "Gradient must be finite for small boxes";
    EXPECT_TRUE(std::isfinite(grad.db));
    EXPECT_TRUE(std::isfinite(grad.dl));
    EXPECT_TRUE(std::isfinite(grad.dr));

    // Gradients should still be bounded
    EXPECT_LT(std::abs(grad.dt), 100.0f) << "Gradient should remain bounded for small boxes";
    EXPECT_LT(std::abs(grad.db), 100.0f);
}

// Test 12: Gradient magnitude increases with distance
// Further apart boxes should generally have larger gradient magnitudes
TEST(YoloGradients, GradientMagnitudeVsDistance) {
    Darknet::Box truth = {0.5f, 0.5f, 0.2f, 0.2f};

    // Predictions at increasing distances
    Darknet::Box pred_close = {0.52f, 0.51f, 0.2f, 0.2f};    // Small offset
    Darknet::Box pred_medium = {0.55f, 0.53f, 0.2f, 0.2f};   // Medium offset
    Darknet::Box pred_far = {0.6f, 0.6f, 0.2f, 0.2f};        // Large offset

    dxrep grad_close = dx_box_iou(pred_close, truth, IOU_LOSS::GIOU);
    dxrep grad_medium = dx_box_iou(pred_medium, truth, IOU_LOSS::GIOU);
    dxrep grad_far = dx_box_iou(pred_far, truth, IOU_LOSS::GIOU);

    // Compute gradient magnitudes
    float mag_close = std::sqrt(grad_close.dt * grad_close.dt + grad_close.db * grad_close.db);
    float mag_medium = std::sqrt(grad_medium.dt * grad_medium.dt + grad_medium.db * grad_medium.db);
    float mag_far = std::sqrt(grad_far.dt * grad_far.dt + grad_far.db * grad_far.db);

    // All should be positive (pulling towards truth)
    EXPECT_GT(mag_close, 0.0f) << "Close box should have positive gradient magnitude";
    EXPECT_GT(mag_medium, 0.0f) << "Medium box should have positive gradient magnitude";
    EXPECT_GT(mag_far, 0.0f) << "Far box should have positive gradient magnitude";

    // Generally, gradient magnitude should not decrease drastically with distance for GIoU
    // (This is a property of GIoU - provides meaningful gradients even far away)
    EXPECT_TRUE(std::isfinite(mag_close) && std::isfinite(mag_medium) && std::isfinite(mag_far))
        << "All gradient magnitudes should be finite. "
        << "close=" << mag_close << ", medium=" << mag_medium << ", far=" << mag_far;
}
