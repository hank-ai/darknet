#include <gtest/gtest.h>
#include "darknet_internal.hpp"
#include <cmath>

// ============================================================================
// ROTATED IoU GRADIENT TESTING
// ============================================================================
//
// PURPOSE: Validate dx_box_riou() - numerical gradients for rotated IoU
//
// WHY THIS MATTERS:
// - Phase 3 replaces axis-aligned IoU with true rotated IoU (box_riou)
// - Training requires gradients w.r.t. all 6 parameters: (x, y, w, h, fx, fy)
// - Unlike axis-aligned IoU, rotated IoU depends on fx,fy (orientation)
// - Numerical gradients via finite differences provide correct gradients
//
// WHAT WE'RE TESTING:
// 1. Gradient correctness: Verify numerical gradients match finite differences
// 2. Gradient direction: Moving in gradient direction should increase IoU
// 3. All 6 parameters: Test gradients for x, y, w, h, fx, fy
// 4. Numerical stability: No NaN/inf, bounded magnitudes
// 5. Orientation sensitivity: fx,fy gradients should be non-zero (unlike axis-aligned)
//
// KEY DIFFERENCES FROM dx_box_iou_bdp():
// - dx_box_iou_bdp: Returns dxrep (4 values: x,y,w,h only)
// - dx_box_riou: Returns dxrep_bdp (6 values: x,y,w,h,fx,fy)
// - dx_box_iou_bdp: Analytic gradients (fast, complex derivation)
// - dx_box_riou: Numerical gradients via finite differences (simple, slightly slower)
//
// IMPLEMENTATION: box.cpp:1536-1617
// ============================================================================

// Helper: Compute numerical gradient via central differences
// Used to validate that dx_box_riou() implementation is correct
dxrep_bdp compute_numerical_riou_gradients(const DarknetBoxBDP& pred, const DarknetBoxBDP& truth, float epsilon = 1e-4f) {
    dxrep_bdp gradients = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Gradient w.r.t. x
    {
        DarknetBoxBDP pred_plus = pred;   pred_plus.x = pred.x + epsilon;
        DarknetBoxBDP pred_minus = pred;  pred_minus.x = pred.x - epsilon;
        float iou_plus = box_riou(pred_plus, truth);
        float iou_minus = box_riou(pred_minus, truth);
        gradients.dx = (iou_plus - iou_minus) / (2.0f * epsilon);
    }

    // Gradient w.r.t. y
    {
        DarknetBoxBDP pred_plus = pred;   pred_plus.y = pred.y + epsilon;
        DarknetBoxBDP pred_minus = pred;  pred_minus.y = pred.y - epsilon;
        float iou_plus = box_riou(pred_plus, truth);
        float iou_minus = box_riou(pred_minus, truth);
        gradients.dy = (iou_plus - iou_minus) / (2.0f * epsilon);
    }

    // Gradient w.r.t. w
    {
        DarknetBoxBDP pred_plus = pred;   pred_plus.w = pred.w + epsilon;
        DarknetBoxBDP pred_minus = pred;  pred_minus.w = std::max(epsilon, pred.w - epsilon);
        float iou_plus = box_riou(pred_plus, truth);
        float iou_minus = box_riou(pred_minus, truth);
        gradients.dw = (iou_plus - iou_minus) / (2.0f * epsilon);
    }

    // Gradient w.r.t. h
    {
        DarknetBoxBDP pred_plus = pred;   pred_plus.h = pred.h + epsilon;
        DarknetBoxBDP pred_minus = pred;  pred_minus.h = std::max(epsilon, pred.h - epsilon);
        float iou_plus = box_riou(pred_plus, truth);
        float iou_minus = box_riou(pred_minus, truth);
        gradients.dh = (iou_plus - iou_minus) / (2.0f * epsilon);
    }

    // Gradient w.r.t. fx
    {
        DarknetBoxBDP pred_plus = pred;   pred_plus.fx = pred.fx + epsilon;
        DarknetBoxBDP pred_minus = pred;  pred_minus.fx = pred.fx - epsilon;
        float iou_plus = box_riou(pred_plus, truth);
        float iou_minus = box_riou(pred_minus, truth);
        gradients.dfx = (iou_plus - iou_minus) / (2.0f * epsilon);
    }

    // Gradient w.r.t. fy
    {
        DarknetBoxBDP pred_plus = pred;   pred_plus.fy = pred.fy + epsilon;
        DarknetBoxBDP pred_minus = pred;  pred_minus.fy = pred.fy - epsilon;
        float iou_plus = box_riou(pred_plus, truth);
        float iou_minus = box_riou(pred_minus, truth);
        gradients.dfy = (iou_plus - iou_minus) / (2.0f * epsilon);
    }

    return gradients;
}

// ============================================================================
// TEST GROUP 1: BASIC GRADIENT CORRECTNESS
// ============================================================================

// Test 1: dx_box_riou() matches independent numerical computation
// WHY: Validates that the implementation is correct
// HOW: Compare dx_box_riou() output with helper function above
// EXPECTED: All 6 gradients match within numerical tolerance
TEST(RotatedIoUGradients, MatchesNumericalGradients) {
    // Partially overlapping rotated boxes
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};

    // Compute gradients both ways
    dxrep_bdp analytical = dx_box_riou(pred, truth, IOU_LOSS::IOU);
    dxrep_bdp numerical = compute_numerical_riou_gradients(pred, truth);

    // Should match (both use same finite difference method, just verifying implementation)
    float tolerance = 1e-5f;
    EXPECT_NEAR(analytical.dx, numerical.dx, tolerance)
        << "dx gradient mismatch: analytical=" << analytical.dx << ", numerical=" << numerical.dx;
    EXPECT_NEAR(analytical.dy, numerical.dy, tolerance)
        << "dy gradient mismatch: analytical=" << analytical.dy << ", numerical=" << numerical.dy;
    EXPECT_NEAR(analytical.dw, numerical.dw, tolerance)
        << "dw gradient mismatch: analytical=" << analytical.dw << ", numerical=" << numerical.dw;
    EXPECT_NEAR(analytical.dh, numerical.dh, tolerance)
        << "dh gradient mismatch: analytical=" << analytical.dh << ", numerical=" << numerical.dh;
    EXPECT_NEAR(analytical.dfx, numerical.dfx, tolerance)
        << "dfx gradient mismatch: analytical=" << analytical.dfx << ", numerical=" << numerical.dfx;
    EXPECT_NEAR(analytical.dfy, numerical.dfy, tolerance)
        << "dfy gradient mismatch: analytical=" << analytical.dfy << ", numerical=" << numerical.dfy;
}

// Test 2: All gradients are finite
// WHY: NaN/inf causes training divergence
// HOW: Test various box configurations
// EXPECTED: No NaN, no inf in any gradient component
TEST(RotatedIoUGradients, AllGradientsFinite) {
    std::vector<std::pair<DarknetBoxBDP, DarknetBoxBDP>> test_cases = {
        // Partially overlapping
        {{0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f}, {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f}},
        // Nearly identical
        {{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f}, {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f}},
        // Rotated 90 degrees
        {{0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f}, {0.5f, 0.5f, 0.3f, 0.2f, 0.5f, 0.65f}},
        // Small boxes
        {{0.5f, 0.5f, 0.05f, 0.05f, 0.525f, 0.5f}, {0.51f, 0.51f, 0.06f, 0.06f, 0.535f, 0.51f}}
    };

    for (size_t i = 0; i < test_cases.size(); i++) {
        const auto& pred = test_cases[i].first;
        const auto& truth = test_cases[i].second;

        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        EXPECT_TRUE(std::isfinite(grad.dx)) << "Test case " << i << ": dx must be finite";
        EXPECT_TRUE(std::isfinite(grad.dy)) << "Test case " << i << ": dy must be finite";
        EXPECT_TRUE(std::isfinite(grad.dw)) << "Test case " << i << ": dw must be finite";
        EXPECT_TRUE(std::isfinite(grad.dh)) << "Test case " << i << ": dh must be finite";
        EXPECT_TRUE(std::isfinite(grad.dfx)) << "Test case " << i << ": dfx must be finite";
        EXPECT_TRUE(std::isfinite(grad.dfy)) << "Test case " << i << ": dfy must be finite";
    }
}

// Test 3: Gradients are bounded (heuristic check)
// WHY: Extremely large gradients can cause training instability
// HOW: Check that |grad| < 50 for normalized [0,1] coordinates
// EXPECTED: All gradient magnitudes reasonable
TEST(RotatedIoUGradients, GradientsBounded) {
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    float max_reasonable = 50.0f;
    EXPECT_LT(std::abs(grad.dx), max_reasonable) << "dx gradient too large: " << grad.dx;
    EXPECT_LT(std::abs(grad.dy), max_reasonable) << "dy gradient too large: " << grad.dy;
    EXPECT_LT(std::abs(grad.dw), max_reasonable) << "dw gradient too large: " << grad.dw;
    EXPECT_LT(std::abs(grad.dh), max_reasonable) << "dh gradient too large: " << grad.dh;
    EXPECT_LT(std::abs(grad.dfx), max_reasonable) << "dfx gradient too large: " << grad.dfx;
    EXPECT_LT(std::abs(grad.dfy), max_reasonable) << "dfy gradient too large: " << grad.dfy;
}

// ============================================================================
// TEST GROUP 2: GRADIENT DIRECTION (ASCENT INCREASES IoU)
// ============================================================================

// Test 4: Moving in gradient direction increases rotated IoU
// WHY: Gradient ascent should improve IoU
// HOW: Take small step in gradient direction, verify IoU increases
// EXPECTED: RIoU_after >= RIoU_before
TEST(RotatedIoUGradients, GradientDirectionIncreasesIoU) {
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};

    float riou_before = box_riou(pred, truth);
    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // Gradient ascent: move in direction that increases IoU
    float step = 0.005f;  // Small step to stay in linear region
    DarknetBoxBDP pred_moved = pred;
    pred_moved.x += step * grad.dx;
    pred_moved.y += step * grad.dy;
    pred_moved.w = std::max(0.01f, pred.w + step * grad.dw);  // Keep positive
    pred_moved.h = std::max(0.01f, pred.h + step * grad.dh);  // Keep positive
    pred_moved.fx += step * grad.dfx;
    pred_moved.fy += step * grad.dfy;

    float riou_after = box_riou(pred_moved, truth);

    EXPECT_GE(riou_after, riou_before - 1e-4f)
        << "Moving in gradient direction should increase rotated IoU. "
        << "RIoU_before=" << riou_before << ", RIoU_after=" << riou_after << ". "
        << "Gradient: dx=" << grad.dx << ", dy=" << grad.dy << ", dw=" << grad.dw
        << ", dh=" << grad.dh << ", dfx=" << grad.dfx << ", dfy=" << grad.dfy;
}

// ============================================================================
// TEST GROUP 3: ORIENTATION GRADIENTS (fx, fy)
// ============================================================================

// Test 5: Orientation gradients (dfx, dfy) are non-zero for rotated boxes
// WHY: Unlike axis-aligned IoU, rotated IoU depends on orientation
// HOW: Test boxes with different orientations AND different aspect ratios
// EXPECTED: dfx and/or dfy should be non-zero (provides learning signal for rotation)
TEST(RotatedIoUGradients, OrientationGradientsNonZero) {
    // Boxes with same center but DIFFERENT aspect ratios and orientations
    // Using different aspect ratios breaks symmetry
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.3f, 0.15f, 0.65f, 0.5f};    // Horizontal, 2:1 aspect
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.3f, 0.2f, 0.56f, 0.6f};    // Rotated ~45°, 3:2 aspect

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // At least one orientation gradient should be non-zero
    // This is KEY: rotation affects IoU, so gradients must reflect this
    float orientation_grad_magnitude = std::abs(grad.dfx) + std::abs(grad.dfy);

    EXPECT_GT(orientation_grad_magnitude, 1e-4f)
        << "Orientation gradients (dfx, dfy) should be non-zero for rotated boxes. "
        << "This is the key difference from axis-aligned IoU. "
        << "dfx=" << grad.dfx << ", dfy=" << grad.dfy << ". "
        << "Without these gradients, the network cannot learn rotation!";
}

// Test 6: Orientation gradients point toward correct rotation
// WHY: Gradients should guide network to match ground truth orientation
// HOW: Adjust orientation in gradient direction, verify it moves toward truth
// EXPECTED: Angle difference decreases
TEST(RotatedIoUGradients, OrientationGradientsCorrectDirection) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};     // Horizontal
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.3f, 0.2f, 0.56f, 0.6f};    // Slightly rotated

    // Compute angle before
    float pred_vec_x_before = pred.fx - pred.x;
    float pred_vec_y_before = pred.fy - pred.y;
    float truth_vec_x = truth.fx - truth.x;
    float truth_vec_y = truth.fy - truth.y;

    float dot_before = pred_vec_x_before * truth_vec_x + pred_vec_y_before * truth_vec_y;
    float angle_before = std::acos(std::max(-1.0f, std::min(1.0f,
        dot_before / (std::sqrt(pred_vec_x_before*pred_vec_x_before + pred_vec_y_before*pred_vec_y_before) *
                      std::sqrt(truth_vec_x*truth_vec_x + truth_vec_y*truth_vec_y)))));

    // Apply gradient to orientation
    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);
    float step = 0.01f;
    DarknetBoxBDP pred_moved = pred;
    pred_moved.fx += step * grad.dfx;
    pred_moved.fy += step * grad.dfy;

    // Compute angle after
    float pred_vec_x_after = pred_moved.fx - pred_moved.x;
    float pred_vec_y_after = pred_moved.fy - pred_moved.y;

    float dot_after = pred_vec_x_after * truth_vec_x + pred_vec_y_after * truth_vec_y;
    float angle_after = std::acos(std::max(-1.0f, std::min(1.0f,
        dot_after / (std::sqrt(pred_vec_x_after*pred_vec_x_after + pred_vec_y_after*pred_vec_y_after) *
                     std::sqrt(truth_vec_x*truth_vec_x + truth_vec_y*truth_vec_y)))));

    // Angle should decrease (moving toward truth orientation)
    EXPECT_LE(angle_after, angle_before + 0.01f)  // Small tolerance for numerical error
        << "Moving in gradient direction should reduce angle difference. "
        << "angle_before=" << angle_before << ", angle_after=" << angle_after;
}

// ============================================================================
// TEST GROUP 4: EDGE CASES
// ============================================================================

// Test 7: Zero gradient at perfect match
// WHY: At optimum, gradients should be zero
// HOW: Set pred == truth
// EXPECTED: All gradients ≈ 0
TEST(RotatedIoUGradients, ZeroGradientAtPerfectMatch) {
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};
    DarknetBoxBDP pred = truth;  // Perfect match

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // Relaxed tolerance for numerical gradients (finite differences near IoU=1.0 have numerical error)
    float tolerance = 0.01f;
    EXPECT_NEAR(grad.dx, 0.0f, tolerance) << "dx should be ~0 at perfect match";
    EXPECT_NEAR(grad.dy, 0.0f, tolerance) << "dy should be ~0 at perfect match";
    EXPECT_NEAR(grad.dw, 0.0f, tolerance) << "dw should be ~0 at perfect match";
    EXPECT_NEAR(grad.dh, 0.0f, tolerance) << "dh should be ~0 at perfect match";
    EXPECT_NEAR(grad.dfx, 0.0f, tolerance) << "dfx should be ~0 at perfect match";
    EXPECT_NEAR(grad.dfy, 0.0f, tolerance) << "dfy should be ~0 at perfect match";
}

// Test 8: Gradients stable for very small boxes
// WHY: Small boxes can cause numerical issues
// HOW: Test with boxes of size 0.01 x 0.01
// EXPECTED: All gradients finite and bounded
TEST(RotatedIoUGradients, SmallBoxStability) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.01f, 0.01f, 0.505f, 0.5f};
    DarknetBoxBDP truth = {0.505f, 0.505f, 0.012f, 0.012f, 0.51f, 0.505f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite for small boxes";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite for small boxes";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite for small boxes";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite for small boxes";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite for small boxes";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite for small boxes";

    // Should still be bounded
    EXPECT_LT(std::abs(grad.dx), 100.0f) << "Gradients should not explode for small boxes";
}

// Test 9: Gradients consistent across multiple calls
// WHY: Gradient computation should be deterministic
// HOW: Call dx_box_riou multiple times
// EXPECTED: Identical results
TEST(RotatedIoUGradients, DeterministicComputation) {
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};

    dxrep_bdp grad1 = dx_box_riou(pred, truth, IOU_LOSS::IOU);
    dxrep_bdp grad2 = dx_box_riou(pred, truth, IOU_LOSS::IOU);
    dxrep_bdp grad3 = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    EXPECT_FLOAT_EQ(grad1.dx, grad2.dx) << "Gradient computation should be deterministic";
    EXPECT_FLOAT_EQ(grad1.dy, grad2.dy);
    EXPECT_FLOAT_EQ(grad1.dw, grad2.dw);
    EXPECT_FLOAT_EQ(grad1.dh, grad2.dh);
    EXPECT_FLOAT_EQ(grad1.dfx, grad2.dfx);
    EXPECT_FLOAT_EQ(grad1.dfy, grad2.dfy);

    EXPECT_FLOAT_EQ(grad2.dx, grad3.dx);
    EXPECT_FLOAT_EQ(grad2.dy, grad3.dy);
}
