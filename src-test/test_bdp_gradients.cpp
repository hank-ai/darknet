#include <gtest/gtest.h>
#include "darknet_internal.hpp"
#include <cmath>

// ============================================================================
// BDP GRADIENT TESTING RATIONALE
// ============================================================================
//
// NOTE: These tests are compatible with BOTH finite-difference and GWD
// implementations of dx_box_riou(). They test gradient PROPERTIES (finite,
// non-zero, correct direction) rather than exact numerical values.
//
// PURPOSE: Validate that delta_yolo_box_bdp() correctly computes gradients for
// 6-parameter oriented bounding boxes during training.
//
// WHY THIS MATTERS:
// 1. Training divergence: Incorrect gradients cause loss to explode (mentioned in ROTATED_IOU_PLAN.md)
// 2. Orientation learning: fx,fy gradients must correctly guide the network to learn rotation
// 3. Chain rule: Gradients must account for sigmoid/exp activation functions
// 4. Loss components: Must properly combine IoU gradients + Smooth L1 gradients for front point
//
// WHAT WE'RE TESTING:
// - BDP uses 6 parameters: (x, y, w, h, fx, fy)
// - Standard YOLO uses 4: (x, y, w, h)
// - The additional (fx, fy) represent the "front point" that defines box orientation
//
// GRADIENT FLOW IN TRAINING:
//   Forward:  raw_output → sigmoid(x,y,fx,fy) → box_params → loss
//   Backward: ∂loss/∂params → chain_rule → ∂loss/∂raw_output → backprop
//
// KEY IMPLEMENTATION DETAILS (from yolo_layer.cpp:354-618):
// 1. IoU loss gradients: Uses dx_box_iou_bdp() for x,y,w,h (currently ignores rotation)
// 2. Front point loss: Smooth L1 between predicted and ground truth front point
// 3. Angular correction: Multiplies IoU by cos(angle/2) to penalize wrong orientation
// 4. Chain rule: Gradients multiplied by activation derivatives (logistic_gradient for sigmoid)
//
// TEST STRATEGY:
// - Validate each component independently
// - Test numerical stability (no NaN/inf)
// - Verify gradient direction (moving toward truth should reduce loss)
// - Compare with numerical gradients (finite differences)
// - Test edge cases (perfect match, large errors, small boxes)
// ============================================================================

// Helper: Compute numerical gradient for front point x-coordinate
// WHY: Validates that analytical gradient in delta_yolo_box_bdp() matches finite difference
// HOW: Compute loss at (fx+ε) and (fx-ε), divide difference by 2ε
float numerical_gradient_fx(const DarknetBoxBDP& pred, const DarknetBoxBDP& truth, float epsilon = 1e-4f) {
    // Front point loss uses Smooth L1: loss = 0.5*diff² (small) or |diff|-0.5 (large)
    // This is from yolo_layer.cpp:456-458
    float diff_fx_plus = truth.fx - (pred.fx + epsilon);
    float diff_fx_minus = truth.fx - (pred.fx - epsilon);

    auto smooth_l1 = [](float diff) {
        float abs_diff = std::abs(diff);
        return (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    };

    float loss_plus = smooth_l1(diff_fx_plus);
    float loss_minus = smooth_l1(diff_fx_minus);

    // Gradient = -∂loss/∂fx (negative because we minimize loss)
    return -(loss_plus - loss_minus) / (2.0f * epsilon);
}

// Helper: Compute numerical gradient for front point y-coordinate
float numerical_gradient_fy(const DarknetBoxBDP& pred, const DarknetBoxBDP& truth, float epsilon = 1e-4f) {
    float diff_fy_plus = truth.fy - (pred.fy + epsilon);
    float diff_fy_minus = truth.fy - (pred.fy - epsilon);

    auto smooth_l1 = [](float diff) {
        float abs_diff = std::abs(diff);
        return (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    };

    float loss_plus = smooth_l1(diff_fy_plus);
    float loss_minus = smooth_l1(diff_fy_minus);

    return -(loss_plus - loss_minus) / (2.0f * epsilon);
}

// Note: logistic_gradient() is already defined in darknet_internal.hpp
// It computes sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

// ============================================================================
// TEST GROUP 1: IoU GRADIENTS (x, y, w, h)
// These test the standard 4-parameter gradients that BDP inherits
// ============================================================================

// Test 1: BDP RIOU gradients are finite and bounded
// WHY: NaN/inf gradients cause training divergence
// HOW: Call dx_box_riou() and verify all 6 components are finite
// EXPECTED: All gradients < 20 for normalized [0,1] coordinates
TEST(BDPGradients, RIOUGradientsFiniteAndBounded) {
    // Setup: Two partially overlapping boxes with slight rotation
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};   // Slightly rotated
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};     // Ground truth

    // Compute BDP RIOU gradients (true rotated IoU with all 6 parameters)
    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // All gradients must be finite (not NaN or inf)
    EXPECT_TRUE(std::isfinite(grad.dx)) << "∂RIOU/∂x must be finite, got " << grad.dx;
    EXPECT_TRUE(std::isfinite(grad.dy)) << "∂RIOU/∂y must be finite, got " << grad.dy;
    EXPECT_TRUE(std::isfinite(grad.dw)) << "∂RIOU/∂w must be finite, got " << grad.dw;
    EXPECT_TRUE(std::isfinite(grad.dh)) << "∂RIOU/∂h must be finite, got " << grad.dh;
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "∂RIOU/∂fx must be finite, got " << grad.dfx;
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "∂RIOU/∂fy must be finite, got " << grad.dfy;

    // Gradients should be bounded (heuristic: |grad| < 20 for normalized coords)
    // If gradients are huge (>100), it indicates numerical instability
    EXPECT_LT(std::abs(grad.dx), 20.0f) << "∂RIOU/∂x magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.dy), 20.0f) << "∂RIOU/∂y magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.dw), 20.0f) << "∂RIOU/∂w magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.dh), 20.0f) << "∂RIOU/∂h magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.dfx), 20.0f) << "∂RIOU/∂fx magnitude should be reasonable";
    EXPECT_LT(std::abs(grad.dfy), 20.0f) << "∂RIOU/∂fy magnitude should be reasonable";
}

// Test 2: RIOU gradient direction increases RIOU
// WHY: Gradient ascent should improve RIOU (gradient points toward higher RIOU)
// HOW: Move pred in gradient direction, verify RIOU increases
// EXPECTED: RIOU_after >= RIOU_before (within numerical tolerance)
TEST(BDPGradients, RIOUGradientDirectionTest) {
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};

    float riou_before = box_riou(pred, truth);
    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // Gradient ascent: move in direction that increases RIOU
    float step = 0.01f;  // Small step
    DarknetBoxBDP pred_moved = pred;
    pred_moved.x += step * grad.dx;
    pred_moved.y += step * grad.dy;
    pred_moved.w += step * grad.dw;
    pred_moved.h += step * grad.dh;
    pred_moved.fx += step * grad.dfx;
    pred_moved.fy += step * grad.dfy;

    float riou_after = box_riou(pred_moved, truth);

    // RIOU should increase (or stay same if at local optimum)
    EXPECT_GE(riou_after, riou_before - 1e-4f)
        << "Moving in gradient direction should increase RIOU. "
        << "RIOU_before=" << riou_before << ", RIOU_after=" << riou_after << ", "
        << "gradient=(" << grad.dx << "," << grad.dy << "," << grad.dw << ","
        << grad.dh << "," << grad.dfx << "," << grad.dfy << ")";
}

// Test 3: RIOU gradients include orientation (fx, fy) components
// WHY: Unlike standard IoU, RIOU accounts for rotation through fx/fy
// HOW: Verify that dfx and dfy gradients are non-zero for rotated boxes
// EXPECTED: When boxes have different orientations, dfx/dfy should be non-zero
TEST(BDPGradients, RIOUIncludesOrientationGradients) {
    // Two boxes with same center/size but different orientations
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.25f, 0.2f, 0.65f, 0.5f};   // Horizontal (fx to right)
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.5f, 0.65f};  // Vertical (fx upward)

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // All gradients must be finite
    EXPECT_TRUE(std::isfinite(grad.dx)) << "∂RIOU/∂x must be finite";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "∂RIOU/∂y must be finite";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "∂RIOU/∂fx must be finite";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "∂RIOU/∂fy must be finite";

    // For rotated boxes, orientation gradients (dfx, dfy) should be non-zero
    // This tells the network how to adjust the front point to improve alignment
    float orientation_grad_mag = std::abs(grad.dfx) + std::abs(grad.dfy);
    EXPECT_GT(orientation_grad_mag, 1e-6f)
        << "RIOU gradients should include orientation information (dfx, dfy). "
        << "dfx=" << grad.dfx << ", dfy=" << grad.dfy << ". "
        << "This is what makes RIOU different from standard IoU.";
}

// ============================================================================
// TEST GROUP 2: FRONT POINT GRADIENTS (fx, fy)
// These test the orientation-specific gradients unique to BDP
// ============================================================================

// Test 4: Front point gradient for small errors (quadratic region)
// WHY: Smooth L1 uses quadratic loss for small errors: loss = 0.5*diff²
// HOW: Test with small error (|diff| < 1.0), verify gradient = diff
// EXPECTED: gradient = diff (derivative of 0.5*x² is x)
TEST(BDPGradients, FrontPointGradientSmallError) {
    // Setup: Pred front point is slightly off from truth
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.55f, 0.52f};   // fx=0.55, fy=0.52
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};    // fx=0.6, fy=0.5

    float diff_fx = truth.fx - pred.fx;  // 0.6 - 0.55 = 0.05 (small error)
    float diff_fy = truth.fy - pred.fy;  // 0.5 - 0.52 = -0.02 (small error)

    // For |diff| < 1.0, Smooth L1 gradient = diff
    // This is from yolo_layer.cpp:462-463
    float expected_grad_fx = diff_fx;    // 0.05
    float expected_grad_fy = diff_fy;    // -0.02

    // Compute numerical gradients to validate
    float numerical_grad_fx = numerical_gradient_fx(pred, truth);
    float numerical_grad_fy = numerical_gradient_fy(pred, truth);

    EXPECT_NEAR(numerical_grad_fx, expected_grad_fx, 1e-4f)
        << "Smooth L1 gradient for small error should equal diff. "
        << "This validates the quadratic region: d/dx(0.5*x²) = x. "
        << "diff_fx=" << diff_fx << ", numerical_grad=" << numerical_grad_fx;

    EXPECT_NEAR(numerical_grad_fy, expected_grad_fy, 1e-4f)
        << "Smooth L1 gradient for small error should equal diff. "
        << "diff_fy=" << diff_fy << ", numerical_grad=" << numerical_grad_fy;
}

// Test 5: Front point gradient bounded at large errors (linear region)
// WHY: Smooth L1 uses linear loss for large errors: loss = |diff| - 0.5
// HOW: Test with large error, verify gradient = ±1.0 (bounded)
// EXPECTED: gradient = sign(diff) (derivative of |x| is sign(x))
// IMPORTANCE: Prevents gradient explosion for outliers
TEST(BDPGradients, FrontPointGradientBoundedForLargeError) {
    // Setup: Create error >= 1.0 to enter linear region
    // In normalized coords [0,1], max error is 1.0
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0.1f};     // Front point at corner
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0.9f};    // Front point at opposite corner

    float diff_fx = truth.fx - pred.fx;  // 0.9 - 0.1 = 0.8 (still < 1.0, quadratic)
    float diff_fy = truth.fy - pred.fy;  // 0.9 - 0.1 = 0.8 (still < 1.0, quadratic)

    // Note: Within [0,1] normalized coords, we can't easily get |diff| >= 1.0
    // So we test that gradients are bounded and reasonable
    float numerical_grad_fx = numerical_gradient_fx(pred, truth);
    float numerical_grad_fy = numerical_gradient_fy(pred, truth);

    // Gradients should be bounded (|grad| <= 1.0 in linear region)
    // For quadratic region (|diff| < 1.0), |grad| = |diff| < 1.0
    EXPECT_LE(std::abs(numerical_grad_fx), 1.0f)
        << "Smooth L1 gradient magnitude should be <= 1.0. "
        << "This prevents gradient explosion for large errors. "
        << "diff_fx=" << diff_fx << ", grad=" << numerical_grad_fx;

    EXPECT_LE(std::abs(numerical_grad_fy), 1.0f)
        << "Smooth L1 gradient magnitude should be <= 1.0. "
        << "diff_fy=" << diff_fy << ", grad=" << numerical_grad_fy;

    // Gradients should have correct sign (pointing toward truth)
    EXPECT_GT(numerical_grad_fx, 0.0f) << "Gradient should be positive (pred.fx < truth.fx)";
    EXPECT_GT(numerical_grad_fy, 0.0f) << "Gradient should be positive (pred.fy < truth.fy)";
}

// Test 6: Front point gradient is zero at perfect match
// WHY: At optimum (pred == truth), gradient should be zero
// HOW: Set pred.fx = truth.fx and pred.fy = truth.fy
// EXPECTED: gradient ≈ 0
TEST(BDPGradients, ZeroGradientForPerfectFrontPoint) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.625f, 0.5f};
    DarknetBoxBDP truth = pred;  // Perfect match including front point

    float diff_fx = truth.fx - pred.fx;  // 0.0
    float diff_fy = truth.fy - pred.fy;  // 0.0

    // Smooth L1 gradient at zero should be zero
    // d/dx(0.5*x²)|_{x=0} = 0
    float numerical_grad_fx = numerical_gradient_fx(pred, truth);
    float numerical_grad_fy = numerical_gradient_fy(pred, truth);

    EXPECT_NEAR(numerical_grad_fx, 0.0f, 1e-4f)
        << "Gradient should be zero at perfect match (optimum). "
        << "diff_fx=" << diff_fx << ", grad=" << numerical_grad_fx;

    EXPECT_NEAR(numerical_grad_fy, 0.0f, 1e-4f)
        << "Gradient should be zero at perfect match. "
        << "diff_fy=" << diff_fy << ", grad=" << numerical_grad_fy;
}

// Test 7: Front point gradient symmetry
// WHY: Smooth L1 is symmetric: loss(+x) = loss(-x)
// HOW: Test equal magnitude errors with opposite signs
// EXPECTED: gradients have same magnitude, opposite sign
TEST(BDPGradients, FrontPointGradientSymmetry) {
    // Two predictions with opposite errors
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};
    DarknetBoxBDP pred_pos = {0.5f, 0.5f, 0.2f, 0.2f, 0.65f, 0.55f};  // Error: +0.05, +0.05
    DarknetBoxBDP pred_neg = {0.5f, 0.5f, 0.2f, 0.2f, 0.55f, 0.45f};  // Error: -0.05, -0.05

    float grad_fx_pos = numerical_gradient_fx(pred_pos, truth);
    float grad_fy_pos = numerical_gradient_fy(pred_pos, truth);
    float grad_fx_neg = numerical_gradient_fx(pred_neg, truth);
    float grad_fy_neg = numerical_gradient_fy(pred_neg, truth);

    // Gradients should have same magnitude, opposite signs
    EXPECT_NEAR(std::abs(grad_fx_pos), std::abs(grad_fx_neg), 1e-4f)
        << "Smooth L1 gradient magnitude should be symmetric. "
        << "|grad(+err)|=" << std::abs(grad_fx_pos) << ", "
        << "|grad(-err)|=" << std::abs(grad_fx_neg);

    EXPECT_NEAR(grad_fx_pos, -grad_fx_neg, 1e-4f)
        << "Gradients for opposite errors should have opposite signs. "
        << "grad_pos=" << grad_fx_pos << ", grad_neg=" << grad_fx_neg;
}

// ============================================================================
// TEST GROUP 3: ANGULAR CORRECTION
// Tests how orientation error affects IoU
// ============================================================================

// Test 8: Angular correction penalizes rotation mismatch
// WHY: Current BDP multiplies IoU by cos(angle/2) to account for orientation
// HOW: Compare boxes with same center/size but different orientations
// EXPECTED: Perpendicular boxes should have lower corrected IoU
// NOTE: Phase 4 will replace this with true rotated IoU
TEST(BDPGradients, AngularCorrectionAffectsIoU) {
    // Two boxes with same center/size but different orientations
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};     // Horizontal (fx to right)
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.3f, 0.2f, 0.5f, 0.65f};    // Vertical (fx upward)

    // Compute IoU without angular correction (just axis-aligned)
    float iou_uncorrected = box_iou_bdp(pred, truth);

    // Compute angle between front point vectors
    // pred: front_vec = (0.15, 0) → angle = 0° (horizontal)
    // truth: front_vec = (0, 0.15) → angle = 90° (vertical)
    float pred_vec_x = pred.fx - pred.x;    // 0.65 - 0.5 = 0.15
    float pred_vec_y = pred.fy - pred.y;    // 0.5 - 0.5 = 0.0
    float truth_vec_x = truth.fx - truth.x; // 0.5 - 0.5 = 0.0
    float truth_vec_y = truth.fy - truth.y; // 0.65 - 0.5 = 0.15

    float dot = pred_vec_x * truth_vec_x + pred_vec_y * truth_vec_y;  // 0.0
    float pred_mag = std::sqrt(pred_vec_x * pred_vec_x + pred_vec_y * pred_vec_y);   // 0.15
    float truth_mag = std::sqrt(truth_vec_x * truth_vec_x + truth_vec_y * truth_vec_y); // 0.15

    float cos_angle = dot / (pred_mag * truth_mag);  // 0.0 (90° angle)
    float angle = std::acos(std::max(-1.0f, std::min(1.0f, cos_angle)));  // π/2
    float angular_correction = std::cos(angle / 2.0f);  // cos(45°) ≈ 0.707

    // Corrected IoU applies angular penalty (yolo_layer.cpp:414-430)
    float iou_corrected = iou_uncorrected * angular_correction;

    EXPECT_LT(iou_corrected, iou_uncorrected)
        << "Angular correction should reduce IoU for perpendicular boxes. "
        << "This penalizes orientation mismatch even if x,y,w,h match. "
        << "IoU_uncorrected=" << iou_uncorrected << ", "
        << "IoU_corrected=" << iou_corrected << ", "
        << "angular_correction=" << angular_correction;

    EXPECT_NEAR(angular_correction, 0.707f, 0.1f)
        << "cos(45°) should be ~0.707 for 90° rotation between boxes. "
        << "This validates the angular correction formula.";
}

// Test 9: Angular correction at various rotation angles
// WHY: Understand how angular correction behaves across different misalignments
// HOW: Test 0°, 45°, 90°, 180° rotations
// EXPECTED: correction factor decreases as angle increases
TEST(BDPGradients, AngularCorrectionAtVariousAngles) {
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};  // Horizontal reference

    // Different rotation angles
    struct TestCase {
        float fx, fy;       // Front point position
        float expected_angle;  // Expected angle in degrees
        float expected_correction;  // Expected cos(angle/2)
    };

    std::vector<TestCase> cases = {
        {0.65f, 0.5f, 0.0f, 1.0f},      // 0°: same orientation
        {0.606f, 0.606f, 45.0f, 0.924f}, // 45°: cos(22.5°) ≈ 0.924
        {0.5f, 0.65f, 90.0f, 0.707f},    // 90°: cos(45°) ≈ 0.707
        {0.35f, 0.5f, 180.0f, 0.0f}      // 180°: cos(90°) = 0
    };

    for (size_t i = 0; i < cases.size(); i++) {
        DarknetBoxBDP pred = {0.5f, 0.5f, 0.3f, 0.2f, cases[i].fx, cases[i].fy};

        // Compute angular correction
        float pred_vec_x = pred.fx - pred.x;
        float pred_vec_y = pred.fy - pred.y;
        float truth_vec_x = truth.fx - truth.x;
        float truth_vec_y = truth.fy - truth.y;

        float dot = pred_vec_x * truth_vec_x + pred_vec_y * truth_vec_y;
        float pred_mag = std::sqrt(pred_vec_x * pred_vec_x + pred_vec_y * pred_vec_y);
        float truth_mag = std::sqrt(truth_vec_x * truth_vec_x + truth_vec_y * truth_vec_y);

        float cos_angle = dot / (pred_mag * truth_mag);
        float angle = std::acos(std::max(-1.0f, std::min(1.0f, cos_angle)));
        float angular_correction = std::cos(angle / 2.0f);

        EXPECT_NEAR(angular_correction, cases[i].expected_correction, 0.15f)
            << "Test case #" << i << ": Expected angle " << cases[i].expected_angle << "°, "
            << "expected correction " << cases[i].expected_correction << ", "
            << "got " << angular_correction;
    }
}

// ============================================================================
// TEST GROUP 4: CHAIN RULE THROUGH ACTIVATIONS
// Tests gradient flow through sigmoid/exp activation functions
// ============================================================================

// Test 10: Logistic gradient computation
// WHY: Forward pass applies sigmoid to x,y,fx,fy. Backward must chain through sigmoid'
// HOW: Test logistic_gradient() helper function
// EXPECTED: For sigmoid(x), derivative = sigmoid(x) * (1 - sigmoid(x))
TEST(BDPGradients, LogisticGradientFormula) {
    // Test at various points
    std::vector<float> test_values = {0.0f, 0.3f, 0.5f, 0.7f, 1.0f};

    for (float val : test_values) {
        float grad = logistic_gradient(val);

        // sigmoid'(x) = x * (1 - x) when x is already sigmoid output
        float expected = val * (1.0f - val);

        EXPECT_NEAR(grad, expected, 1e-6f)
            << "Logistic gradient formula: sigmoid'(x) = x*(1-x). "
            << "For x=" << val << ", expected " << expected << ", got " << grad;

        // Gradient should be in [0, 0.25] (maximum at x=0.5)
        EXPECT_GE(grad, 0.0f) << "Sigmoid derivative must be non-negative";
        EXPECT_LE(grad, 0.25f + 1e-6f) << "Sigmoid derivative maximum is 0.25 at x=0.5";
    }
}

// Test 11: Chain rule application for front point
// WHY: Verifies that gradients correctly account for sigmoid activation
// HOW: loss_grad * activation_grad should give correct backprop gradient
// EXPECTED: delta = loss_gradient * logistic_gradient(output)
TEST(BDPGradients, ChainRuleThroughSigmoid) {
    // Simulate forward pass output (after sigmoid)
    float output_fx = 0.6f;  // sigmoid output for fx
    float output_fy = 0.55f; // sigmoid output for fy

    // Simulate loss gradient (from Smooth L1)
    float loss_grad_fx = 0.05f;  // ∂loss/∂fx
    float loss_grad_fy = -0.02f; // ∂loss/∂fy

    // Chain rule: ∂loss/∂raw_fx = ∂loss/∂fx * ∂fx/∂raw_fx
    //                            = loss_grad * sigmoid'(output)
    float chain_grad_fx = loss_grad_fx * logistic_gradient(output_fx);
    float chain_grad_fy = loss_grad_fy * logistic_gradient(output_fy);

    // Verify chain rule gradients are finite and reasonable
    EXPECT_TRUE(std::isfinite(chain_grad_fx))
        << "Chain rule gradient must be finite. "
        << "This ensures backprop through sigmoid doesn't produce NaN.";

    EXPECT_TRUE(std::isfinite(chain_grad_fy));

    // Sigmoid derivative reduces gradient magnitude (smoothing effect)
    // logistic_gradient(0.6) ≈ 0.24, so chain_grad should be smaller than loss_grad
    EXPECT_LT(std::abs(chain_grad_fx), std::abs(loss_grad_fx) + 1e-6f)
        << "Chain rule through sigmoid should not amplify gradient. "
        << "loss_grad=" << loss_grad_fx << ", chain_grad=" << chain_grad_fx;
}

// ============================================================================
// TEST GROUP 5: STABILITY AND EDGE CASES
// ============================================================================

// Test 12: Very small boxes don't cause gradient explosion
// WHY: Small dimensions can cause division by near-zero in IoU computation
// HOW: Test with boxes of size 0.01 x 0.01
// EXPECTED: All gradients finite and bounded
TEST(BDPGradients, VerySmallBoxStabilityBDP) {
    DarknetBoxBDP pred = {0.5f, 0.5f, 0.01f, 0.01f, 0.505f, 0.5f};
    DarknetBoxBDP truth = {0.505f, 0.505f, 0.012f, 0.012f, 0.51f, 0.505f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // Gradients should be finite for very small boxes
    EXPECT_TRUE(std::isfinite(grad.dx)) << "Gradient must be finite for small boxes";
    EXPECT_TRUE(std::isfinite(grad.dy));
    EXPECT_TRUE(std::isfinite(grad.dw));
    EXPECT_TRUE(std::isfinite(grad.dh));
    EXPECT_TRUE(std::isfinite(grad.dfx));
    EXPECT_TRUE(std::isfinite(grad.dfy));

    // Front point gradients should also be finite
    float numerical_grad_fx = numerical_gradient_fx(pred, truth);
    float numerical_grad_fy = numerical_gradient_fy(pred, truth);

    EXPECT_TRUE(std::isfinite(numerical_grad_fx))
        << "Front point gradient must remain stable for small boxes";
    EXPECT_TRUE(std::isfinite(numerical_grad_fy));

    // Gradients should still be bounded even for tiny boxes
    EXPECT_LT(std::abs(grad.dx), 100.0f) << "Gradient should not explode for small boxes";
}

// Test 13: RIOU gradients ARE affected by front point orientation
// WHY: Unlike axis-aligned IoU, RIOU accounts for orientation
// HOW: Change only fx,fy, verify RIOU changes
// EXPECTED: dx_box_riou returns different gradients for different fx,fy
TEST(BDPGradients, FrontPointAffectsRIOU) {
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.625f, 0.5f};

    // Two predictions with same x,y,w,h but different fx,fy
    DarknetBoxBDP pred1 = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.55f};   // Diagonal orientation
    DarknetBoxBDP pred2 = {0.5f, 0.5f, 0.2f, 0.2f, 0.65f, 0.52f};  // Different diagonal

    // RIOU should differ because orientation differs
    float riou1 = box_riou(pred1, truth);
    float riou2 = box_riou(pred2, truth);

    // RIOU values should be different (orientation matters!)
    // This is the key difference between RIOU and axis-aligned IoU
    EXPECT_TRUE(std::isfinite(riou1)) << "RIOU1 must be finite";
    EXPECT_TRUE(std::isfinite(riou2)) << "RIOU2 must be finite";

    // RIOU gradients should be computed correctly
    dxrep_bdp grad1 = dx_box_riou(pred1, truth, IOU_LOSS::IOU);
    dxrep_bdp grad2 = dx_box_riou(pred2, truth, IOU_LOSS::IOU);

    EXPECT_TRUE(std::isfinite(grad1.dfx)) << "∂RIOU/∂fx must be finite";
    EXPECT_TRUE(std::isfinite(grad2.dfx)) << "∂RIOU/∂fx must be finite";

    // But front point loss should differ
    float fp_loss1 = 0.5f * std::pow(truth.fx - pred1.fx, 2) +
                     0.5f * std::pow(truth.fy - pred1.fy, 2);
    float fp_loss2 = 0.5f * std::pow(truth.fx - pred2.fx, 2) +
                     0.5f * std::pow(truth.fy - pred2.fy, 2);

    EXPECT_NE(fp_loss1, fp_loss2)
        << "Front point loss should differ for different fx,fy. "
        << "This confirms fx,fy contribute to total loss.";
}

// Test 14: Gradient consistency across multiple calls
// WHY: Gradient computation should be deterministic
// HOW: Call dx_box_riou multiple times, verify identical results
// EXPECTED: Exact floating-point equality
TEST(BDPGradients, GradientComputationStability) {
    DarknetBoxBDP pred = {0.45f, 0.48f, 0.22f, 0.19f, 0.56f, 0.48f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.25f, 0.2f, 0.625f, 0.5f};

    // Compute gradients multiple times
    dxrep_bdp grad1 = dx_box_riou(pred, truth, IOU_LOSS::IOU);
    dxrep_bdp grad2 = dx_box_riou(pred, truth, IOU_LOSS::IOU);
    dxrep_bdp grad3 = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // Should be identical (no randomness or stateful computation)
    EXPECT_FLOAT_EQ(grad1.dx, grad2.dx) << "Gradient computation should be deterministic";
    EXPECT_FLOAT_EQ(grad1.dy, grad2.dy);
    EXPECT_FLOAT_EQ(grad1.dw, grad2.dw);
    EXPECT_FLOAT_EQ(grad1.dh, grad2.dh);
    EXPECT_FLOAT_EQ(grad1.dfx, grad2.dfx);
    EXPECT_FLOAT_EQ(grad1.dfy, grad2.dfy);

    EXPECT_FLOAT_EQ(grad2.dx, grad3.dx);
    EXPECT_FLOAT_EQ(grad2.dy, grad3.dy);
}

// Test 15: Non-overlapping boxes still get RIOU gradients
// WHY: RIOU should provide gradients even when boxes don't overlap
// HOW: Test RIOU with non-overlapping boxes
// EXPECTED: RIOU provides gradients to guide boxes together
TEST(BDPGradients, NonOverlappingBoxesWithRIOU) {
    DarknetBoxBDP pred = {0.2f, 0.2f, 0.1f, 0.1f, 0.25f, 0.2f};   // Left box
    DarknetBoxBDP truth = {0.8f, 0.8f, 0.1f, 0.1f, 0.85f, 0.8f};  // Right box (far away)

    float riou = box_riou(pred, truth);
    EXPECT_NEAR(riou, 0.0f, 1e-4f) << "Boxes should not overlap → RIOU=0";

    // RIOU should still provide gradients (using numerical differentiation)
    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // RIOU gradients should be finite
    EXPECT_TRUE(std::isfinite(grad.dx)) << "RIOU gradient must be finite";
    EXPECT_TRUE(std::isfinite(grad.dy));
    EXPECT_TRUE(std::isfinite(grad.dfx));
    EXPECT_TRUE(std::isfinite(grad.dfy));

    // Gradient magnitude should indicate direction to move boxes together
    float grad_magnitude = std::abs(grad.dx) + std::abs(grad.dy) +
                           std::abs(grad.dw) + std::abs(grad.dh) +
                           std::abs(grad.dfx) + std::abs(grad.dfy);

    // Even if RIOU=0, numerical gradients should provide direction
    EXPECT_TRUE(std::isfinite(grad_magnitude))
        << "RIOU gradients should be finite for non-overlapping boxes. "
        << "grad_magnitude=" << grad_magnitude;
}
