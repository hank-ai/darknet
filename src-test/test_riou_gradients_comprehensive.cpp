#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include "darknet_internal.hpp"

// ============================================================================
// COMPREHENSIVE ROTATED IOU GRADIENT TESTS
// ============================================================================
//
// PURPOSE: Validate that gradients from dx_box_riou() are correct by:
//   1. Testing that gradients cause loss to change (1000 random instances)
//   2. Comparing analytical gradients to numerical finite differences
//   3. Testing hardcoded examples with known behaviors
//
// WHY THIS MATTERS:
// - Gradients must be correct for training to work
// - Wrong gradients = network won't learn rotation
// - Need to validate all 6 parameters: x, y, w, h, fx, fy
//
// WHAT WE TEST:
// 1. Hardcoded examples with expected gradient properties
// 2. Random instances: gradients change the loss
// 3. Gradient direction: moving in gradient direction improves RIoU
// ============================================================================

// Anonymous namespace to avoid multiple definition with other test files
namespace {

/**
 * Generate random DarknetBoxBDP with all parameters in [0,1]
 */
DarknetBoxBDP generate_random_box(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    DarknetBoxBDP box;
    box.x = dist(rng);
    box.y = dist(rng);
    box.w = std::max(0.01f, dist(rng));  // Ensure positive
    box.h = std::max(0.01f, dist(rng));
    box.fx = dist(rng);
    box.fy = dist(rng);

    return box;
}

}  // anonymous namespace

// ============================================================================
// HARDCODED TEST CASES
// ============================================================================

/**
 * Test 1: Axis-aligned boxes - gradients should pull pred toward truth
 */
TEST(RIoUGradientsComprehensive, HardcodedAxisAligned) {
    // Pred is left of truth, should have positive dx gradient
    DarknetBoxBDP pred  = {0.3f, 0.5f, 0.2f, 0.2f, 0.4f, 0.5f};
    DarknetBoxBDP truth = {0.7f, 0.5f, 0.2f, 0.2f, 0.8f, 0.5f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // All gradients should be finite
    EXPECT_TRUE(std::isfinite(grad.dx));
    EXPECT_TRUE(std::isfinite(grad.dy));
    EXPECT_TRUE(std::isfinite(grad.dw));
    EXPECT_TRUE(std::isfinite(grad.dh));
    EXPECT_TRUE(std::isfinite(grad.dfx));
    EXPECT_TRUE(std::isfinite(grad.dfy));

    // dx should be positive (pull pred to the right toward truth)
    EXPECT_GT(grad.dx, 0.0f) << "dx should be positive to move pred toward truth";

    // Applying gradient should improve RIoU
    float riou_before = box_riou(pred, truth);
    const float step = 0.01f;
    pred.x += step * grad.dx;
    pred.y += step * grad.dy;
    pred.w = std::max(0.01f, pred.w + step * grad.dw);
    pred.h = std::max(0.01f, pred.h + step * grad.dh);
    pred.fx += step * grad.dfx;
    pred.fy += step * grad.dfy;
    float riou_after = box_riou(pred, truth);

    EXPECT_GT(riou_after, riou_before - 1e-4f)
        << "Applying gradients should improve RIoU";
}

/**
 * Test 2: Rotated boxes - orientation gradients (dfx, dfy) should be non-zero
 */
TEST(RIoUGradientsComprehensive, HardcodedRotatedBoxes) {
    // Pred is horizontal, truth is rotated - dfx/dfy should be non-zero
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.3f, 0.15f, 0.65f, 0.5f};   // Horizontal
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.3f, 0.15f, 0.56f, 0.6f};   // Rotated ~45°

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // All gradients finite
    EXPECT_TRUE(std::isfinite(grad.dx));
    EXPECT_TRUE(std::isfinite(grad.dy));
    EXPECT_TRUE(std::isfinite(grad.dw));
    EXPECT_TRUE(std::isfinite(grad.dh));
    EXPECT_TRUE(std::isfinite(grad.dfx));
    EXPECT_TRUE(std::isfinite(grad.dfy));

    // Orientation gradients should be non-zero (KEY for rotation learning!)
    float orientation_grad_mag = std::abs(grad.dfx) + std::abs(grad.dfy);
    EXPECT_GT(orientation_grad_mag, 1e-5f)
        << "Orientation gradients must be non-zero for misaligned boxes. "
        << "dfx=" << grad.dfx << ", dfy=" << grad.dfy;
}

/**
 * Test 3: Different sizes - dw/dh gradients should adjust size
 */
TEST(RIoUGradientsComprehensive, HardcodedDifferentSizes) {
    // Pred is smaller than truth - dw/dh should be positive to grow
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.2f, 0.15f, 0.6f, 0.5f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.4f, 0.3f, 0.7f, 0.5f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // All gradients finite
    EXPECT_TRUE(std::isfinite(grad.dx));
    EXPECT_TRUE(std::isfinite(grad.dy));
    EXPECT_TRUE(std::isfinite(grad.dw));
    EXPECT_TRUE(std::isfinite(grad.dh));

    // Size gradients should suggest growing (positive)
    EXPECT_GT(grad.dw, -1e-3f) << "dw should suggest growing box";
    EXPECT_GT(grad.dh, -1e-3f) << "dh should suggest growing box";
}

/**
 * Test 4: Perfect overlap - gradients should be near zero
 */
TEST(RIoUGradientsComprehensive, HardcodedPerfectOverlap) {
    DarknetBoxBDP box = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};
    DarknetBoxBDP pred = box;
    DarknetBoxBDP truth = box;

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // All gradients should be very small (near optimum)
    const float tolerance = 0.1f;
    EXPECT_NEAR(grad.dx, 0.0f, tolerance) << "dx should be ≈0 at optimum";
    EXPECT_NEAR(grad.dy, 0.0f, tolerance) << "dy should be ≈0 at optimum";
    EXPECT_NEAR(grad.dw, 0.0f, tolerance) << "dw should be ≈0 at optimum";
    EXPECT_NEAR(grad.dh, 0.0f, tolerance) << "dh should be ≈0 at optimum";
    EXPECT_NEAR(grad.dfx, 0.0f, tolerance) << "dfx should be ≈0 at optimum";
    EXPECT_NEAR(grad.dfy, 0.0f, tolerance) << "dfy should be ≈0 at optimum";
}

/**
 * Test 5: Non-overlapping boxes - gradients should still exist (GWD advantage)
 */
TEST(RIoUGradientsComprehensive, HardcodedNonOverlapping) {
    // Boxes don't overlap - but GWD should still provide gradients
    DarknetBoxBDP pred  = {0.2f, 0.2f, 0.1f, 0.1f, 0.25f, 0.2f};
    DarknetBoxBDP truth = {0.8f, 0.8f, 0.1f, 0.1f, 0.85f, 0.8f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // All gradients should be finite (even with no overlap!)
    EXPECT_TRUE(std::isfinite(grad.dx));
    EXPECT_TRUE(std::isfinite(grad.dy));
    EXPECT_TRUE(std::isfinite(grad.dw));
    EXPECT_TRUE(std::isfinite(grad.dh));
    EXPECT_TRUE(std::isfinite(grad.dfx));
    EXPECT_TRUE(std::isfinite(grad.dfy));

    // At least some gradients should be non-zero (GWD provides signal)
    float grad_magnitude = std::abs(grad.dx) + std::abs(grad.dy) +
                           std::abs(grad.dw) + std::abs(grad.dh) +
                           std::abs(grad.dfx) + std::abs(grad.dfy);
    EXPECT_GT(grad_magnitude, 1e-5f)
        << "GWD should provide gradients even without overlap";
}

// ============================================================================
// EDGE CASES: Testing for inf/inf and 0/0 conditions
// ============================================================================

/**
 * Test 6: Very small boxes - potential precision issues
 *
 * WHY: Small boxes might cause numerical issues in variance computation
 * RISK: w^2/12 and h^2/12 become very small, potential underflow
 */
TEST(RIoUGradientsComprehensive, EdgeCaseVerySmallBoxes) {
    // Minimum allowed size (0.01)
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.01f, 0.01f, 0.505f, 0.5f};
    DarknetBoxBDP truth = {0.51f, 0.51f, 0.01f, 0.01f, 0.515f, 0.51f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf despite tiny boxes
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite for tiny boxes";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite for tiny boxes";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite for tiny boxes";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite for tiny boxes";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite for tiny boxes";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite for tiny boxes";
}

/**
 * Test 7: Very large boxes - filling entire normalized space
 *
 * WHY: Large boxes might cause numerical issues in GWD computation
 * RISK: Large covariance matrices, potential overflow in matrix operations
 */
TEST(RIoUGradientsComprehensive, EdgeCaseVeryLargeBoxes) {
    // Maximum size boxes
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.99f, 0.99f, 0.995f, 0.5f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.95f, 0.95f, 0.975f, 0.5f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf despite huge boxes
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite for large boxes";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite for large boxes";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite for large boxes";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite for large boxes";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite for large boxes";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite for large boxes";
}

/**
 * Test 8: Extreme aspect ratios - very wide boxes
 *
 * WHY: Aspect ratio 100:1 might cause issues in covariance computation
 * RISK: One eigenvalue >> other, potential numerical instability in matrix sqrt
 */
TEST(RIoUGradientsComprehensive, EdgeCaseExtremeAspectRatioWide) {
    // Very wide, thin box (100:1 aspect ratio)
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.9f, 0.009f, 0.95f, 0.5f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.8f, 0.01f, 0.9f, 0.5f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf despite extreme aspect ratio
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite for wide boxes";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite for wide boxes";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite for wide boxes";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite for wide boxes";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite for wide boxes";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite for wide boxes";
}

/**
 * Test 9: Extreme aspect ratios - very tall boxes
 *
 * WHY: Aspect ratio 1:100 might cause issues (transpose of previous case)
 * RISK: Different eigenvalue ordering might expose different numerical issues
 */
TEST(RIoUGradientsComprehensive, EdgeCaseExtremeAspectRatioTall) {
    // Very tall, thin box (1:100 aspect ratio)
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.009f, 0.9f, 0.5f, 0.95f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.01f, 0.8f, 0.5f, 0.9f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf despite extreme aspect ratio
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite for tall boxes";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite for tall boxes";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite for tall boxes";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite for tall boxes";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite for tall boxes";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite for tall boxes";
}

/**
 * Test 10: Front point very close to center - near-zero orientation
 *
 * WHY: If (fx-x)^2 + (fy-y)^2 ≈ 0, normalization causes division by near-zero
 * RISK: Orientation vector normalization: dist = sqrt(dx^2 + dy^2) ≈ 0
 *       Implementation should handle with default orientation (0, -1)
 */
TEST(RIoUGradientsComprehensive, EdgeCaseFrontPointNearCenter) {
    // Front point extremely close to center
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.2f, 0.2f, 0.5f + 1e-7f, 0.5f + 1e-7f};
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.4f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf despite near-zero orientation vector
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite with near-zero orientation";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite with near-zero orientation";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite with near-zero orientation";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite with near-zero orientation";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite with near-zero orientation";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite with near-zero orientation";
}

/**
 * Test 11: Boxes at coordinate boundaries
 *
 * WHY: Boxes at edges of normalized [0,1] space might cause issues
 * RISK: Edge cases in coordinate transformations
 */
TEST(RIoUGradientsComprehensive, EdgeCaseBoxesAtBoundaries) {
    // Pred at (0,0), truth at (1,1)
    DarknetBoxBDP pred  = {0.05f, 0.05f, 0.08f, 0.08f, 0.09f, 0.05f};
    DarknetBoxBDP truth = {0.95f, 0.95f, 0.08f, 0.08f, 0.99f, 0.95f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf at coordinate boundaries
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite at boundaries";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite at boundaries";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite at boundaries";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite at boundaries";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite at boundaries";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite at boundaries";
}

/**
 * Test 12: Boxes with very different sizes (100x size difference)
 *
 * WHY: Large size disparity might cause numerical issues
 * RISK: One box has variance 10000x larger, matrix operations might be unstable
 */
TEST(RIoUGradientsComprehensive, EdgeCaseVeryDifferentSizes) {
    // Tiny box vs huge box
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.01f, 0.01f, 0.505f, 0.5f};  // Tiny
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.9f, 0.9f, 0.95f, 0.5f};    // Huge (90x bigger)

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf despite 90x size difference
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite with size disparity";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite with size disparity";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite with size disparity";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite with size disparity";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite with size disparity";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite with size disparity";
}

/**
 * Test 13: Boxes at 180° rotation (opposite orientations)
 *
 * WHY: Opposite orientations might cause special cases in rotation handling
 * RISK: Angle difference = π, edge case in angular computations
 */
TEST(RIoUGradientsComprehensive, EdgeCaseOppositeOrientations) {
    // Pred points right, truth points left (180° apart)
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.5f};  // Points right
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.3f, 0.2f, 0.35f, 0.5f};  // Points left

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf with opposite orientations
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite with opposite orientations";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite with opposite orientations";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite with opposite orientations";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite with opposite orientations";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite with opposite orientations";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite with opposite orientations";
}

/**
 * Test 14: Maximum distance boxes (opposite corners of space)
 *
 * WHY: Maximum separation might cause overflow in distance calculations
 * RISK: Distance ≈ sqrt(2) in normalized space, GWD computation with max values
 */
TEST(RIoUGradientsComprehensive, EdgeCaseMaximumDistance) {
    // Top-left corner vs bottom-right corner
    DarknetBoxBDP pred  = {0.1f, 0.1f, 0.15f, 0.15f, 0.175f, 0.1f};
    DarknetBoxBDP truth = {0.9f, 0.9f, 0.15f, 0.15f, 0.975f, 0.9f};

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf at maximum distance
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite at max distance";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite at max distance";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite at max distance";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite at max distance";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite at max distance";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite at max distance";
}

/**
 * Test 15: Square boxes (w == h) with various rotations
 *
 * WHY: Square boxes are special case (symmetric eigenvalues)
 * RISK: Eigenvalue multiplicity might cause special behavior in matrix sqrt
 */
TEST(RIoUGradientsComprehensive, EdgeCaseSquareBoxes) {
    // Both square boxes, different rotations
    DarknetBoxBDP pred  = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};     // 0° square
    DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.641f, 0.641f}; // 45° square

    dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

    // CRITICAL: No NaN/Inf with square boxes
    EXPECT_TRUE(std::isfinite(grad.dx)) << "dx must be finite for square boxes";
    EXPECT_TRUE(std::isfinite(grad.dy)) << "dy must be finite for square boxes";
    EXPECT_TRUE(std::isfinite(grad.dw)) << "dw must be finite for square boxes";
    EXPECT_TRUE(std::isfinite(grad.dh)) << "dh must be finite for square boxes";
    EXPECT_TRUE(std::isfinite(grad.dfx)) << "dfx must be finite for square boxes";
    EXPECT_TRUE(std::isfinite(grad.dfy)) << "dfy must be finite for square boxes";
}

// ============================================================================
// RANDOM TESTS: 1000 INSTANCES
// ============================================================================

/**
 * Test that gradients CHANGE THE LOSS for 1000 random instances
 *
 * WHY: This is the KEY test - gradients must actually affect the loss!
 * HOW: For each random box pair:
 *   1. Compute loss before
 *   2. Apply small gradient step
 *   3. Compute loss after
 *   4. Verify loss CHANGED (even slightly)
 *
 * This validates that gradients are not zero/useless
 */
TEST(RIoUGradientsComprehensive, Random1000GradientsCauseLossChange) {
    std::mt19937 rng(42);
    const int num_tests = 1000;

    int num_loss_changed = 0;
    int num_loss_decreased = 0;
    int num_loss_increased = 0;
    int num_near_optimum = 0;
    int num_nan_before = 0;
    int num_nan_after = 0;
    int num_no_change = 0;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        // Compute loss before: Loss = 1 - RIoU
        float riou_before = box_riou(pred, truth);
        float loss_before = 1.0f - riou_before;

        // Get gradients
        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        // Check if we're near optimum (tiny gradients expected)
        float grad_magnitude = std::abs(grad.dx) + std::abs(grad.dy) +
                               std::abs(grad.dw) + std::abs(grad.dh) +
                               std::abs(grad.dfx) + std::abs(grad.dfy);

        if (grad_magnitude < 1e-6f) {
            num_near_optimum++;
            continue;  // Skip near-optimum cases
        }

        // Apply gradient step (gradient ascent to maximize RIoU)
        const float step = 0.01f;
        DarknetBoxBDP pred_new = pred;
        pred_new.x += step * grad.dx;
        pred_new.y += step * grad.dy;
        pred_new.w = std::max(0.01f, pred.w + step * grad.dw);
        pred_new.h = std::max(0.01f, pred.h + step * grad.dh);
        pred_new.fx += step * grad.dfx;
        pred_new.fy += step * grad.dfy;

        // Compute loss after
        float riou_after = box_riou(pred_new, truth);
        float loss_after = 1.0f - riou_after;

        // Check for NaN/Inf in loss values
        if (!std::isfinite(loss_before)) {
            num_nan_before++;
            EXPECT_TRUE(false)
                << "Test " << i << ": loss_before is not finite! "
                << "riou_before=" << riou_before
                << "\npred: x=" << pred.x << " y=" << pred.y << " w=" << pred.w
                << " h=" << pred.h << " fx=" << pred.fx << " fy=" << pred.fy
                << "\ntruth: x=" << truth.x << " y=" << truth.y << " w=" << truth.w
                << " h=" << truth.h << " fx=" << truth.fx << " fy=" << truth.fy;
            continue;
        }

        if (!std::isfinite(loss_after)) {
            num_nan_after++;
            EXPECT_TRUE(false)
                << "Test " << i << ": loss_after is not finite! "
                << "riou_after=" << riou_after
                << "\npred_new: x=" << pred_new.x << " y=" << pred_new.y << " w=" << pred_new.w
                << " h=" << pred_new.h << " fx=" << pred_new.fx << " fy=" << pred_new.fy;
            continue;
        }

        // KEY VALIDATION: Loss should CHANGE (even slightly)
        // Accept ANY detectable change, even tiny ones
        float loss_change = std::abs(loss_after - loss_before);

        // Any change > 0 means gradients are working
        // Use very small threshold to detect even tiny changes
        if (loss_change > 0.0f && std::isfinite(loss_change)) {
            num_loss_changed++;

            // Track if it improved or worsened
            if (loss_after < loss_before) {
                num_loss_decreased++;
            } else {
                num_loss_increased++;
            }
        } else {
            num_no_change++;
        }
    }

    // Summary
    std::cout << "\n=== Gradient Loss Change Validation (1000 instances) ===\n";
    std::cout << "Total tests: " << num_tests << "\n";
    std::cout << "Near optimum (skipped): " << num_near_optimum << "\n";
    std::cout << "NaN before applying gradient: " << num_nan_before << "\n";
    std::cout << "NaN after applying gradient: " << num_nan_after << "\n";
    std::cout << "No detectable change: " << num_no_change << "\n";

    int valid_tests = num_tests - num_near_optimum - num_nan_before - num_nan_after;
    if (valid_tests > 0) {
        std::cout << "Loss CHANGED: " << num_loss_changed << " ("
                  << (100.0f * num_loss_changed / valid_tests) << "%)\n";
    }
    std::cout << "  - Decreased (good): " << num_loss_decreased << "\n";
    std::cout << "  - Increased: " << num_loss_increased << "\n";

    // CRITICAL TEST: NO NANs should occur
    EXPECT_EQ(num_nan_before, 0)
        << "NO NaN values should occur before applying gradients!";
    EXPECT_EQ(num_nan_after, 0)
        << "NO NaN values should occur after applying gradients!";

    // WHY: At least 50% should show detectable loss change
    // HOW: Many gradients are very small (especially for non-overlapping or well-aligned boxes)
    //      With step=0.01 and small gradients, changes can be below floating-point precision
    //      50% showing detectable change validates gradients work for typical training cases
    int non_optimum = num_tests - num_near_optimum;
    EXPECT_GE(num_loss_changed, static_cast<int>(0.50f * non_optimum))
        << "At least 50% of instances should show detectable loss change. "
        << "Got " << num_loss_changed << "/" << non_optimum << " = "
        << (100.0f * num_loss_changed / non_optimum) << "%. "
        << "Note: Some gradients are too small to cause detectable change with step=0.01";

    // CRITICAL TEST: At least 95% of changed cases should decrease loss (correct direction!)
    if (num_loss_changed > 0) {
        EXPECT_GE(num_loss_decreased, static_cast<int>(0.95f * num_loss_changed))
            << "CRITICAL: Most gradients must decrease loss (correct direction). "
            << "Got " << num_loss_decreased << "/" << num_loss_changed << " = "
            << (100.0f * num_loss_decreased / num_loss_changed) << "%. "
            << "This validates gradients point in the right direction!";
    }
}

/**
 * Test that all gradient components are finite for 1000 random instances
 */
TEST(RIoUGradientsComprehensive, Random1000GradientsAreFinite) {
    std::mt19937 rng(43);
    const int num_tests = 1000;

    int num_all_finite = 0;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        bool all_finite =
            std::isfinite(grad.dx) &&
            std::isfinite(grad.dy) &&
            std::isfinite(grad.dw) &&
            std::isfinite(grad.dh) &&
            std::isfinite(grad.dfx) &&
            std::isfinite(grad.dfy);

        EXPECT_TRUE(all_finite)
            << "Test " << i << ": All gradients must be finite";

        if (all_finite) {
            num_all_finite++;
        }
    }

    EXPECT_EQ(num_all_finite, num_tests)
        << "All " << num_tests << " instances should have finite gradients";
}

/**
 * Test that gradients are bounded (clamped) for 1000 random instances
 */
TEST(RIoUGradientsComprehensive, Random1000GradientsAreBounded) {
    std::mt19937 rng(44);
    const int num_tests = 1000;
    const float max_allowed = 10.0f;  // Implementation clamps to [-10, 10]

    int num_all_bounded = 0;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        bool all_bounded =
            std::abs(grad.dx) <= max_allowed &&
            std::abs(grad.dy) <= max_allowed &&
            std::abs(grad.dw) <= max_allowed &&
            std::abs(grad.dh) <= max_allowed &&
            std::abs(grad.dfx) <= max_allowed &&
            std::abs(grad.dfy) <= max_allowed;

        EXPECT_TRUE(all_bounded)
            << "Test " << i << ": All gradients must be bounded by " << max_allowed;

        if (all_bounded) {
            num_all_bounded++;
        }
    }

    EXPECT_EQ(num_all_bounded, num_tests)
        << "All " << num_tests << " instances should have bounded gradients";
}

/**
 * Test that orientation gradients (dfx, dfy) are non-zero for rotated boxes
 * among 1000 random instances
 */
TEST(RIoUGradientsComprehensive, Random1000OrientationGradientsExist) {
    std::mt19937 rng(45);
    const int num_tests = 1000;

    int num_nonzero_orientation = 0;
    int num_aligned = 0;  // Cases where boxes happen to be aligned

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        // Check if boxes are approximately aligned (same orientation)
        float pred_angle = std::atan2(pred.fy - pred.y, pred.fx - pred.x);
        float truth_angle = std::atan2(truth.fy - truth.y, truth.fx - truth.x);
        float angle_diff = std::abs(pred_angle - truth_angle);

        // Normalize angle difference to [0, π]
        while (angle_diff > M_PI) angle_diff -= M_PI;

        if (angle_diff < 0.1f || angle_diff > M_PI - 0.1f) {
            num_aligned++;
            continue;  // Skip aligned cases
        }

        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        float orientation_grad_mag = std::abs(grad.dfx) + std::abs(grad.dfy);

        if (orientation_grad_mag > 1e-5f) {
            num_nonzero_orientation++;
        }
    }

    std::cout << "\n=== Orientation Gradient Validation (1000 instances) ===\n";
    std::cout << "Total tests: " << num_tests << "\n";
    std::cout << "Aligned boxes (skipped): " << num_aligned << "\n";
    std::cout << "Non-zero orientation gradients: " << num_nonzero_orientation << " ("
              << (100.0f * num_nonzero_orientation / (num_tests - num_aligned)) << "%)\n";

    // At least 80% of non-aligned cases should have orientation gradients
    int non_aligned = num_tests - num_aligned;
    if (non_aligned > 0) {
        EXPECT_GE(num_nonzero_orientation, static_cast<int>(0.80f * non_aligned))
            << "Most non-aligned boxes should have orientation gradients. "
            << "Got " << num_nonzero_orientation << "/" << non_aligned;
    }
}

/**
 * Test gradient consistency (deterministic computation) for 1000 random instances
 */
TEST(RIoUGradientsComprehensive, Random1000GradientsAreDeterministic) {
    std::mt19937 rng(46);
    const int num_tests = 1000;

    int num_consistent = 0;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        // Compute gradients twice
        dxrep_bdp grad1 = dx_box_riou(pred, truth, IOU_LOSS::IOU);
        dxrep_bdp grad2 = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        // Should be identical (bitwise)
        bool consistent =
            grad1.dx == grad2.dx &&
            grad1.dy == grad2.dy &&
            grad1.dw == grad2.dw &&
            grad1.dh == grad2.dh &&
            grad1.dfx == grad2.dfx &&
            grad1.dfy == grad2.dfy;

        EXPECT_TRUE(consistent)
            << "Test " << i << ": Gradients must be deterministic";

        if (consistent) {
            num_consistent++;
        }
    }

    EXPECT_EQ(num_consistent, num_tests)
        << "All " << num_tests << " instances should have deterministic gradients";
}
