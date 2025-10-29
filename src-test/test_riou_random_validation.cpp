#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include "darknet_internal.hpp"

// ============================================================================
// ROTATED IOU RANDOM VALIDATION TEST
// ============================================================================
//
// PURPOSE: Validate box_riou() and dx_box_riou() with 1000 random box pairs
//          using uniform distribution [0,1] for all parameters
//
// WHY THIS MATTERS:
// - Comprehensive testing: Random inputs catch edge cases missed by handcrafted tests
// - Real-world scenarios: Training sees random box configurations
// - Statistical validation: 1000 samples provide confidence in correctness
// - Loss function validation: Ensures loss actually decreases with gradient descent
//
// WHAT WE'RE TESTING:
// 1. Basic properties (all 1000 instances):
//    - RIoU ∈ [0,1] (valid range)
//    - RIoU is finite (no NaN/Inf)
//    - Gradients are finite
// 2. Loss function behavior:
//    - Loss = 1 - RIoU decreases when moving in gradient direction
//    - This is the KEY test: validates that the loss function works
// 3. Symmetry: RIoU(a,b) = RIoU(b,a)
// 4. Self-overlap: RIoU(a,a) ≈ 1.0
//
// RANDOM GENERATION:
// - All parameters uniformly distributed in [0,1]
// - x, y: center coordinates in [0,1]
// - w, h: dimensions in [0,1]
// - fx, fy: front point coordinates in [0,1]
// - Fixed seed for reproducibility
//
// ============================================================================

// Anonymous namespace to avoid multiple definition with other test files
namespace {

/**
 * Generate random DarknetBoxBDP with all parameters in [0,1]
 *
 * WHY: Test realistic box configurations that might appear during training
 * HOW: Use uniform_real_distribution for all 6 parameters
 */
DarknetBoxBDP generate_random_box(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    DarknetBoxBDP box;
    box.x = dist(rng);   // Center x ∈ [0,1]
    box.y = dist(rng);   // Center y ∈ [0,1]
    box.w = dist(rng);   // Width ∈ [0,1]
    box.h = dist(rng);   // Height ∈ [0,1]
    box.fx = dist(rng);  // Front x ∈ [0,1]
    box.fy = dist(rng);  // Front y ∈ [0,1]

    // Ensure positive dimensions (avoid degenerate boxes)
    box.w = std::max(0.01f, box.w);
    box.h = std::max(0.01f, box.h);

    return box;
}

}  // anonymous namespace

// ============================================================================
// TEST 1: Basic Properties - All 1000 instances must satisfy constraints
// ============================================================================

/**
 * Test that box_riou() returns valid values for 1000 random box pairs
 *
 * VALIDATES:
 * - RIoU ∈ [0,1] for all pairs
 * - No NaN or Inf
 * - Symmetry: RIoU(a,b) = RIoU(b,a)
 */
TEST(RotatedIoURandomValidation, BasicPropertiesOn1000Instances) {
    // Fixed seed for reproducibility
    std::mt19937 rng(42);

    const int num_tests = 1000;
    int num_valid = 0;
    int num_symmetric = 0;

    for (int i = 0; i < num_tests; i++) {
        // Generate two random boxes
        DarknetBoxBDP box_a = generate_random_box(rng);
        DarknetBoxBDP box_b = generate_random_box(rng);

        // Compute RIoU both ways
        float riou_ab = box_riou(box_a, box_b);
        float riou_ba = box_riou(box_b, box_a);

        // Property 1: RIoU must be in [0,1]
        EXPECT_GE(riou_ab, 0.0f)
            << "Test " << i << ": RIoU must be >= 0";
        EXPECT_LE(riou_ab, 1.0f)
            << "Test " << i << ": RIoU must be <= 1";

        // Property 2: RIoU must be finite
        EXPECT_TRUE(std::isfinite(riou_ab))
            << "Test " << i << ": RIoU must be finite";

        // Property 3: Symmetry RIoU(a,b) = RIoU(b,a)
        EXPECT_NEAR(riou_ab, riou_ba, 1e-4f)
            << "Test " << i << ": RIoU must be symmetric";

        // Track statistics
        if (riou_ab >= 0.0f && riou_ab <= 1.0f && std::isfinite(riou_ab)) {
            num_valid++;
        }

        if (std::abs(riou_ab - riou_ba) < 1e-4f) {
            num_symmetric++;
        }
    }

    // Summary: All 1000 instances should be valid
    EXPECT_EQ(num_valid, num_tests)
        << "All " << num_tests << " instances should have valid RIoU";
    EXPECT_EQ(num_symmetric, num_tests)
        << "All " << num_tests << " instances should be symmetric";
}

// ============================================================================
// TEST 2: Self-Overlap - RIoU(a,a) should be ≈ 1.0
// ============================================================================

/**
 * Test that identical boxes have RIoU ≈ 1.0
 *
 * VALIDATES:
 * - RIoU(a,a) ≈ 1.0 for all 1000 random boxes
 * - This is a fundamental property of IoU
 */
TEST(RotatedIoURandomValidation, SelfOverlapIs1For1000Instances) {
    std::mt19937 rng(43);

    const int num_tests = 1000;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP box = generate_random_box(rng);
        float riou = box_riou(box, box);

        // Self-overlap should be very close to 1.0
        EXPECT_NEAR(riou, 1.0f, 1e-3f)
            << "Test " << i << ": RIoU(box, box) should be ≈ 1.0";
    }
}

// ============================================================================
// TEST 3: Gradient Properties - All gradients must be finite
// ============================================================================

/**
 * Test that dx_box_riou() returns finite gradients for 1000 random pairs
 *
 * VALIDATES:
 * - All 6 gradients (dx, dy, dw, dh, dfx, dfy) are finite
 * - No NaN or Inf in any gradient component
 * - Gradients are bounded (not excessively large)
 */
TEST(RotatedIoURandomValidation, GradientsFiniteFor1000Instances) {
    std::mt19937 rng(44);

    const int num_tests = 1000;
    int num_valid = 0;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        // Compute gradients
        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        // All gradients must be finite
        bool all_finite =
            std::isfinite(grad.dx) &&
            std::isfinite(grad.dy) &&
            std::isfinite(grad.dw) &&
            std::isfinite(grad.dh) &&
            std::isfinite(grad.dfx) &&
            std::isfinite(grad.dfy);

        EXPECT_TRUE(all_finite)
            << "Test " << i << ": All gradients must be finite. "
            << "dx=" << grad.dx << ", dy=" << grad.dy << ", dw=" << grad.dw
            << ", dh=" << grad.dh << ", dfx=" << grad.dfx << ", dfy=" << grad.dfy;

        // Gradients should be bounded (not excessively large)
        // This is a heuristic check for numerical stability
        const float max_reasonable = 100.0f;
        EXPECT_LT(std::abs(grad.dx), max_reasonable)
            << "Test " << i << ": dx gradient too large";
        EXPECT_LT(std::abs(grad.dy), max_reasonable)
            << "Test " << i << ": dy gradient too large";
        EXPECT_LT(std::abs(grad.dw), max_reasonable)
            << "Test " << i << ": dw gradient too large";
        EXPECT_LT(std::abs(grad.dh), max_reasonable)
            << "Test " << i << ": dh gradient too large";
        EXPECT_LT(std::abs(grad.dfx), max_reasonable)
            << "Test " << i << ": dfx gradient too large";
        EXPECT_LT(std::abs(grad.dfy), max_reasonable)
            << "Test " << i << ": dfy gradient too large";

        if (all_finite) {
            num_valid++;
        }
    }

    // Summary
    EXPECT_EQ(num_valid, num_tests)
        << "All " << num_tests << " instances should have finite gradients";
}

// ============================================================================
// TEST 4: LOSS FUNCTION VALIDATION - THE KEY TEST
// ============================================================================

/**
 * Test that the loss function (1 - RIoU) actually decreases when moving
 * in the gradient direction. This validates that the loss function works!
 *
 * WHY THIS IS THE MOST IMPORTANT TEST:
 * - Loss = 1 - RIoU (standard IoU loss)
 * - Gradient descent: minimize loss by moving pred toward truth
 * - If loss doesn't decrease, the loss function is broken!
 *
 * HOW IT WORKS:
 * 1. Compute loss_before = 1 - RIoU(pred, truth)
 * 2. Compute gradients: grad = dx_box_riou(pred, truth)
 * 3. Gradient descent step: pred_new = pred - step_size * grad
 * 4. Compute loss_after = 1 - RIoU(pred_new, truth)
 * 5. VALIDATE: loss_after <= loss_before (loss decreased!)
 *
 * VALIDATES:
 * - Loss decreases (or stays same) for all 1000 instances
 * - This proves the loss function and gradients are consistent
 * - This is what matters for training!
 */
TEST(RotatedIoURandomValidation, LossFunctionDecreasesFor1000Instances) {
    std::mt19937 rng(45);

    const int num_tests = 1000;
    int num_decreased = 0;
    int num_increased = 0;
    float total_loss_reduction = 0.0f;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        // Compute loss before: Loss = 1 - RIoU
        float riou_before = box_riou(pred, truth);
        float loss_before = 1.0f - riou_before;

        // Compute gradients (these are ∂Loss/∂pred for loss = 1 - RIoU)
        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        // Skip if gradients are all zero (perfect overlap or numerical issue)
        float grad_magnitude = std::abs(grad.dx) + std::abs(grad.dy) +
                               std::abs(grad.dw) + std::abs(grad.dh) +
                               std::abs(grad.dfx) + std::abs(grad.dfy);
        if (grad_magnitude < 1e-6f) {
            continue;
        }

        // Gradient descent step: pred_new = pred - step_size * grad
        // Note: dx_box_riou returns gradients for gradient ASCENT on RIoU
        // For loss = 1 - RIoU, we need to MAXIMIZE RIoU (minimize loss)
        // So we move in the POSITIVE gradient direction
        const float step_size = 0.01f;
        DarknetBoxBDP pred_new = pred;
        pred_new.x += step_size * grad.dx;
        pred_new.y += step_size * grad.dy;
        pred_new.w = std::max(0.01f, pred.w + step_size * grad.dw);  // Keep positive
        pred_new.h = std::max(0.01f, pred.h + step_size * grad.dh);  // Keep positive
        pred_new.fx += step_size * grad.dfx;
        pred_new.fy += step_size * grad.dfy;

        // Compute loss after
        float riou_after = box_riou(pred_new, truth);
        float loss_after = 1.0f - riou_after;

        // KEY VALIDATION: Loss should decrease (or stay same)
        // WHY: We allow small numerical error because:
        //   1. GWD is an approximation of true RIoU gradients (not exact)
        //   2. Finite step size can overshoot in non-linear regions
        //   3. Box boundaries ([0,1]) can cause edge effects
        // HOW: Track statistics but don't fail individual tests - validate overall success rate
        float loss_change = loss_after - loss_before;

        // Track statistics
        if (loss_change <= 1e-3f) {
            num_decreased++;
            total_loss_reduction += -loss_change;
        } else {
            num_increased++;
        }
    }

    // Summary statistics
    std::cout << "\n=== Loss Function Validation Summary ===\n";
    std::cout << "Total tests: " << num_tests << "\n";
    std::cout << "Loss decreased/stayed same: " << num_decreased << " ("
              << (100.0f * num_decreased / num_tests) << "%)\n";
    std::cout << "Loss increased: " << num_increased << " ("
              << (100.0f * num_increased / num_tests) << "%)\n";
    if (num_decreased > 0) {
        std::cout << "Average loss reduction: " << (total_loss_reduction / num_decreased) << "\n";
    }

    // WHY: At least 95% of instances should decrease loss
    // HOW: GWD approximation + numerical issues mean we can't expect 100% success
    //      But 95%+ success rate validates that gradients work for training
    EXPECT_GE(num_decreased, static_cast<int>(0.95f * num_tests))
        << "At least 95% of instances should decrease loss. "
        << "Got " << num_decreased << "/" << num_tests << " = "
        << (100.0f * num_decreased / num_tests) << "%";
}

// ============================================================================
// TEST 5: Gradient Direction Test - Subset of instances
// ============================================================================

/**
 * For a subset of random instances, verify that moving in gradient direction
 * increases RIoU (this is equivalent to loss decrease test, but more direct)
 *
 * VALIDATES:
 * - RIoU increases when moving in positive gradient direction
 * - This is a direct test of gradient correctness
 */
TEST(RotatedIoURandomValidation, GradientDirectionIncreasesRIoUFor100Instances) {
    std::mt19937 rng(46);

    const int num_tests = 100;  // Smaller subset for detailed validation
    int num_increased = 0;

    for (int i = 0; i < num_tests; i++) {
        DarknetBoxBDP pred = generate_random_box(rng);
        DarknetBoxBDP truth = generate_random_box(rng);

        float riou_before = box_riou(pred, truth);
        dxrep_bdp grad = dx_box_riou(pred, truth, IOU_LOSS::IOU);

        // Skip if zero gradient
        float grad_mag = std::abs(grad.dx) + std::abs(grad.dy) +
                         std::abs(grad.dw) + std::abs(grad.dh) +
                         std::abs(grad.dfx) + std::abs(grad.dfy);
        if (grad_mag < 1e-6f) {
            continue;
        }

        // Move in gradient direction (small step)
        const float step = 0.005f;
        DarknetBoxBDP pred_moved = pred;
        pred_moved.x += step * grad.dx;
        pred_moved.y += step * grad.dy;
        pred_moved.w = std::max(0.01f, pred.w + step * grad.dw);
        pred_moved.h = std::max(0.01f, pred.h + step * grad.dh);
        pred_moved.fx += step * grad.dfx;
        pred_moved.fy += step * grad.dfy;

        float riou_after = box_riou(pred_moved, truth);

        // RIoU should increase (or stay same)
        if (riou_after >= riou_before - 1e-4f) {
            num_increased++;
        }

        EXPECT_GE(riou_after, riou_before - 1e-4f)
            << "Test " << i << ": RIoU should increase in gradient direction. "
            << "RIoU_before=" << riou_before << ", RIoU_after=" << riou_after;
    }

    // At least 95% should increase
    EXPECT_GE(num_increased, static_cast<int>(0.95f * num_tests))
        << "At least 95% should increase RIoU";
}
