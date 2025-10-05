#include <gtest/gtest.h>
#include "darknet_internal.hpp"
#include <cmath>
#include <chrono>
#include <vector>

// Tests for Front Point Loss (Smooth L1)
// Used in BDP to train orientation: loss = 0.5*diff² (|diff|<1) or |diff|-0.5 (|diff|≥1)
// This robust loss reduces outlier sensitivity while enabling precise alignment

float smooth_l1_loss(float diff) {
    return std::abs(diff) < 1.0f ? 0.5f * diff * diff : std::abs(diff) - 0.5f;
}

float smooth_l1_grad(float diff) {
    return std::abs(diff) < 1.0f ? diff : (diff > 0 ? 1.0f : -1.0f);
}

// Perfect match → zero loss
TEST(FrontPointLoss, PerfectMatch) {
    EXPECT_FLOAT_EQ(smooth_l1_loss(0.0f), 0.0f) << "No error → no loss";
}

// Small errors use quadratic: 0.5*diff²
TEST(FrontPointLoss, SmallErrorQuadratic) {
    EXPECT_NEAR(smooth_l1_loss(0.5f), 0.125f, 1e-5f) << "0.5² * 0.5 = 0.125";
    EXPECT_NEAR(smooth_l1_loss(0.3f), 0.045f, 1e-5f) << "0.3² * 0.5 = 0.045";
}

// Large errors use linear: |diff| - 0.5
TEST(FrontPointLoss, LargeErrorLinear) {
    EXPECT_NEAR(smooth_l1_loss(2.0f), 1.5f, 1e-5f) << "2.0 - 0.5 = 1.5";
    EXPECT_NEAR(smooth_l1_loss(3.0f), 2.5f, 1e-5f) << "3.0 - 0.5 = 2.5";
}

// Transition at |diff|=1 is continuous
TEST(FrontPointLoss, TransitionContinuous) {
    EXPECT_NEAR(smooth_l1_loss(1.0f), 0.5f, 1e-3f) << "Both formulas give 0.5 at |diff|=1";
}

// Loss is symmetric: loss(+x) = loss(-x)
TEST(FrontPointLoss, Symmetric) {
    EXPECT_FLOAT_EQ(smooth_l1_loss(0.7f), smooth_l1_loss(-0.7f)) << "Symmetric loss";
}

// Gradient = diff for small errors
TEST(FrontPointLoss, GradSmallError) {
    EXPECT_FLOAT_EQ(smooth_l1_grad(0.6f), 0.6f) << "d/dx(0.5x²) = x";
}

// Gradient = ±1 for large errors (bounded)
TEST(FrontPointLoss, GradLargeError) {
    EXPECT_FLOAT_EQ(smooth_l1_grad(2.5f), 1.0f) << "Large positive → +1";
    EXPECT_FLOAT_EQ(smooth_l1_grad(-3.2f), -1.0f) << "Large negative → -1";
}

// Gradient zero at perfect match
TEST(FrontPointLoss, GradZeroAtOptimum) {
    EXPECT_FLOAT_EQ(smooth_l1_grad(0.0f), 0.0f) << "Optimum → zero gradient";
}

// Gradients are symmetric
TEST(FrontPointLoss, GradSymmetric) {
    EXPECT_NEAR(smooth_l1_grad(0.5f), -smooth_l1_grad(-0.5f), 1e-4f) << "Opposite sign";
}

// Loss always non-negative
TEST(FrontPointLoss, NonNegative) {
    for (float diff : {-5.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 5.0f}) {
        EXPECT_GE(smooth_l1_loss(diff), 0.0f) << "Loss ≥ 0 for diff=" << diff;
    }
}

// Loss increases with |error|
TEST(FrontPointLoss, Monotonic) {
    float l1 = smooth_l1_loss(0.2f), l2 = smooth_l1_loss(0.5f);
    float l3 = smooth_l1_loss(1.0f), l4 = smooth_l1_loss(2.0f);
    EXPECT_LT(l1, l2) << "Larger error → larger loss";
    EXPECT_LT(l2, l3);
    EXPECT_LT(l3, l4);
}

// Gradient magnitude bounded by 1.0
TEST(FrontPointLoss, GradBounded) {
    for (float diff : {-10.0f, -2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 10.0f}) {
        EXPECT_LE(std::abs(smooth_l1_grad(diff)), 1.0f) << "|grad| ≤ 1 for diff=" << diff;
    }
}

// Realistic: small training error
TEST(FrontPointLoss, RealisticSmallError) {
    float diff_fx = 0.05f, diff_fy = -0.05f;
    float loss = smooth_l1_loss(diff_fx) + smooth_l1_loss(diff_fy);
    EXPECT_NEAR(loss, 0.0025f, 1e-5f) << "Small errors → small loss (0.5*0.05²)*2";
}

// Edge case: very tiny errors
TEST(FrontPointLoss, TinyErrors) {
    float loss = smooth_l1_loss(0.001f) + smooth_l1_loss(0.0001f);
    EXPECT_LT(loss, 1e-5f) << "Tiny errors → tiny loss";
    EXPECT_GT(loss, 0.0f) << "Non-zero error → non-zero loss";
}

// Stress test various magnitudes
TEST(FrontPointLoss, StressTest) {
    std::vector<std::pair<float,float>> cases = {
        {0.0f, 0.0f}, {0.01f, 0.01f}, {0.1f, 0.2f}, {0.5f, 0.5f},
        {0.99f, 0.99f}, {1.0f, 1.0f}, {1.5f, 2.0f}, {-0.5f, 0.5f}
    };

    for (auto [fx, fy] : cases) {
        float loss = smooth_l1_loss(fx) + smooth_l1_loss(fy);
        EXPECT_GE(loss, 0.0f) << "Non-negative";
        EXPECT_TRUE(std::isfinite(loss)) << "Finite";

        float gx = smooth_l1_grad(fx), gy = smooth_l1_grad(fy);
        EXPECT_TRUE(std::isfinite(gx) && std::isfinite(gy)) << "Finite grads";
        EXPECT_LE(std::abs(gx), 1.0f) << "Bounded";
        EXPECT_LE(std::abs(gy), 1.0f);
    }
}

// ============================================================================
// Cosine Similarity Tests for Orientation Alignment
// ============================================================================

// Helper: compute cosine of angle between two vectors
float vector_cosine(float x1, float y1, float x2, float y2) {
    float dot = x1 * x2 + y1 * y2;
    float mag1 = std::sqrt(x1 * x1 + y1 * y1);
    float mag2 = std::sqrt(x2 * x2 + y2 * y2);
    if (mag1 < 1e-6f || mag2 < 1e-6f) return 1.0f;  // Undefined → assume aligned
    return dot / (mag1 * mag2);
}

// Perfect alignment → cosine = 1.0
TEST(FrontPointCosine, PerfectAlignment) {
    float cos_sim = vector_cosine(1.0f, 0.0f, 1.0f, 0.0f);
    EXPECT_NEAR(cos_sim, 1.0f, 1e-5f) << "Parallel vectors → cos=1";
}

// Opposite direction → cosine = -1.0
TEST(FrontPointCosine, OppositeDirection) {
    float cos_sim = vector_cosine(1.0f, 0.0f, -1.0f, 0.0f);
    EXPECT_NEAR(cos_sim, -1.0f, 1e-5f) << "Antiparallel → cos=-1";
}

// Perpendicular → cosine = 0.0
TEST(FrontPointCosine, Perpendicular) {
    float cos_sim = vector_cosine(1.0f, 0.0f, 0.0f, 1.0f);
    EXPECT_NEAR(cos_sim, 0.0f, 1e-5f) << "Perpendicular → cos=0";
}

// 45 degree angle → cosine = √2/2
TEST(FrontPointCosine, FortyFiveDegrees) {
    float cos_sim = vector_cosine(1.0f, 0.0f, 1.0f, 1.0f);
    EXPECT_NEAR(cos_sim, std::sqrt(2.0f)/2.0f, 1e-5f) << "45° → cos=√2/2≈0.707";
}

// Cosine range is [-1, 1]
TEST(FrontPointCosine, Range) {
    const size_t number_of_vectors = 10000;

    for (size_t i = 0; i < number_of_vectors; i++) {
        float x1 = rand_uniform(-1.0f, 1.0f);
        float y1 = rand_uniform(-1.0f, 1.0f);
        float x2 = rand_uniform(-1.0f, 1.0f);
        float y2 = rand_uniform(-1.0f, 1.0f);

        float cos_sim = vector_cosine(x1, y1, x2, y2);

        ASSERT_GE(cos_sim, -1.0f) << "cosine >= -1";
        ASSERT_LE(cos_sim, 1.0f) << "cosine <= 1";
        ASSERT_TRUE(std::isfinite(cos_sim)) << "cosine is finite";
    }
}

// Symmetric: cos(a,b) = cos(b,a)
TEST(FrontPointCosine, Symmetric) {
    float cos_ab = vector_cosine(0.5f, 0.3f, 0.7f, 0.2f);
    float cos_ba = vector_cosine(0.7f, 0.2f, 0.5f, 0.3f);
    EXPECT_FLOAT_EQ(cos_ab, cos_ba) << "Cosine is symmetric";
}

// Zero vector handling
TEST(FrontPointCosine, ZeroVector) {
    float cos_sim = vector_cosine(0.0f, 0.0f, 1.0f, 0.0f);
    EXPECT_NEAR(cos_sim, 1.0f, 1e-5f) << "Zero vector → default to aligned";
}

// Small magnitude vectors (numerical stability)
TEST(FrontPointCosine, SmallMagnitude) {
    float cos_sim = vector_cosine(1e-4f, 1e-4f, 1e-4f, -1e-4f);
    ASSERT_TRUE(std::isfinite(cos_sim)) << "Small vectors → finite result";
    ASSERT_GE(cos_sim, -1.0f);
    ASSERT_LE(cos_sim, 1.0f);
}

// ============================================================================
// RIOU (Rotated IoU with Angular Correction) Tests
// ============================================================================

// Perfect overlap → RIOU = 1.0
TEST(RIOU, PerfectOverlap) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};  // Same box
    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};

    float riou = box_iou_bdp(a, b);
    EXPECT_NEAR(riou, 1.0f, 1e-3f) << "Identical boxes → IoU=1";
}

// No overlap → RIOU = 0.0
TEST(RIOU, NoOverlap) {
    DarknetBoxBDP a = {0.2f, 0.2f, 0.1f, 0.1f, 0.25f, 0.2f};
    DarknetBoxBDP b = {0.8f, 0.8f, 0.1f, 0.1f, 0.85f, 0.8f};  // Far apart

    float riou = box_iou_bdp(a, b);
    EXPECT_NEAR(riou, 0.0f, 1e-3f) << "Separated boxes → IoU=0";
}

// RIOU must be in [0, 1]
TEST(RIOU, Range) {
    const size_t number_of_boxes = 50000;

    std::vector<DarknetBoxBDP> boxes;
    boxes.reserve(number_of_boxes);

    // Generate random BDP boxes
    while (boxes.size() < number_of_boxes) {
        DarknetBoxBDP box;
        box.x = rand_uniform(0.0f, 1.0f);
        box.y = rand_uniform(0.0f, 1.0f);
        box.w = rand_uniform(0.05f, 0.5f);
        box.h = rand_uniform(0.05f, 0.5f);

        // Ensure box stays within bounds
        if (box.x + box.w/2.0f > 1.0f || box.y + box.h/2.0f > 1.0f) continue;
        if (box.x - box.w/2.0f < 0.0f || box.y - box.h/2.0f < 0.0f) continue;

        // Front point: random direction from center
        float angle = rand_uniform(0.0f, 2.0f * M_PI);
        float radius = rand_uniform(0.0f, std::min(box.w, box.h) / 2.0f);
        box.fx = box.x + radius * std::cos(angle);
        box.fy = box.y + radius * std::sin(angle);

        boxes.push_back(box);
    }

    std::chrono::high_resolution_clock::duration duration = std::chrono::milliseconds(0);

    for (size_t idx = 0; idx < boxes.size() - 1; idx++) {
        const auto & b1 = boxes[idx];
        const auto & b2 = boxes[idx + 1];

        const auto t1 = std::chrono::high_resolution_clock::now();
        float riou = box_iou_bdp(b1, b2);
        const auto t2 = std::chrono::high_resolution_clock::now();
        duration += (t2 - t1);

        ASSERT_GE(riou, 0.0f) << "RIOU >= 0";
        ASSERT_LE(riou, 1.0f) << "RIOU <= 1";
        ASSERT_TRUE(std::isfinite(riou)) << "RIOU is finite";
    }

    // std::cout << "RIOU took " << Darknet::format_duration_string(duration) << std::endl;
}

// Symmetric: RIOU(a,b) = RIOU(b,a)
TEST(RIOU, Symmetric) {
    DarknetBoxBDP a = {0.3f, 0.3f, 0.2f, 0.15f, 0.4f, 0.3f};
    DarknetBoxBDP b = {0.35f, 0.32f, 0.18f, 0.14f, 0.44f, 0.32f};

    float riou_ab = box_iou_bdp(a, b);
    float riou_ba = box_iou_bdp(b, a);

    EXPECT_NEAR(riou_ab, riou_ba, 1e-5f) << "RIOU is symmetric";
}

// Partial overlap → 0 < RIOU < 1
TEST(RIOU, PartialOverlap) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.3f, 0.2f, 0.6f, 0.5f};
    DarknetBoxBDP b = {0.55f, 0.52f, 0.3f, 0.2f, 0.65f, 0.52f};  // Slight shift

    float riou = box_iou_bdp(a, b);

    EXPECT_GT(riou, 0.0f) << "Overlapping → RIOU > 0";
    EXPECT_LT(riou, 1.0f) << "Not identical → RIOU < 1";
}

// Same position, different orientation → RIOU affected by angular correction
TEST(RIOU, OrientationDifference) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};   // Right-pointing
    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.1f, 0.5f, 0.6f};   // Up-pointing

    float riou = box_iou_bdp(a, b);

    // RIOU should be less than 1.0 due to angular correction
    EXPECT_LT(riou, 1.0f) << "Different orientation → RIOU < 1";
    EXPECT_GT(riou, 0.0f) << "Same position → RIOU > 0";
}

// Aligned boxes → higher RIOU than misaligned
TEST(RIOU, AlignmentBoost) {
    // Two boxes at same location, same orientation
    DarknetBoxBDP aligned1 = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};
    DarknetBoxBDP aligned2 = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};

    // Two boxes at same location, opposite orientation
    DarknetBoxBDP misaligned1 = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};
    DarknetBoxBDP misaligned2 = {0.5f, 0.5f, 0.2f, 0.1f, 0.4f, 0.5f};  // Opposite

    float riou_aligned = box_iou_bdp(aligned1, aligned2);
    float riou_misaligned = box_iou_bdp(misaligned1, misaligned2);

    EXPECT_GT(riou_aligned, riou_misaligned) << "Aligned → higher RIOU";
}

// Zero-area boxes → RIOU = 0
TEST(RIOU, ZeroAreaBox) {
    DarknetBoxBDP a = {0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f};  // Zero area
    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.1f, 0.6f, 0.5f};

    float riou = box_iou_bdp(a, b);
    EXPECT_NEAR(riou, 0.0f, 1e-5f) << "Zero area → IoU=0";
}
