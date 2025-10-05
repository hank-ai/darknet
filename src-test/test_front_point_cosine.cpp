#include <gtest/gtest.h>
#include <cmath>
#include "darknet_internal.hpp"

// Tests for computeFrontPointCosine: Angular correction for BDP orientation loss
// Computes cos(angle/2) between predicted and target orientation vectors
// Used in delta_yolo_box_bdp() to penalize orientation misalignment

// Identical orientation → cos(0/2) = 1
TEST(FrontPointCosineTest, IdenticalOrientation) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};  // Front point to the right
    RectParams target = pred;

    float result = computeFrontPointCosine(pred, target);
    EXPECT_NEAR(result, 1.0f, 1e-6f) << "Identical orientation → cos(0/2) = 1";
}

// Opposite orientation (180°) → cos(180°/2) = cos(90°) = 0
TEST(FrontPointCosineTest, OppositeOrientation) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};    // Right
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.4f, 0.5f};  // Left (180°)

    float result = computeFrontPointCosine(pred, target);
    EXPECT_NEAR(result, 0.0f, 1e-6f) << "Opposite orientation → cos(180°/2) = 0";
}

// Perpendicular orientation (90°) → cos(90°/2) = cos(45°) = √2/2
TEST(FrontPointCosineTest, PerpendicularOrientation) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};    // Right (0°)
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.6f};  // Up (90°)

    float result = computeFrontPointCosine(pred, target);
    float expected = std::cos(M_PI / 4.0f);  // cos(90°/2) = cos(45°) = √2/2
    EXPECT_NEAR(result, expected, 1e-6f) << "Perpendicular → cos(90°/2) = √2/2";
}

// 45° angle → cos(45°/2) = cos(22.5°)
TEST(FrontPointCosineTest, FortyFiveDegrees) {
    // First vector: center (0.5, 0.5) to front (0.6, 0.5) = vector (0.1, 0) pointing right (0°)
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};

    // Second vector: center (0.5, 0.5) to front (0.6, 0.6) = vector (0.1, 0.1) pointing 45° up-right
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.6f};

    float result = computeFrontPointCosine(pred, target);
    float expected = std::cos(M_PI / 8.0f);  // cos(45°/2) = cos(22.5°)
    EXPECT_NEAR(result, expected, 1e-5f) << "45° angle → cos(22.5°)";
}

// Different centers but same orientation vectors
TEST(FrontPointCosineTest, DifferentCenters) {
    // Centers differ, but orientation vectors are the same
    RectParams pred{0.3f, 0.3f, 0.2f, 0.2f, 0.4f, 0.3f};    // vec = (0.1, 0)
    RectParams target{0.7f, 0.7f, 0.2f, 0.2f, 0.8f, 0.7f};  // vec = (0.1, 0)

    float result = computeFrontPointCosine(pred, target);
    EXPECT_NEAR(result, 1.0f, 1e-6f) << "Same orientation vector → aligned";
}

// Range validation: result must be in [0, 1]
TEST(FrontPointCosineTest, RangeValidation) {
    // Random orientations should give result in [0, 1]
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.6f};
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.4f, 0.6f};

    float result = computeFrontPointCosine(pred, target);
    EXPECT_GE(result, 0.0f) << "Result must be >= 0";
    EXPECT_LE(result, 1.0f) << "Result must be <= 1";
    EXPECT_TRUE(std::isfinite(result)) << "Result must be finite";
}

// Zero-length vector (degenerate case: front point equals center)
TEST(FrontPointCosineTest, ZeroLengthVector) {
    // Front point equals center (degenerate case)
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.5f};  // Zero vector
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};

    float result = computeFrontPointCosine(pred, target);
    EXPECT_NEAR(result, 1.0f, 1e-6f) << "Degenerate case → default to aligned";
}

// Both vectors zero-length
TEST(FrontPointCosineTest, BothZeroLength) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.5f};
    RectParams target{0.7f, 0.7f, 0.2f, 0.2f, 0.7f, 0.7f};

    float result = computeFrontPointCosine(pred, target);
    EXPECT_NEAR(result, 1.0f, 1e-6f) << "Both degenerate → default to aligned";
}

// Small angle (near 0°)
TEST(FrontPointCosineTest, SmallAngle) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};       // 0°
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.501f};   // ~0.57°

    float result = computeFrontPointCosine(pred, target);
    EXPECT_GT(result, 0.999f) << "Small angle → near 1.0";
    EXPECT_LE(result, 1.0f) << "Must be <= 1";
}

// Large angle (near 180°)
TEST(FrontPointCosineTest, LargeAngle) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};       // Right
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.4f, 0.501f};   // ~179°

    float result = computeFrontPointCosine(pred, target);
    EXPECT_LT(result, 0.01f) << "Large angle → near 0.0";
    EXPECT_GE(result, 0.0f) << "Must be >= 0";
}

// Symmetry test: cos(θ/2) should be same regardless of which is pred/target
TEST(FrontPointCosineTest, Symmetry) {
    RectParams a{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};
    RectParams b{0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.6f};

    float result_ab = computeFrontPointCosine(a, b);
    float result_ba = computeFrontPointCosine(b, a);

    EXPECT_NEAR(result_ab, result_ba, 1e-6f) << "Should be symmetric";
}

// Numerical stability with very small vectors
TEST(FrontPointCosineTest, VerySmallVectors) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.5001f, 0.5f};
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.5001f};

    float result = computeFrontPointCosine(pred, target);
    EXPECT_TRUE(std::isfinite(result)) << "Small vectors → finite result";
    EXPECT_GE(result, 0.0f) << "Must be >= 0";
    EXPECT_LE(result, 1.0f) << "Must be <= 1";
}

// Different magnitudes, same direction
TEST(FrontPointCosineTest, DifferentMagnitudesSameDirection) {
    // Same direction but different magnitudes of front point vectors
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.55f, 0.5f};    // Small magnitude
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.8f, 0.5f};   // Large magnitude

    float result = computeFrontPointCosine(pred, target);
    EXPECT_NEAR(result, 1.0f, 1e-6f) << "Same direction → aligned regardless of magnitude";
}

// 135° angle test
TEST(FrontPointCosineTest, Angle135Degrees) {
    RectParams pred{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};      // Right (0°)
    RectParams target{0.5f, 0.5f, 0.2f, 0.2f, 0.4f, 0.6f};    // 135°

    float result = computeFrontPointCosine(pred, target);
    float expected = std::cos(135.0f * M_PI / 360.0f);  // cos(135°/2) = cos(67.5°)
    EXPECT_NEAR(result, expected, 1e-5f) << "135° angle → cos(67.5°)";
}
