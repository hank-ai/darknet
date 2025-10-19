#include <gtest/gtest.h>
#include "darknet_internal.hpp"
#include <cmath>

// Tests for Angular Correction: cos(angle/2) penalty for orientation mismatch
// Multiplies IoU by correction factor ∈ [0,1]: perfect alignment→1, opposite→0

float compute_angle(float px, float py, float tx, float ty) {
    float pred_mag = std::sqrt(px*px + py*py);
    float truth_mag = std::sqrt(tx*tx + ty*ty);
    if (pred_mag < 1e-6f || truth_mag < 1e-6f) return 0.0f;
    float dot = px*tx + py*ty, cross = px*ty - py*tx;
    return std::abs(std::atan2(cross, dot));
}

float angular_correction(float angle) {
    return std::cos(angle / 2.0f);
}

// Identical vectors → angle=0, correction=1
TEST(AngularCorrection, IdenticalVectors) {
    float angle = compute_angle(0.1f, 0.05f, 0.1f, 0.05f);
    EXPECT_NEAR(angle, 0.0f, 1e-5f) << "Same direction → angle=0";
    EXPECT_NEAR(angular_correction(angle), 1.0f, 1e-5f) << "angle=0 → correction=1";
}

// Scaled vectors same direction
TEST(AngularCorrection, ScaledVectors) {
    float angle = compute_angle(0.1f, 0.05f, 0.2f, 0.1f);  // 2x scaled
    EXPECT_NEAR(angle, 0.0f, 1e-5f) << "Scaling doesn't change angle";
}

// 90° perpendicular → correction ≈ 0.707
TEST(AngularCorrection, Perpendicular) {
    float angle = compute_angle(0.1f, 0.0f, 0.0f, 0.1f);  // Horizontal vs vertical
    EXPECT_NEAR(angle, M_PI/2.0f, 1e-4f) << "90° angle";
    EXPECT_NEAR(angular_correction(angle), std::cos(M_PI/4.0f), 1e-4f) << "cos(45°)≈0.707";
}

// 180° opposite → correction ≈ 0
TEST(AngularCorrection, Opposite) {
    float angle = compute_angle(0.1f, 0.0f, -0.1f, 0.0f);  // Opposite directions
    EXPECT_NEAR(angle, M_PI, 1e-4f) << "180° angle";
    EXPECT_NEAR(angular_correction(angle), 0.0f, 1e-2f) << "cos(90°)=0";
}

// 45° rotation
TEST(AngularCorrection, Rotated45) {
    float angle = compute_angle(1.0f, 0.0f, 0.707f, 0.707f);  // 0° vs 45°
    EXPECT_NEAR(angle, M_PI/4.0f, 1e-3f) << "45° angle";
    EXPECT_NEAR(angular_correction(angle), std::cos(M_PI/8.0f), 1e-3f) << "cos(22.5°)≈0.924";
}

// Small angle → correction ≈ 1
TEST(AngularCorrection, SmallAngle) {
    float angle = compute_angle(1.0f, 0.0f, 0.9962f, 0.0872f);  // ~5°
    EXPECT_NEAR(angle, 5.0f*M_PI/180.0f, 1e-2f) << "~5° angle";
    EXPECT_GT(angular_correction(angle), 0.99f) << "Small angle → correction≈1";
}

// Zero prediction vector → angle=0 by convention
TEST(AngularCorrection, ZeroPrediction) {
    float angle = compute_angle(0.0f, 0.0f, 0.1f, 0.05f);
    EXPECT_NEAR(angle, 0.0f, 1e-5f) << "Zero pred → angle=0 (convention)";
}

// Zero truth vector → angle=0 by convention
TEST(AngularCorrection, ZeroTruth) {
    float angle = compute_angle(0.1f, 0.05f, 0.0f, 0.0f);
    EXPECT_NEAR(angle, 0.0f, 1e-5f) << "Zero truth → angle=0 (convention)";
}

// Both zero → angle=0
TEST(AngularCorrection, BothZero) {
    float angle = compute_angle(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_NEAR(angle, 0.0f, 1e-5f) << "Both zero → angle=0";
}

// Angle computation is symmetric
TEST(AngularCorrection, Symmetric) {
    float angle_pt = compute_angle(0.1f, 0.08f, 0.12f, 0.06f);
    float angle_tp = compute_angle(0.12f, 0.06f, 0.1f, 0.08f);
    EXPECT_NEAR(angle_pt, angle_tp, 1e-5f) << "angle(p,t) = angle(t,p)";
}

// Correction factor ∈ [0,1]
TEST(AngularCorrection, CorrectionBounded) {
    std::vector<float> angles = {0.0f, M_PI/6, M_PI/4, M_PI/3, M_PI/2, M_PI};
    for (float angle : angles) {
        float corr = angular_correction(angle);
        EXPECT_GE(corr, -1e-6f) << "correction ≥ 0 (tolerance for π)";
        EXPECT_LE(corr, 1.0f) << "correction ≤ 1";
    }
}

// Correction decreases as angle increases
TEST(AngularCorrection, MonotonicDecrease) {
    std::vector<float> angles = {0.0f, M_PI/6, M_PI/4, M_PI/2, 3*M_PI/4, M_PI};
    for (size_t i = 1; i < angles.size(); i++) {
        float c1 = angular_correction(angles[i-1]);
        float c2 = angular_correction(angles[i]);
        EXPECT_GT(c1, c2) << "Larger angle → smaller correction";
    }
}

// Realistic: slight misalignment
TEST(AngularCorrection, SlightMisalignment) {
    float px = 0.15f, py = 0.05f;  // Pred vector
    float tx = 0.2f, ty = 0.0f;    // Truth vector
    float angle = compute_angle(px, py, tx, ty);
    EXPECT_LT(angle, M_PI/6.0f) << "Slight misalignment <30°";
    EXPECT_GT(angular_correction(angle), 0.9f) << "Small angle → minimal penalty";
}

// IoU penalty example
TEST(AngularCorrection, IoUPenalty) {
    float base_iou = 0.8f, angle = M_PI/3.0f;  // 60° misalignment
    float corrected_iou = base_iou * angular_correction(angle);
    EXPECT_NEAR(corrected_iou, 0.8f * std::cos(M_PI/6.0f), 1e-4f) << "IoU×cos(30°)≈0.693";
}

// Tiny vectors numerical stability
TEST(AngularCorrection, TinyVectors) {
    float angle = compute_angle(1e-5f, 5e-6f, 1e-5f, 5e-6f);  // Same tiny direction
    EXPECT_NEAR(angle, 0.0f, 1e-4f) << "Tiny same direction → angle≈0";
    EXPECT_NEAR(angular_correction(angle), 1.0f, 1e-4f) << "correction≈1";
}
