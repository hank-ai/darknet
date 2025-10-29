// Test file for RotatedRectTransform: BDP to pixel coordinate conversion
// These functions are used for RIOU loss calculation during training
//
// Purpose: Convert BDP parameters (x,y,w,h,fx,fy) in normalized [0,1] coords
//          to 4 corner points in pixel coordinates for rotated IoU computation

#include "darknet_internal.hpp"
#include <gtest/gtest.h>
#include <cmath>

// ============================================================================
// TEST GROUP 1: BASIC STRUCTURE VALIDATION
// ============================================================================

TEST(RotatedRectTransform, Point2DDefaultConstructor) {
    Point2D p;
    EXPECT_FLOAT_EQ(p.x, 0.0f);
    EXPECT_FLOAT_EQ(p.y, 0.0f);
}

TEST(RotatedRectTransform, Point2DParameterizedConstructor) {
    Point2D p(10.5f, 20.3f);
    EXPECT_FLOAT_EQ(p.x, 10.5f);
    EXPECT_FLOAT_EQ(p.y, 20.3f);
}

TEST(RotatedRectTransform, RectParamsIsValid) {
    // Valid parameters
    RectParams valid = {0.5f, 0.5f, 0.2f, 0.3f, 0.6f, 0.5f};
    EXPECT_TRUE(valid.isValid());

    // Invalid: negative width
    RectParams invalid_w = {0.5f, 0.5f, -0.2f, 0.3f, 0.6f, 0.5f};
    EXPECT_FALSE(invalid_w.isValid());

    // Invalid: zero height
    RectParams invalid_h = {0.5f, 0.5f, 0.2f, 0.0f, 0.6f, 0.5f};
    EXPECT_FALSE(invalid_h.isValid());

    // Invalid: NaN in fx
    RectParams invalid_nan = {0.5f, 0.5f, 0.2f, 0.3f, NAN, 0.5f};
    EXPECT_FALSE(invalid_nan.isValid());

    // Invalid: Inf in fy
    RectParams invalid_inf = {0.5f, 0.5f, 0.2f, 0.3f, 0.6f, INFINITY};
    EXPECT_FALSE(invalid_inf.isValid());
}

// ============================================================================
// TEST GROUP 2: AXIS-ALIGNED RECTANGLES (NO ROTATION)
// ============================================================================

TEST(RotatedRectTransform, AxisAlignedRectangle) {
    // Axis-aligned box: front point directly above center
    // Center at (0.5, 0.5), size 0.4x0.2, front point at (0.5, 0.4) -> pointing up
    RectParams params;
    params.x = 0.5f;   // Center x
    params.y = 0.5f;   // Center y
    params.w = 0.4f;   // Width
    params.h = 0.2f;   // Height
    params.fx = 0.5f;  // Front point x (same as center -> vertical)
    params.fy = 0.4f;  // Front point y (above center -> pointing up)

    int img_w = 100;
    int img_h = 100;

    RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);

    // For axis-aligned rectangle pointing up:
    // Expected corners in pixel coords:
    // Top-left:     (30, 40) = (0.5 - 0.4/2, 0.5 - 0.2/2) * 100
    // Top-right:    (70, 40) = (0.5 + 0.4/2, 0.5 - 0.2/2) * 100
    // Bottom-right: (70, 60) = (0.5 + 0.4/2, 0.5 + 0.2/2) * 100
    // Bottom-left:  (30, 60) = (0.5 - 0.4/2, 0.5 + 0.2/2) * 100

    EXPECT_NEAR(corners.p1.x, 30.0f, 1.0f);  // Top-left x
    EXPECT_NEAR(corners.p1.y, 40.0f, 1.0f);  // Top-left y
    EXPECT_NEAR(corners.p2.x, 70.0f, 1.0f);  // Top-right x
    EXPECT_NEAR(corners.p2.y, 40.0f, 1.0f);  // Top-right y
    EXPECT_NEAR(corners.p3.x, 70.0f, 1.0f);  // Bottom-right x
    EXPECT_NEAR(corners.p3.y, 60.0f, 1.0f);  // Bottom-right y
    EXPECT_NEAR(corners.p4.x, 30.0f, 1.0f);  // Bottom-left x
    EXPECT_NEAR(corners.p4.y, 60.0f, 1.0f);  // Bottom-left y
}

TEST(RotatedRectTransform, AxisAlignedRectangleHorizontal) {
    // Front point to the right of center -> pointing right
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.2f;  // Width (perpendicular to front direction)
    params.h = 0.4f;  // Height (along front direction)
    params.fx = 0.7f; // Front point to the right
    params.fy = 0.5f; // Same y as center

    int img_w = 100;
    int img_h = 100;

    RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);

    // All corners should be finite
    EXPECT_TRUE(std::isfinite(corners.p1.x) && std::isfinite(corners.p1.y));
    EXPECT_TRUE(std::isfinite(corners.p2.x) && std::isfinite(corners.p2.y));
    EXPECT_TRUE(std::isfinite(corners.p3.x) && std::isfinite(corners.p3.y));
    EXPECT_TRUE(std::isfinite(corners.p4.x) && std::isfinite(corners.p4.y));
}

// ============================================================================
// TEST GROUP 3: ROTATED RECTANGLES
// ============================================================================

TEST(RotatedRectTransform, RotatedRectangle45Degrees) {
    // Create a rectangle rotated approximately 45 degrees
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.2f;
    params.h = 0.2f;
    params.fx = 0.6414f;  // ~45 degree rotation: cos(45)*0.2 ≈ 0.1414
    params.fy = 0.3586f;  // sin(45)*0.2 ≈ -0.1414

    int img_w = 100;
    int img_h = 100;

    RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);

    // Verify all corners are finite and in reasonable range
    for (const Point2D* p : {&corners.p1, &corners.p2, &corners.p3, &corners.p4}) {
        EXPECT_TRUE(std::isfinite(p->x));
        EXPECT_TRUE(std::isfinite(p->y));
        EXPECT_GE(p->x, -50.0f);  // Allow some out-of-bounds due to rotation
        EXPECT_LE(p->x, 150.0f);
        EXPECT_GE(p->y, -50.0f);
        EXPECT_LE(p->y, 150.0f);
    }
}

// ============================================================================
// TEST GROUP 4: EDGE CASES
// ============================================================================

TEST(RotatedRectTransform, DegenerateFrontPoint) {
    // Front point equals center (degenerate case)
    // Should default to pointing up
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.2f;
    params.h = 0.3f;
    params.fx = 0.5f;  // Same as center x
    params.fy = 0.5f;  // Same as center y

    int img_w = 100;
    int img_h = 100;

    // Should not crash, should handle gracefully
    EXPECT_NO_THROW({
        RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);
        // Verify all corners are finite
        EXPECT_TRUE(std::isfinite(corners.p1.x) && std::isfinite(corners.p1.y));
        EXPECT_TRUE(std::isfinite(corners.p2.x) && std::isfinite(corners.p2.y));
        EXPECT_TRUE(std::isfinite(corners.p3.x) && std::isfinite(corners.p3.y));
        EXPECT_TRUE(std::isfinite(corners.p4.x) && std::isfinite(corners.p4.y));
    });
}

TEST(RotatedRectTransform, VerySmallRectangle) {
    // Very small rectangle (1% of image size)
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.01f;
    params.h = 0.01f;
    params.fx = 0.505f;
    params.fy = 0.5f;

    int img_w = 640;
    int img_h = 480;

    RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);

    // All corners should be close to center
    float center_x = 0.5f * img_w;
    float center_y = 0.5f * img_h;

    EXPECT_NEAR(corners.p1.x, center_x, 10.0f);
    EXPECT_NEAR(corners.p1.y, center_y, 10.0f);
}

TEST(RotatedRectTransform, VeryLargeRectangle) {
    // Rectangle larger than image
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 1.5f;  // 150% of image width
    params.h = 1.2f;  // 120% of image height
    params.fx = 0.6f;
    params.fy = 0.5f;

    int img_w = 100;
    int img_h = 100;

    // Should not crash
    EXPECT_NO_THROW({
        RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);
        // Corners will be outside image bounds, but should be finite
        EXPECT_TRUE(std::isfinite(corners.p1.x));
        EXPECT_TRUE(std::isfinite(corners.p2.x));
    });
}

// ============================================================================
// TEST GROUP 5: DIFFERENT IMAGE SIZES
// ============================================================================

TEST(RotatedRectTransform, SquareImage) {
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.2f;
    params.h = 0.3f;
    params.fx = 0.6f;
    params.fy = 0.5f;

    RectCorners corners = RotatedRectTransform::forward(params, 512, 512);

    // Verify corners are in reasonable range for 512x512 image
    for (const Point2D* p : {&corners.p1, &corners.p2, &corners.p3, &corners.p4}) {
        EXPECT_TRUE(std::isfinite(p->x));
        EXPECT_TRUE(std::isfinite(p->y));
    }
}

TEST(RotatedRectTransform, RectangularImage) {
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.2f;
    params.h = 0.3f;
    params.fx = 0.6f;
    params.fy = 0.5f;

    // Wide image (1920x1080)
    RectCorners corners_wide = RotatedRectTransform::forward(params, 1920, 1080);
    EXPECT_TRUE(std::isfinite(corners_wide.p1.x));

    // Tall image (1080x1920)
    RectCorners corners_tall = RotatedRectTransform::forward(params, 1080, 1920);
    EXPECT_TRUE(std::isfinite(corners_tall.p1.x));
}

// ============================================================================
// TEST GROUP 6: INVERSE TRANSFORM
// ============================================================================

TEST(RotatedRectTransform, InverseAxisAlignedRectangle) {
    // Test inverse transform with axis-aligned rectangle
    // Pixel corners for 40x20 rect centered at (50,50) pointing up
    RectCorners corners;
    corners.p1 = Point2D(30, 40);  // Front-left
    corners.p2 = Point2D(70, 40);  // Front-right
    corners.p3 = Point2D(70, 60);  // Back-right
    corners.p4 = Point2D(30, 60);  // Back-left

    int img_w = 100;
    int img_h = 100;

    auto result = RotatedRectTransform::inverse(corners, img_w, img_h);

    if (result.has_value()) {
        RectParams params = result.value();

        // Center should be at (0.5, 0.5)
        EXPECT_NEAR(params.x, 0.5f, 0.01f);
        EXPECT_NEAR(params.y, 0.5f, 0.01f);

        // Width should be 0.4 (40 pixels / 100)
        EXPECT_NEAR(params.w, 0.4f, 0.01f);

        // Height should be 0.2 (20 pixels / 100)
        EXPECT_NEAR(params.h, 0.2f, 0.01f);

        // Front point should be at (0.5, 0.4) - pointing up
        EXPECT_NEAR(params.fx, 0.5f, 0.01f);
        EXPECT_NEAR(params.fy, 0.4f, 0.01f);

        // Parameters should be valid
        EXPECT_TRUE(params.isValid());
    }
}

TEST(RotatedRectTransform, InverseHorizontalRectangle) {
    // Test inverse transform with horizontal rectangle pointing right
    // 40x20 rect (h=40 along direction, w=20 perpendicular) centered at (50,50)
    RectCorners corners;
    corners.p1 = Point2D(70, 40);  // Front-left
    corners.p2 = Point2D(70, 60);  // Front-right
    corners.p3 = Point2D(30, 60);  // Back-right
    corners.p4 = Point2D(30, 40);  // Back-left

    int img_w = 100;
    int img_h = 100;

    auto result = RotatedRectTransform::inverse(corners, img_w, img_h);

    if (result.has_value()) {
        RectParams params = result.value();

        // Center should be at (0.5, 0.5)
        EXPECT_NEAR(params.x, 0.5f, 0.01f);
        EXPECT_NEAR(params.y, 0.5f, 0.01f);

        // Parameters should be valid
        EXPECT_TRUE(params.isValid());

        // Front point should be to the right of center
        EXPECT_GT(params.fx, params.x);
    }
}

TEST(RotatedRectTransform, ForwardInverseRoundTrip) {
    // Test that forward -> inverse returns original parameters
    RectParams original;
    original.x = 0.5f;
    original.y = 0.5f;
    original.w = 0.3f;
    original.h = 0.4f;
    original.fx = 0.65f;
    original.fy = 0.45f;

    int img_w = 200;
    int img_h = 200;

    // Forward transform
    RectCorners corners = RotatedRectTransform::forward(original, img_w, img_h);

    // Inverse transform
    auto result = RotatedRectTransform::inverse(corners, img_w, img_h);

    if (result.has_value()) {
        RectParams recovered = result.value();

        // Should recover original parameters within tolerance
        // Note: fx,fy tolerance is higher due to compounding rounding errors
        EXPECT_NEAR(recovered.x, original.x, 0.02f);
        EXPECT_NEAR(recovered.y, original.y, 0.02f);
        EXPECT_NEAR(recovered.w, original.w, 0.02f);
        EXPECT_NEAR(recovered.h, original.h, 0.02f);
        EXPECT_NEAR(recovered.fx, original.fx, 0.05f);
        EXPECT_NEAR(recovered.fy, original.fy, 0.05f);
    }
}

TEST(RotatedRectTransform, InverseRotated45Degrees) {
    // Test inverse for rotated rectangle
    RectParams original;
    original.x = 0.5f;
    original.y = 0.5f;
    original.w = 0.2f;
    original.h = 0.3f;
    original.fx = 0.6414f;  // ~45 degrees
    original.fy = 0.3586f;

    int img_w = 100;
    int img_h = 100;

    RectCorners corners = RotatedRectTransform::forward(original, img_w, img_h);
    auto result = RotatedRectTransform::inverse(corners, img_w, img_h);

    if (result.has_value()) {
        RectParams recovered = result.value();

        // Should be valid
        EXPECT_TRUE(recovered.isValid());

        // Center should be preserved
        EXPECT_NEAR(recovered.x, original.x, 0.01f);
        EXPECT_NEAR(recovered.y, original.y, 0.01f);

        // Dimensions should be preserved
        EXPECT_NEAR(recovered.w, original.w, 0.01f);
        EXPECT_NEAR(recovered.h, original.h, 0.01f);
    }
}

TEST(RotatedRectTransform, InverseWithDifferentImageSizes) {
    // Test inverse works correctly with rectangular images
    RectParams original;
    original.x = 0.5f;
    original.y = 0.5f;
    original.w = 0.25f;
    original.h = 0.3f;
    original.fx = 0.6f;
    original.fy = 0.5f;

    // Test with wide image
    int img_w = 640;
    int img_h = 480;

    RectCorners corners = RotatedRectTransform::forward(original, img_w, img_h);
    auto result = RotatedRectTransform::inverse(corners, img_w, img_h);

    if (result.has_value()) {
        RectParams recovered = result.value();

        EXPECT_NEAR(recovered.x, original.x, 0.01f);
        EXPECT_NEAR(recovered.y, original.y, 0.01f);
        EXPECT_NEAR(recovered.w, original.w, 0.01f);
        EXPECT_NEAR(recovered.h, original.h, 0.01f);
    }
}

// ============================================================================
// TEST GROUP 7: GEOMETRY VALIDATION
// ============================================================================

TEST(RotatedRectTransform, OppositeEdgesApproximatelyEqual) {
    // For a valid rectangle, opposite edges should have similar lengths
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.3f;
    params.h = 0.2f;
    params.fx = 0.65f;
    params.fy = 0.45f;

    int img_w = 200;
    int img_h = 200;

    RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);

    // Calculate edge lengths
    auto distance = [](const Point2D& a, const Point2D& b) {
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        return std::sqrt(dx * dx + dy * dy);
    };

    float edge_12 = distance(corners.p1, corners.p2);
    float edge_34 = distance(corners.p3, corners.p4);
    float edge_23 = distance(corners.p2, corners.p3);
    float edge_41 = distance(corners.p4, corners.p1);

    // Opposite edges should be approximately equal
    float tolerance = 5.0f;  // pixels
    EXPECT_NEAR(edge_12, edge_34, tolerance) << "Top and bottom edges should be equal";
    EXPECT_NEAR(edge_23, edge_41, tolerance) << "Left and right edges should be equal";
}

TEST(RotatedRectTransform, AreaConsistency) {
    // Rectangle area should equal w * h * img_w * img_h (approximately)
    RectParams params;
    params.x = 0.5f;
    params.y = 0.5f;
    params.w = 0.4f;
    params.h = 0.3f;
    params.fx = 0.6f;
    params.fy = 0.5f;

    int img_w = 100;
    int img_h = 100;

    RectCorners corners = RotatedRectTransform::forward(params, img_w, img_h);

    // Expected area in pixels
    float expected_area = params.w * params.h * img_w * img_h;

    // Compute actual area using shoelace formula
    float actual_area = 0.0f;
    const Point2D* pts[] = {&corners.p1, &corners.p2, &corners.p3, &corners.p4};
    for (int i = 0; i < 4; i++) {
        int next_i = (i + 1) % 4;
        actual_area += pts[i]->x * pts[next_i]->y - pts[next_i]->x * pts[i]->y;
    }
    actual_area = std::abs(actual_area) / 2.0f;

    // Areas should match within reasonable tolerance
    EXPECT_NEAR(actual_area, expected_area, expected_area * 0.05f)
        << "Computed area should match w*h*img_w*img_h";
}

// Note: main() is provided by test_main.cpp
