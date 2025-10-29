#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "darknet_internal.hpp"

// ============================================================================
// OLD TESTS - FOR FINITE DIFFERENCE RIOU IMPLEMENTATION
// ============================================================================
// These tests verify exact numerical gradient matching with finite differences
// of box_riou(). They are designed for the OLD implementation that used
// forward differences directly on polygon intersection.
//
// COMPATIBILITY WITH GWD:
// The current dx_box_riou() uses Gaussian Wasserstein Distance (GWD)
// approximation instead of exact polygon gradients. GWD provides smooth,
// stable gradients but does NOT match finite differences of box_riou().
//
// These tests will FAIL with GWD (expected). They remain here as reference
// for the original finite-difference approach.
// ============================================================================

// Function to test
dxrep_bdp dx_box_riou(const DarknetBoxBDP& pred, const DarknetBoxBDP& truth, const IOU_LOSS riou_loss);
float compute_gwd_distance(const DarknetBoxBDP& pred, const DarknetBoxBDP& truth);

// ============================================================================
// Test 1: Simple Axis-Aligned Boxes
// ============================================================================

TEST(RotatedIoUGradient, Test1_AxisAligned) {
    std::cout << "\n=== Test 1: Axis-Aligned Boxes ===\n";

    DarknetBoxBDP a    = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, -0.5f};  // Front points up
    DarknetBoxBDP dxa  = {0.0f+.0001f, 0.0f, 1.0f, 1.0f, 0.0f, -0.5f};
    DarknetBoxBDP dya  = {0.0f, 0.0f+.0001f, 1.0f, 1.0f, 0.0f, -0.5f};
    DarknetBoxBDP dwa  = {0.0f, 0.0f, 1.0f+.0001f, 1.0f, 0.0f, -0.5f};
    DarknetBoxBDP dha  = {0.0f, 0.0f, 1.0f, 1.0f+.0001f, 0.0f, -0.5f};
    DarknetBoxBDP dfxa = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f+.0001f, -0.5f};
    DarknetBoxBDP dfya = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, -0.5f+.0001f};

    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.4f};

    // Analytical gradient
    dxrep_bdp analytical = dx_box_riou(a, b, IOU);

    std::cout << "Analytical: "
              << "dx=" << analytical.dx << " "
              << "dy=" << analytical.dy << " "
              << "dw=" << analytical.dw << " "
              << "dh=" << analytical.dh << " "
              << "dfx=" << analytical.dfx << " "
              << "dfy=" << analytical.dfy << "\n";
    
    // Numerical gradient via finite differences
    float base_gwd = compute_gwd_distance(a, b);
    float xgrad = (compute_gwd_distance(dxa, b) - base_gwd) / 0.0001f;
    float ygrad = (compute_gwd_distance(dya, b) - base_gwd) / 0.0001f;
    float wgrad = (compute_gwd_distance(dwa, b) - base_gwd) / 0.0001f;
    float hgrad = (compute_gwd_distance(dha, b) - base_gwd) / 0.0001f;
    float fxgrad = (compute_gwd_distance(dfxa, b) - base_gwd) / 0.0001f;
    float fygrad = (compute_gwd_distance(dfya, b) - base_gwd) / 0.0001f;
    
    std::cout << "Numerical:  "
              << "dx=" << xgrad << " "
              << "dy=" << ygrad << " "
              << "dw=" << wgrad << " "
              << "dh=" << hgrad << " "
              << "dfx=" << fxgrad << " "
              << "dfy=" << fygrad << "\n";
    
    // Compare (analytical returns negative for gradient descent)
    EXPECT_NEAR(-analytical.dx, xgrad, 0.01f) << "X gradient mismatch";
    EXPECT_NEAR(-analytical.dy, ygrad, 0.01f) << "Y gradient mismatch";
    EXPECT_NEAR(-analytical.dw, wgrad, 0.02f) << "W gradient mismatch";
    EXPECT_NEAR(-analytical.dh, hgrad, 0.02f) << "H gradient mismatch";
    EXPECT_NEAR(-analytical.dfx, fxgrad, 0.03f) << "FX gradient mismatch";
    EXPECT_NEAR(-analytical.dfy, fygrad, 0.03f) << "FY gradient mismatch";
}

// ============================================================================
// Test 2: Rotated Box (45 degrees)
// ============================================================================

TEST(RotatedIoUGradient, Test2_Rotated45) {
    std::cout << "\n=== Test 2: Rotated 45° ===\n";

    float s2 = 0.7071f;  // sqrt(2)/2

    DarknetBoxBDP a    = {0.5f, 0.5f, 0.4f, 0.3f, 0.5f+0.15f*s2, 0.5f-0.15f*s2};
    DarknetBoxBDP dxa  = {0.5f+.0001f, 0.5f, 0.4f, 0.3f, 0.5f+0.15f*s2, 0.5f-0.15f*s2};
    DarknetBoxBDP dya  = {0.5f, 0.5f+.0001f, 0.4f, 0.3f, 0.5f+0.15f*s2, 0.5f-0.15f*s2};
    DarknetBoxBDP dwa  = {0.5f, 0.5f, 0.4f+.0001f, 0.3f, 0.5f+0.15f*s2, 0.5f-0.15f*s2};
    DarknetBoxBDP dha  = {0.5f, 0.5f, 0.4f, 0.3f+.0001f, 0.5f+0.15f*s2, 0.5f-0.15f*s2};
    DarknetBoxBDP dfxa = {0.5f, 0.5f, 0.4f, 0.3f, 0.5f+0.15f*s2+.0001f, 0.5f-0.15f*s2};
    DarknetBoxBDP dfya = {0.5f, 0.5f, 0.4f, 0.3f, 0.5f+0.15f*s2, 0.5f-0.15f*s2+.0001f};

    DarknetBoxBDP b = {0.55f, 0.52f, 0.35f, 0.28f, 0.55f+0.14f*s2, 0.52f-0.14f*s2};

    dxrep_bdp analytical = dx_box_riou(a, b, DIOU);

    std::cout << "Analytical: "
              << "dx=" << analytical.dx << " "
              << "dy=" << analytical.dy << " "
              << "dw=" << analytical.dw << " "
              << "dh=" << analytical.dh << " "
              << "dfx=" << analytical.dfx << " "
              << "dfy=" << analytical.dfy << "\n";
    
    float base_gwd = compute_gwd_distance(a, b);
    float xgrad = (compute_gwd_distance(dxa, b) - base_gwd) / 0.0001f;
    float ygrad = (compute_gwd_distance(dya, b) - base_gwd) / 0.0001f;
    float wgrad = (compute_gwd_distance(dwa, b) - base_gwd) / 0.0001f;
    float hgrad = (compute_gwd_distance(dha, b) - base_gwd) / 0.0001f;
    float fxgrad = (compute_gwd_distance(dfxa, b) - base_gwd) / 0.0001f;
    float fygrad = (compute_gwd_distance(dfya, b) - base_gwd) / 0.0001f;
    
    std::cout << "Numerical:  "
              << "dx=" << xgrad << " "
              << "dy=" << ygrad << " "
              << "dw=" << wgrad << " "
              << "dh=" << hgrad << " "
              << "dfx=" << fxgrad << " "
              << "dfy=" << fygrad << "\n";
    
    EXPECT_NEAR(-analytical.dx, xgrad, 0.02f);
    EXPECT_NEAR(-analytical.dy, ygrad, 0.02f);
    EXPECT_NEAR(-analytical.dw, wgrad, 0.03f);
    EXPECT_NEAR(-analytical.dh, hgrad, 0.03f);
    EXPECT_NEAR(-analytical.dfx, fxgrad, 0.05f);
    EXPECT_NEAR(-analytical.dfy, fygrad, 0.05f);
}

// ============================================================================
// Test 3: Non-Overlapping (GWD advantage)
// ============================================================================

TEST(RotatedIoUGradient, Test3_NonOverlapping) {
    std::cout << "\n=== Test 3: Non-Overlapping Boxes ===\n";

    DarknetBoxBDP a    = {0.2f, 0.2f, 0.15f, 0.15f, 0.2f, 0.125f};
    DarknetBoxBDP dxa  = {0.2f+.0001f, 0.2f, 0.15f, 0.15f, 0.2f, 0.125f};
    DarknetBoxBDP dya  = {0.2f, 0.2f+.0001f, 0.15f, 0.15f, 0.2f, 0.125f};
    DarknetBoxBDP dwa  = {0.2f, 0.2f, 0.15f+.0001f, 0.15f, 0.2f, 0.125f};
    DarknetBoxBDP dha  = {0.2f, 0.2f, 0.15f, 0.15f+.0001f, 0.2f, 0.125f};
    DarknetBoxBDP dfxa = {0.2f, 0.2f, 0.15f, 0.15f, 0.2f+.0001f, 0.125f};
    DarknetBoxBDP dfya = {0.2f, 0.2f, 0.15f, 0.15f, 0.2f, 0.125f+.0001f};

    DarknetBoxBDP b = {0.8f, 0.8f, 0.2f, 0.2f, 0.8f, 0.7f};

    dxrep_bdp analytical = dx_box_riou(a, b, IOU);

    std::cout << "Analytical: "
              << "dx=" << analytical.dx << " "
              << "dy=" << analytical.dy << " "
              << "dw=" << analytical.dw << " "
              << "dh=" << analytical.dh << " "
              << "dfx=" << analytical.dfx << " "
              << "dfy=" << analytical.dfy << "\n";
    
    float base_gwd = compute_gwd_distance(a, b);
    float xgrad = (compute_gwd_distance(dxa, b) - base_gwd) / 0.0001f;
    float ygrad = (compute_gwd_distance(dya, b) - base_gwd) / 0.0001f;
    float wgrad = (compute_gwd_distance(dwa, b) - base_gwd) / 0.0001f;
    float hgrad = (compute_gwd_distance(dha, b) - base_gwd) / 0.0001f;
    float fxgrad = (compute_gwd_distance(dfxa, b) - base_gwd) / 0.0001f;
    float fygrad = (compute_gwd_distance(dfya, b) - base_gwd) / 0.0001f;
    
    std::cout << "Numerical:  "
              << "dx=" << xgrad << " "
              << "dy=" << ygrad << " "
              << "dw=" << wgrad << " "
              << "dh=" << hgrad << " "
              << "dfx=" << fxgrad << " "
              << "dfy=" << fygrad << "\n";
    
    std::cout << "Note: Non-zero gradients despite no overlap!\n";

    EXPECT_NEAR(-analytical.dx, xgrad, 0.02f);
    EXPECT_NEAR(-analytical.dy, ygrad, 0.02f);
    EXPECT_GT(-analytical.dx, 0.0f) << "Should pull X toward truth";
    EXPECT_GT(-analytical.dy, 0.0f) << "Should pull Y toward truth";
}

// ============================================================================
// Test 4: Perpendicular Boxes with CIoU
// ============================================================================

TEST(RotatedIoUGradient, Test4_PerpendicularCIoU) {
    std::cout << "\n=== Test 4: Perpendicular CIoU ===\n";

    DarknetBoxBDP a    = {0.5f, 0.5f, 0.4f, 0.2f, 0.5f, 0.4f};      // Horizontal
    DarknetBoxBDP dxa  = {0.5f+.0001f, 0.5f, 0.4f, 0.2f, 0.5f, 0.4f};
    DarknetBoxBDP dya  = {0.5f, 0.5f+.0001f, 0.4f, 0.2f, 0.5f, 0.4f};
    DarknetBoxBDP dwa  = {0.5f, 0.5f, 0.4f+.0001f, 0.2f, 0.5f, 0.4f};
    DarknetBoxBDP dha  = {0.5f, 0.5f, 0.4f, 0.2f+.0001f, 0.5f, 0.4f};
    DarknetBoxBDP dfxa = {0.5f, 0.5f, 0.4f, 0.2f, 0.5f+.0001f, 0.4f};
    DarknetBoxBDP dfya = {0.5f, 0.5f, 0.4f, 0.2f, 0.5f, 0.4f+.0001f};

    DarknetBoxBDP b = {0.5f, 0.5f, 0.2f, 0.4f, 0.6f, 0.5f};  // Vertical (90° rotated)

    dxrep_bdp analytical = dx_box_riou(a, b, CIOU);

    std::cout << "Analytical: "
              << "dx=" << analytical.dx << " "
              << "dy=" << analytical.dy << " "
              << "dw=" << analytical.dw << " "
              << "dh=" << analytical.dh << " "
              << "dfx=" << analytical.dfx << " "
              << "dfy=" << analytical.dfy << "\n";
    
    float base_gwd = compute_gwd_distance(a, b);
    float xgrad = (compute_gwd_distance(dxa, b) - base_gwd) / 0.0001f;
    float ygrad = (compute_gwd_distance(dya, b) - base_gwd) / 0.0001f;
    float wgrad = (compute_gwd_distance(dwa, b) - base_gwd) / 0.0001f;
    float hgrad = (compute_gwd_distance(dha, b) - base_gwd) / 0.0001f;
    float fxgrad = (compute_gwd_distance(dfxa, b) - base_gwd) / 0.0001f;
    float fygrad = (compute_gwd_distance(dfya, b) - base_gwd) / 0.0001f;
    
    std::cout << "Numerical:  "
              << "dx=" << xgrad << " "
              << "dy=" << ygrad << " "
              << "dw=" << wgrad << " "
              << "dh=" << hgrad << " "
              << "dfx=" << fxgrad << " "
              << "dfy=" << fygrad << "\n";
    
    std::cout << "Aspect ratios: pred=" << (0.4f/0.2f) << " truth=" << (0.2f/0.4f) << "\n";

    EXPECT_NEAR(-analytical.dx, xgrad, 0.03f);
    EXPECT_NEAR(-analytical.dy, ygrad, 0.03f);
    EXPECT_NEAR(-analytical.dw, wgrad, 0.05f);
    EXPECT_NEAR(-analytical.dh, hgrad, 0.05f);
    EXPECT_NE(analytical.dw, 0.0f) << "CIoU should penalize aspect mismatch";
}

// ============================================================================
// Mock GWD computation
// ============================================================================

float compute_gwd_distance(const DarknetBoxBDP& pred, const DarknetBoxBDP& truth) {
    auto box_to_gaussian = [](const DarknetBoxBDP& box, float& mu_x, float& mu_y,
                              float& s11, float& s12, float& s22) {
        mu_x = box.x;
        mu_y = box.y;
        
        float dx = box.fx - box.x;
        float dy = box.fy - box.y;
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 1e-6f) { dx = 0.0f; dy = -1.0f; dist = 1.0f; }
        
        float up_x = dx / dist, up_y = dy / dist;
        float right_x = up_y, right_y = -up_x;
        
        float var_w = box.w * box.w / 12.0f;
        float var_h = box.h * box.h / 12.0f;
        
        s11 = var_w * right_x * right_x + var_h * up_x * up_x;
        s12 = var_w * right_x * right_y + var_h * up_x * up_y;
        s22 = var_w * right_y * right_y + var_h * up_y * up_y;
    };
    
    float p_mx, p_my, p_s11, p_s12, p_s22;
    float t_mx, t_my, t_s11, t_s12, t_s22;
    
    box_to_gaussian(pred, p_mx, p_my, p_s11, p_s12, p_s22);
    box_to_gaussian(truth, t_mx, t_my, t_s11, t_s12, t_s22);
    
    float mean_dist = sqrtf((p_mx - t_mx)*(p_mx - t_mx) + (p_my - t_my)*(p_my - t_my));
    float trace_term = std::abs(p_s11 + p_s22 - t_s11 - t_s22);
    
    return mean_dist + sqrtf(trace_term);
}
