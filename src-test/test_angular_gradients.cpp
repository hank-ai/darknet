#include <gtest/gtest.h>
#include "darknet_internal.hpp"
#include <cmath>

// Tests for analytical gradients of cos(θ/2) angular correction term
// Validates dx_computeFrontPointCosine() implementation

/** Compute numerical gradient of cos(θ/2) using finite differences
 *
 * WHY: Provides ground truth for validating analytical gradients
 *
 * HOW: For each parameter, compute [f(x+ε) - f(x)] / ε
 *
 * @param pred Predicted BDP box
 * @param target Ground truth BDP box
 * @param epsilon Step size for finite differences (default 1e-5)
 * @return Numerical gradients {dx, dy, dw, dh, dfx, dfy}
 */
dxrep_bdp compute_numerical_cosine_gradient(const DarknetBoxBDP& pred, const DarknetBoxBDP& target, float epsilon = 1e-5f) {
	dxrep_bdp gradients = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	// Base value
	float base = computeFrontPointCosine(*(RectParams*)&pred, *(RectParams*)&target);

	// Gradient w.r.t. x
	{
		DarknetBoxBDP pred_plus = pred;
		pred_plus.x = pred.x + epsilon;
		float val_plus = computeFrontPointCosine(*(RectParams*)&pred_plus, *(RectParams*)&target);
		gradients.dx = (val_plus - base) / epsilon;
	}

	// Gradient w.r.t. y
	{
		DarknetBoxBDP pred_plus = pred;
		pred_plus.y = pred.y + epsilon;
		float val_plus = computeFrontPointCosine(*(RectParams*)&pred_plus, *(RectParams*)&target);
		gradients.dy = (val_plus - base) / epsilon;
	}

	// Gradient w.r.t. w
	{
		DarknetBoxBDP pred_plus = pred;
		pred_plus.w = pred.w + epsilon;
		float val_plus = computeFrontPointCosine(*(RectParams*)&pred_plus, *(RectParams*)&target);
		gradients.dw = (val_plus - base) / epsilon;
	}

	// Gradient w.r.t. h
	{
		DarknetBoxBDP pred_plus = pred;
		pred_plus.h = pred.h + epsilon;
		float val_plus = computeFrontPointCosine(*(RectParams*)&pred_plus, *(RectParams*)&target);
		gradients.dh = (val_plus - base) / epsilon;
	}

	// Gradient w.r.t. fx
	{
		DarknetBoxBDP pred_plus = pred;
		pred_plus.fx = pred.fx + epsilon;
		float val_plus = computeFrontPointCosine(*(RectParams*)&pred_plus, *(RectParams*)&target);
		gradients.dfx = (val_plus - base) / epsilon;
	}

	// Gradient w.r.t. fy
	{
		DarknetBoxBDP pred_plus = pred;
		pred_plus.fy = pred.fy + epsilon;
		float val_plus = computeFrontPointCosine(*(RectParams*)&pred_plus, *(RectParams*)&target);
		gradients.dfy = (val_plus - base) / epsilon;
	}

	return gradients;
}

// Test 1: Analytical gradients match numerical gradients
TEST(AngularGradients, MatchesNumericalGradients) {
	// Test multiple orientations
	std::vector<std::pair<DarknetBoxBDP, DarknetBoxBDP>> cases = {
		// Axis-aligned, close orientations
		{{0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f}, {0.55f, 0.52f, 0.25f, 0.18f, 0.65f, 0.52f}},
		// Partial rotation (~30°)
		{{0.5f, 0.5f, 0.3f, 0.2f, 0.65f, 0.55f}, {0.5f, 0.5f, 0.3f, 0.2f, 0.62f, 0.52f}},
		// Larger rotation (~60°)
		{{0.4f, 0.4f, 0.2f, 0.15f, 0.5f, 0.45f}, {0.4f, 0.4f, 0.2f, 0.15f, 0.43f, 0.53f}},
		// Near-perpendicular
		{{0.5f, 0.5f, 0.25f, 0.25f, 0.625f, 0.5f}, {0.5f, 0.5f, 0.25f, 0.25f, 0.5f, 0.625f}}
	};

	float tolerance = 0.01f;  // Analytical close to numerical (finite differences have ~1% error)

	for (size_t i = 0; i < cases.size(); i++) {
		auto [pred, truth] = cases[i];

		dxrep_bdp analytical = dx_computeFrontPointCosine(pred, truth);
		dxrep_bdp numerical = compute_numerical_cosine_gradient(pred, truth);

		EXPECT_NEAR(analytical.dx, numerical.dx, tolerance)
			<< "Case " << i << ": dx mismatch";
		EXPECT_NEAR(analytical.dy, numerical.dy, tolerance)
			<< "Case " << i << ": dy mismatch";
		EXPECT_NEAR(analytical.dw, numerical.dw, tolerance)
			<< "Case " << i << ": dw mismatch";
		EXPECT_NEAR(analytical.dh, numerical.dh, tolerance)
			<< "Case " << i << ": dh mismatch";
		EXPECT_NEAR(analytical.dfx, numerical.dfx, tolerance)
			<< "Case " << i << ": dfx mismatch";
		EXPECT_NEAR(analytical.dfy, numerical.dfy, tolerance)
			<< "Case " << i << ": dfy mismatch";
	}
}

// Test 2: Gradients w.r.t. w,h are always zero
TEST(AngularGradients, WidthHeightGradientsZero) {
	// Test various box configurations
	std::vector<std::pair<DarknetBoxBDP, DarknetBoxBDP>> cases = {
		{{0.5f, 0.5f, 0.1f, 0.1f, 0.55f, 0.5f}, {0.5f, 0.5f, 0.3f, 0.3f, 0.6f, 0.5f}},  // Different sizes
		{{0.5f, 0.5f, 0.4f, 0.2f, 0.65f, 0.55f}, {0.5f, 0.5f, 0.2f, 0.4f, 0.6f, 0.52f}}, // Aspect ratios
		{{0.3f, 0.7f, 0.05f, 0.05f, 0.325f, 0.7f}, {0.8f, 0.2f, 0.5f, 0.5f, 0.9f, 0.25f}} // Extreme difference
	};

	for (size_t i = 0; i < cases.size(); i++) {
		auto [pred, truth] = cases[i];
		dxrep_bdp grad = dx_computeFrontPointCosine(pred, truth);

		EXPECT_FLOAT_EQ(grad.dw, 0.0f) << "Case " << i << ": dw must be zero";
		EXPECT_FLOAT_EQ(grad.dh, 0.0f) << "Case " << i << ": dh must be zero";
	}
}

// Test 3: Gradient at perfect alignment (θ=0°, cos(θ/2)=1)
TEST(AngularGradients, PerfectAlignment) {
	// Same orientation → θ=0° → cos(θ/2)=1
	DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};
	DarknetBoxBDP truth = {0.55f, 0.52f, 0.25f, 0.18f, 0.65f, 0.52f};

	float cosine = computeFrontPointCosine(*(RectParams*)&pred, *(RectParams*)&truth);
	EXPECT_NEAR(cosine, 1.0f, 1e-3f) << "Aligned → cos(θ/2)=1";

	dxrep_bdp grad = dx_computeFrontPointCosine(pred, truth);

	// All gradients should be very small (near local maximum)
	EXPECT_LT(std::abs(grad.dx), 0.1f) << "dx small at alignment";
	EXPECT_LT(std::abs(grad.dy), 0.1f) << "dy small at alignment";
	EXPECT_LT(std::abs(grad.dfx), 0.1f) << "dfx small at alignment";
	EXPECT_LT(std::abs(grad.dfy), 0.1f) << "dfy small at alignment";
}

// Test 4: Gradient at opposite orientations (θ=180°, cos(θ/2)=0)
TEST(AngularGradients, OppositeOrientations) {
	// Front points in opposite directions
	DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};   // Right →
	DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.4f, 0.5f};  // Left ←

	float cosine = computeFrontPointCosine(*(RectParams*)&pred, *(RectParams*)&truth);
	EXPECT_NEAR(cosine, 0.0f, 1e-3f) << "Opposite → cos(θ/2)=0";

	dxrep_bdp grad = dx_computeFrontPointCosine(pred, truth);

	// Gradients should be zero (clamped due to singularity at θ=180°)
	EXPECT_FLOAT_EQ(grad.dx, 0.0f);
	EXPECT_FLOAT_EQ(grad.dy, 0.0f);
	EXPECT_FLOAT_EQ(grad.dfx, 0.0f);
	EXPECT_FLOAT_EQ(grad.dfy, 0.0f);
}

// Test 5: Gradient direction correctness
TEST(AngularGradients, GradientDirection) {
	DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.55f, 0.52f};  // Slight rotation
	DarknetBoxBDP truth = {0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.5f};   // Horizontal

	dxrep_bdp grad = dx_computeFrontPointCosine(pred, truth);

	// Moving fx toward truth.fx should increase cos(θ/2) → positive dfx
	// truth.fx=0.6, pred.fx=0.55, so gradient should be positive
	EXPECT_GT(grad.dfx, 0.0f) << "Gradient should point toward better alignment";

	// Moving fy toward truth.fy should increase cos(θ/2) → negative dfy
	// truth.fy=0.5, pred.fy=0.52, so gradient should be negative
	EXPECT_LT(grad.dfy, 0.0f) << "Gradient should point toward better alignment";
}

// Test 6: Zero gradient for degenerate case
TEST(AngularGradients, DegenerateCase) {
	// Front point equals center (zero-length vector)
	DarknetBoxBDP pred = {0.5f, 0.5f, 0.2f, 0.2f, 0.5f, 0.5f};  // fx=x, fy=y
	DarknetBoxBDP truth = {0.55f, 0.52f, 0.25f, 0.18f, 0.65f, 0.52f};

	float cosine = computeFrontPointCosine(*(RectParams*)&pred, *(RectParams*)&truth);
	EXPECT_FLOAT_EQ(cosine, 1.0f) << "Degenerate → cos(θ/2)=1";

	dxrep_bdp grad = dx_computeFrontPointCosine(pred, truth);

	// All gradients should be zero (function returns 1.0 constant)
	EXPECT_FLOAT_EQ(grad.dx, 0.0f);
	EXPECT_FLOAT_EQ(grad.dy, 0.0f);
	EXPECT_FLOAT_EQ(grad.dw, 0.0f);
	EXPECT_FLOAT_EQ(grad.dh, 0.0f);
	EXPECT_FLOAT_EQ(grad.dfx, 0.0f);
	EXPECT_FLOAT_EQ(grad.dfy, 0.0f);
}

// Test 7: Gradients are bounded and finite
TEST(AngularGradients, BoundedAndFinite) {
	// Test various random configurations
	std::vector<std::pair<DarknetBoxBDP, DarknetBoxBDP>> cases = {
		{{0.2f, 0.3f, 0.15f, 0.1f, 0.27f, 0.32f}, {0.7f, 0.6f, 0.2f, 0.2f, 0.8f, 0.65f}},
		{{0.8f, 0.2f, 0.3f, 0.25f, 0.95f, 0.25f}, {0.3f, 0.8f, 0.15f, 0.15f, 0.35f, 0.9f}},
		{{0.5f, 0.5f, 0.4f, 0.3f, 0.7f, 0.65f}, {0.5f, 0.5f, 0.3f, 0.4f, 0.35f, 0.7f}},
		{{0.1f, 0.1f, 0.08f, 0.08f, 0.14f, 0.11f}, {0.9f, 0.9f, 0.08f, 0.08f, 0.94f, 0.89f}}
	};

	for (size_t i = 0; i < cases.size(); i++) {
		auto [pred, truth] = cases[i];
		dxrep_bdp grad = dx_computeFrontPointCosine(pred, truth);

		// All gradients must be finite
		EXPECT_TRUE(std::isfinite(grad.dx)) << "Case " << i << ": dx must be finite";
		EXPECT_TRUE(std::isfinite(grad.dy)) << "Case " << i << ": dy must be finite";
		EXPECT_TRUE(std::isfinite(grad.dw)) << "Case " << i << ": dw must be finite";
		EXPECT_TRUE(std::isfinite(grad.dh)) << "Case " << i << ": dh must be finite";
		EXPECT_TRUE(std::isfinite(grad.dfx)) << "Case " << i << ": dfx must be finite";
		EXPECT_TRUE(std::isfinite(grad.dfy)) << "Case " << i << ": dfy must be finite";

		// Gradients should be reasonably bounded (not huge)
		// For normalized coordinates [0,1], gradients > 100 would be abnormal
		EXPECT_LT(std::abs(grad.dx), 100.0f) << "Case " << i << ": dx bounded";
		EXPECT_LT(std::abs(grad.dy), 100.0f) << "Case " << i << ": dy bounded";
		EXPECT_LT(std::abs(grad.dfx), 100.0f) << "Case " << i << ": dfx bounded";
		EXPECT_LT(std::abs(grad.dfy), 100.0f) << "Case " << i << ": dfy bounded";
	}
}
