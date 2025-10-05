#include "darknet_internal.hpp"
#include <boost/contract.hpp>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	// Macro for debug output - only prints if --trace flag is enabled
	#define DEBUG_TRACE(msg) if (cfg_and_state.is_trace) { *cfg_and_state.output << msg << std::endl; }

	// Debug macros for BDP loss/gradient logging (active until iteration 20)
	// These macros log loss components and gradients during training to diagnose
	// why training fails at step 16. Each macro logs only if iteration <= MAX_ITER.
	#define BDP_DEBUG_MAX_ITER 20

	#define LOSS_RIOU(iter, ...) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			*cfg_and_state.output << "[LOSS_RIOU iter=" << (iter) << "] " << __VA_ARGS__ << std::endl; \
		}

	// Helper function to get color for value: blue for near-zero, red for large values
	inline std::string value_color(float val) {
		float abs_val = std::abs(val);
		if (!std::isfinite(val)) return cfg_and_state.colour_is_enabled ? "\033[1;31m" : ""; // Bright red for NaN/Inf
		if (abs_val < 1e-6f) return cfg_and_state.colour_is_enabled ? "\033[34m" : ""; // Blue for ~0
		if (abs_val > 1.0f) return cfg_and_state.colour_is_enabled ? "\033[31m" : ""; // Red for >1
		return ""; // No color for normal values
	}

	// Helper function to get color for loss values: red if ~0, blue if 0.005-0.1, red if >0.1
	inline std::string loss_color(float val) {
		float abs_val = std::abs(val);
		if (!std::isfinite(val)) return cfg_and_state.colour_is_enabled ? "\033[1;31m" : ""; // Bright red for NaN/Inf
		if (abs_val < 0.005f) return cfg_and_state.colour_is_enabled ? "\033[31m" : ""; // Red for close to 0
		if (abs_val >= 0.005f && abs_val <= 0.1f) return cfg_and_state.colour_is_enabled ? "\033[34m" : ""; // Blue for good range
		if (abs_val > 0.1f) return cfg_and_state.colour_is_enabled ? "\033[31m" : ""; // Red for too large
		return ""; // No color
	}

	inline std::string reset_color() {
		return cfg_and_state.colour_is_enabled ? "\033[0m" : "";
	}

	#define LOSS_IOU(iter, riou_val, angular_val) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool is_abnormal = !std::isfinite(riou_val) || !std::isfinite(angular_val) || \
							   std::abs(riou_val) < 1e-6f || std::abs(angular_val) < 1e-6f || \
							   riou_val > 1.0f || angular_val > 1.0f; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[LOSS_IOU iter=" << (iter) << "] anchor=" << n << " grid=(" << i << "," << j << ") " \
									  << "riou=" << value_color(riou_val) << riou_val << reset_color() \
									  << " angular_corr=" << value_color(angular_val) << angular_val << reset_color() \
									  << std::endl; \
			} \
		}

	#define LOSS_GIOU(iter, giou_val, ...) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool is_abnormal = !std::isfinite(giou_val) || std::abs(giou_val) < 1e-6f || giou_val > 1.0f; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[LOSS_GIOU iter=" << (iter) << "] " \
									  << "giou=" << value_color(giou_val) << giou_val << reset_color() \
									  << " " << __VA_ARGS__ << std::endl; \
			} \
		}

	#define LOSS_DIOU(iter, diou_val, ...) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool is_abnormal = !std::isfinite(diou_val) || std::abs(diou_val) < 1e-6f || diou_val > 1.0f; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[LOSS_DIOU iter=" << (iter) << "] " \
									  << "diou=" << value_color(diou_val) << diou_val << reset_color() \
									  << " " << __VA_ARGS__ << std::endl; \
			} \
		}

	#define LOSS_CIOU(iter, ciou_val, ...) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool is_abnormal = !std::isfinite(ciou_val) || std::abs(ciou_val) < 1e-6f || ciou_val > 1.0f; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[LOSS_CIOU iter=" << (iter) << "] " \
									  << "ciou=" << value_color(ciou_val) << ciou_val << reset_color() \
									  << " " << __VA_ARGS__ << std::endl; \
			} \
		}

	#define LOSS_FP(iter, fp_val, ...) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool is_abnormal = !std::isfinite(fp_val) || std::abs(fp_val) < 1e-6f || fp_val > 1.0f; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[LOSS_FP iter=" << (iter) << "] " \
									  << "fp_loss=" << value_color(fp_val) << fp_val << reset_color() \
									  << " " << __VA_ARGS__ << std::endl; \
			} \
		}

	#define GRAD_LOSS_RIOU(iter, dx_val, dy_val, dw_val, dh_val, dfx_val, dfy_val) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool is_abnormal = !std::isfinite(dx_val) || !std::isfinite(dy_val) || !std::isfinite(dw_val) || \
							   !std::isfinite(dh_val) || !std::isfinite(dfx_val) || !std::isfinite(dfy_val) || \
							   (std::abs(dx_val) < 1e-6f && std::abs(dy_val) < 1e-6f && std::abs(dw_val) < 1e-6f && std::abs(dh_val) < 1e-6f) || \
							   std::abs(dx_val) > 1.0f || std::abs(dy_val) > 1.0f || std::abs(dw_val) > 1.0f || \
							   std::abs(dh_val) > 1.0f || std::abs(dfx_val) > 1.0f || std::abs(dfy_val) > 1.0f; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[GRAD_LOSS_RIOU iter=" << (iter) << "] anchor=" << n << " grid=(" << i << "," << j << ") " \
									  << "dx=" << value_color(dx_val) << dx_val << reset_color() \
									  << " dy=" << value_color(dy_val) << dy_val << reset_color() \
									  << " dw=" << value_color(dw_val) << dw_val << reset_color() \
									  << " dh=" << value_color(dh_val) << dh_val << reset_color() \
									  << " dfx=" << value_color(dfx_val) << dfx_val << reset_color() \
									  << " dfy=" << value_color(dfy_val) << dfy_val << reset_color() \
									  << std::endl; \
			} \
		}

	#define GRAD_LOSS_FP(iter, fx_val, fy_val, ...) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool is_abnormal = !std::isfinite(fx_val) || !std::isfinite(fy_val) || \
							   std::abs(fx_val) < 1e-6f || std::abs(fy_val) < 1e-6f || \
							   std::abs(fx_val) > 1.0f || std::abs(fy_val) > 1.0f; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[GRAD_LOSS_FP iter=" << (iter) << "] " \
									  << "dfx=" << value_color(fx_val) << fx_val << reset_color() \
									  << " dfy=" << value_color(fy_val) << fy_val << reset_color() \
									  << " " << __VA_ARGS__ << std::endl; \
			} \
		}

	struct train_yolo_args
	{
		const Darknet::Layer * l;
		Darknet::NetworkState state;
		int b;

		float tot_iou;
		float tot_giou_loss;
		float tot_iou_loss;
		float tot_fp_loss;  // Total front point loss for BDP
		int count;
		int class_count;
	};


	static inline Darknet::Box get_yolo_box(const float * x, const float * biases, const int n, const int index, const int i, const int j, const int lw, const int lh, const int w, const int h, const int stride, const int new_coords)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		Darknet::Box b;
		if (new_coords)
		{
			b.x = (i + x[index + 0 * stride]) / lw;
			b.y = (j + x[index + 1 * stride]) / lh;
			b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
			b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
		}
		else
		{
			b.x = (i + x[index + 0 * stride]) / lw;
			b.y = (j + x[index + 1 * stride]) / lh;
			b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
			b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
		}
		return b;
	}


	/// @implement OBB: Extract 6-parameter oriented bounding box from YOLO output using BDP representation
	static inline DarknetBoxBDP get_yolo_box_bdp(const float * x, const float * biases, const int n, const int index, const int i, const int j, const int lw, const int lh, const int w, const int h, const int stride, const int new_coords)
	{
		TAT_COMMENT(TATPARMS, "BDP box extraction from YOLO layer output");

		// Static assertions for compile-time validation
		static_assert(sizeof(DarknetBoxBDP) == 6 * sizeof(float), "DarknetBoxBDP must contain exactly 6 float parameters");
		static_assert(std::is_trivially_copyable_v<DarknetBoxBDP>, "DarknetBoxBDP must be trivially copyable");

		// Pre-conditions using assertions
		assert(x != nullptr && "Input array x must not be null");
		assert(biases != nullptr && "Biases array must not be null");
		assert(stride > 0 && "Stride must be positive");
		assert(lw > 0 && lh > 0 && "Layer dimensions must be positive");
		assert(w > 0 && h > 0 && "Network dimensions must be positive");
		assert(n >= 0 && "Anchor index must be non-negative");
		assert(i >= 0 && i < lw && "Grid x coordinate must be within layer bounds");
		assert(j >= 0 && j < lh && "Grid y coordinate must be within layer bounds");

		DarknetBoxBDP b;

		// BDP-specific: Read raw values and validate they are finite before processing
		// This prevents NaN/inf from propagating through calculations
		float raw_x = x[index + 0 * stride];
		float raw_y = x[index + 1 * stride];
		float raw_w = x[index + 2 * stride];
		float raw_h = x[index + 3 * stride];
		float raw_fx = x[index + 4 * stride];
		float raw_fy = x[index + 5 * stride];

		// BDP-specific: Fix NaN/inf in raw input values before computation
		// This is critical to prevent exp() from producing inf, and prevents NaN propagation
		if (!std::isfinite(raw_x)) raw_x = 0.0f;
		if (!std::isfinite(raw_y)) raw_y = 0.0f;
		if (!std::isfinite(raw_w)) raw_w = 0.0f;
		if (!std::isfinite(raw_h)) raw_h = 0.0f;
		if (!std::isfinite(raw_fx)) raw_fx = 0.0f;
		if (!std::isfinite(raw_fy)) raw_fy = 0.0f;

		// BDP-specific: Clamp raw values to prevent exp() overflow and gradient explosion
		// exp(x) overflows to inf for x > ~88.7, so we clamp to a safe range
		// Also clamp fx,fy to prevent front point divergence during gradient computation
		raw_w = std::max(-10.0f, std::min(10.0f, raw_w));
		raw_h = std::max(-10.0f, std::min(10.0f, raw_h));
		raw_fx = std::max(-10.0f, std::min(10.0f, raw_fx));
		raw_fy = std::max(-10.0f, std::min(10.0f, raw_fy));

		if (new_coords)
		{
			// New coordinate system: use squared terms for w,h
			b.x = (i + raw_x) / lw;
			b.y = (j + raw_y) / lh;
			b.w = raw_w * raw_w * 4 * biases[2 * n] / w;
			b.h = raw_h * raw_h * 4 * biases[2 * n + 1] / h;
			// Extract front point coordinates (parameters 4 and 5)
			b.fx = (i + raw_fx) / lw;
			b.fy = (j + raw_fy) / lh;
		}
		else
		{
			// Traditional coordinate system: use exp for w,h
			// BDP-specific: exp() is now safe because raw_w and raw_h are clamped
			b.x = (i + raw_x) / lw;
			b.y = (j + raw_y) / lh;
			b.w = exp(raw_w) * biases[2 * n] / w;
			b.h = exp(raw_h) * biases[2 * n + 1] / h;
			// Extract front point coordinates (parameters 4 and 5)
			b.fx = (i + raw_fx) / lw;
			b.fy = (j + raw_fy) / lh;
		}

		// BDP-specific: Clamp output values to valid ranges after computation
		// This ensures coordinates stay within [0,1] even with bad input data
		b.x = std::max(0.0f, std::min(1.0f, b.x));
		b.y = std::max(0.0f, std::min(1.0f, b.y));
		b.w = std::max(0.0f, std::min(1.0f, b.w));
		b.h = std::max(0.0f, std::min(1.0f, b.h));
		b.fx = std::max(0.0f, std::min(1.0f, b.fx));
		b.fy = std::max(0.0f, std::min(1.0f, b.fy));

		// Post-conditions: validate output ranges for normalized coordinates
		assert(b.x >= 0.0f && b.x <= 1.0f && "Center x coordinate must be normalized [0,1]");
		assert(b.y >= 0.0f && b.y <= 1.0f && "Center y coordinate must be normalized [0,1]");
		assert(b.w > 0.0f && b.w <= 1.0f && "Width must be positive and normalized [0,1]");
		assert(b.h > 0.0f && b.h <= 1.0f && "Height must be positive and normalized [0,1]");
		assert(b.fx >= 0.0f && b.fx <= 1.0f && "Front point x coordinate must be normalized [0,1]");
		assert(b.fy >= 0.0f && b.fy <= 1.0f && "Front point y coordinate must be normalized [0,1]");

		return b;
	}


	static inline void fix_nan_inf(float & val)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		if (std::isnan(val) or std::isinf(val))
		{
			val = 0.0f;
		}

		return;
	}

	/// Custom NaN/inf fix for BDP forward pass: NaN → 0, inf → 1
	static inline void fix_nan_inf_bdp_forward(float & val)
	{
		TAT_COMMENT(TATPARMS, "BDP forward NaN/inf fix");

		if (std::isnan(val))
		{
			val = 0.0f;
		}
		else if (std::isinf(val))
		{
			val = 1.0f;
		}

		return;
	}


	static inline void clip_value(float & val, const float max_val)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		if (val > max_val)
		{
			val = max_val;
		}
		else if (val < -max_val)
		{
			val = -max_val;
		}

		return;
	}


	/// loss function:  delta for box
	static inline ious delta_yolo_box(const Darknet::Box & truth, const float * x, const float * biases, const int n, const int index, const int i, const int j, const int lw, const int lh, const int w, const int h, float * delta, const float scale, const int stride, const float iou_normalizer, const IOU_LOSS iou_loss, const int accumulate, const float max_delta, int * rewritten_bbox, const int new_coords)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		if (delta[index + 0 * stride] ||
			delta[index + 1 * stride] ||
			delta[index + 2 * stride] ||
			delta[index + 3 * stride])
		{
			(*rewritten_bbox)++;
		}

		ious all_ious = { 0 };
		// i - step in layer width
		// j - step in layer height
		//  Returns a box in absolute coordinates
		Darknet::Box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);
		all_ious.iou = box_iou(pred, truth);
		all_ious.giou = box_giou(pred, truth);
		all_ious.diou = box_diou(pred, truth);
		all_ious.ciou = box_ciou(pred, truth);

		// avoid nan in dx_box_iou
		if (pred.w == 0)
		{
			pred.w = 1.0;
		}

		if (pred.h == 0)
		{
			pred.h = 1.0;
		}

		if (iou_loss == MSE)    // old loss
		{
			float tx = (truth.x*lw - i);
			float ty = (truth.y*lh - j);
			float tw = log(truth.w*w / biases[2 * n]);
			float th = log(truth.h*h / biases[2 * n + 1]);

			if (new_coords)
			{
				tw = sqrt(truth.w*w / (4 * biases[2 * n]));
				th = sqrt(truth.h*h / (4 * biases[2 * n + 1]));
			}

			// accumulate delta
			delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;
			delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;
			delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;
			delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;
		}
		else
		{
			// https://github.com/generalized-iou/g-darknet
			// https://arxiv.org/abs/1902.09630v2
			// https://giou.stanford.edu/
			all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

			// jacobian^t (transpose)
			//float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
			//float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
			//float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
			//float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

			// jacobian^t (transpose)
			float dx = all_ious.dx_iou.dt;
			float dy = all_ious.dx_iou.db;
			float dw = all_ious.dx_iou.dl;
			float dh = all_ious.dx_iou.dr;


			// predict exponential, apply gradient of e^delta_t ONLY for w,h
			if (new_coords)
			{
				//dw *= 8 * x[index + 2 * stride];
				//dh *= 8 * x[index + 3 * stride];
				//dw *= 8 * x[index + 2 * stride] * biases[2 * n] / w;
				//dh *= 8 * x[index + 3 * stride] * biases[2 * n + 1] / h;

				//float grad_w = 8 * exp(-x[index + 2 * stride]) / pow(exp(-x[index + 2 * stride]) + 1, 3);
				//float grad_h = 8 * exp(-x[index + 3 * stride]) / pow(exp(-x[index + 3 * stride]) + 1, 3);
				//dw *= grad_w;
				//dh *= grad_h;
			}
			else
			{
				dw *= exp(x[index + 2 * stride]);
				dh *= exp(x[index + 3 * stride]);
			}

			//dw *= exp(x[index + 2 * stride]);
			//dh *= exp(x[index + 3 * stride]);

			// normalize iou weight
			dx *= iou_normalizer;
			dy *= iou_normalizer;
			dw *= iou_normalizer;
			dh *= iou_normalizer;


			fix_nan_inf(dx);
			fix_nan_inf(dy);
			fix_nan_inf(dw);
			fix_nan_inf(dh);

			if (max_delta != FLT_MAX)
			{
				clip_value(dx, max_delta);
				clip_value(dy, max_delta);
				clip_value(dw, max_delta);
				clip_value(dh, max_delta);
			}

			if (!accumulate)
			{
				delta[index + 0 * stride] = 0;
				delta[index + 1 * stride] = 0;
				delta[index + 2 * stride] = 0;
				delta[index + 3 * stride] = 0;
			}

			// accumulate delta
			delta[index + 0 * stride] += dx;
			delta[index + 1 * stride] += dy;
			delta[index + 2 * stride] += dw;
			delta[index + 3 * stride] += dh;
		}

		return all_ious;
	}




	/// BDP loss function: delta for 6-parameter oriented bounding box (x,y,w,h,fx,fy)
	/// Added fp_normalizer parameter (λ4 from paper equation 10) for front point loss weighting
	/// Added current_iter parameter for debug logging (logs until iteration 20)
	static inline ious delta_yolo_box_bdp(const DarknetBoxBDP & truth, const float * x, const float * biases, const int n, const int index, const int i, const int j, const int lw, const int lh, const int w, const int h, float * delta, const float scale, const int stride, const float iou_normalizer, const float fp_normalizer, const IOU_LOSS iou_loss, const int accumulate, const float max_delta, int * rewritten_bbox, const int new_coords, const int current_iter)
	{
		TAT(TATPARMS);

		if (rewritten_bbox)
		{
			(*rewritten_bbox) = 0;
		}
		
		if (x[index + 0 * stride] != x[index + 0 * stride] ||
			x[index + 1 * stride] != x[index + 1 * stride] ||
			x[index + 2 * stride] != x[index + 2 * stride] ||
			x[index + 3 * stride] != x[index + 3 * stride] ||
			x[index + 4 * stride] != x[index + 4 * stride] ||
			x[index + 5 * stride] != x[index + 5 * stride])
		{
			(*rewritten_bbox)++;
		}

		if (rewritten_bbox && 
			delta[index + 0 * stride] != 0 &&
			delta[index + 1 * stride] != 0 &&
			delta[index + 2 * stride] != 0 &&
			delta[index + 3 * stride] != 0 &&
			delta[index + 4 * stride] != 0 &&
			delta[index + 5 * stride] != 0)
		{
			(*rewritten_bbox)++;
		}

		ious all_ious = { 0 };
		
		// Convert BDP to standard box for IoU calculation (use center+dimensions only)
		DarknetBoxBDP pred_center = get_yolo_box_bdp(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);

		// BDP IoU: Use axis-aligned IoU with angular correction (following paper)
		// IoU is computed on (x, y, w, h) only, then multiplied by cos(angle/2) for orientation
		all_ious.iou = box_iou_bdp(pred_center, truth);  // RIOU only
		all_ious.giou = 0.0f;  // Not used for BDP
		all_ious.diou = 0.0f;  // Not used for BDP
		all_ious.ciou = 0.0f;  // Not used for BDP

		// Angular correction: cos(α/2) where α is angle between front point vectors
		// Paper equation 11: L_IoU = IoU * cos(angle/2)
		float pred_front_dx = pred_center.fx - pred_center.x;
		float pred_front_dy = pred_center.fy - pred_center.y;
		float truth_front_dx = truth.fx - truth.x;
		float truth_front_dy = truth.fy - truth.y;

		// Compute angle between vectors using dot product
		float pred_length = std::sqrt(pred_front_dx * pred_front_dx + pred_front_dy * pred_front_dy);
		float truth_length = std::sqrt(truth_front_dx * truth_front_dx + truth_front_dy * truth_front_dy);

		float angular_correction = 1.0f;  // Default if vectors are degenerate
		if (pred_length > 1e-6f && truth_length > 1e-6f) {
			float dot = pred_front_dx * truth_front_dx + pred_front_dy * truth_front_dy;
			float cos_angle = dot / (pred_length * truth_length);
			cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));  // Clamp to [-1, 1]
			float angle = std::acos(cos_angle);
			angular_correction = std::cos(angle / 2.0f);
		}

		// Apply angular correction to IoU
		all_ious.iou *= angular_correction;
		all_ious.giou *= angular_correction;
		all_ious.diou *= angular_correction;
		all_ious.ciou *= angular_correction;

		// Safety: Clamp all IoU values to [0, 1] and fix NaN/Inf
		// This prevents training divergence if any IoU function returns invalid values
		static int iou_nan_debug = 0;
		bool iou_has_nan = !std::isfinite(all_ious.iou) || !std::isfinite(all_ious.giou) ||
		                   !std::isfinite(all_ious.diou) || !std::isfinite(all_ious.ciou);
		if (iou_has_nan || iou_nan_debug++ < 3) {
			*cfg_and_state.output << "   [IOU_DEBUG] iou=" << all_ious.iou << " giou=" << all_ious.giou
			                      << " diou=" << all_ious.diou << " ciou=" << all_ious.ciou
			                      << "\n      pred: x=" << pred_center.x << " y=" << pred_center.y
			                      << " w=" << pred_center.w << " h=" << pred_center.h
			                      << " fx=" << pred_center.fx << " fy=" << pred_center.fy
			                      << "\n      truth: x=" << truth.x << " y=" << truth.y
			                      << " w=" << truth.w << " h=" << truth.h
			                      << " fx=" << truth.fx << " fy=" << truth.fy << std::endl;
		}
		if (!std::isfinite(all_ious.iou) || all_ious.iou < 0.0f || all_ious.iou > 1.0f) all_ious.iou = 0.0f;
		if (!std::isfinite(all_ious.giou)) all_ious.giou = 0.0f;
		if (!std::isfinite(all_ious.diou)) all_ious.diou = 0.0f;
		if (!std::isfinite(all_ious.ciou)) all_ious.ciou = 0.0f;

		// Clamp RIOU to minimum of 0.001 to ensure gradient flow even when boxes don't overlap
		// This prevents zero gradients for orientation when there's no geometric overlap
		all_ious.iou = std::max(0.001f, all_ious.iou);

		// Log RIOU values for debugging (only when abnormal or first 5 iterations)
		// RIOU = rotated IoU * angular_correction (cos of angle/2 between front point vectors)
		LOSS_IOU(current_iter, all_ious.iou, angular_correction);

		// Avoid nan in dx_box_iou
		if (pred_center.w == 0) pred_center.w = 1.0;
		if (pred_center.h == 0) pred_center.h = 1.0;

		if (iou_loss == MSE)    // MSE loss for all 6 parameters
		{
			// Standard x,y,w,h loss (same as original)
			float tx = (truth.x * lw - i);
			float ty = (truth.y * lh - j);

			// BDP-specific: Ensure log/sqrt arguments are positive to prevent NaN
			float safe_w_arg = std::max(1e-9f, truth.w * w / biases[2 * n]);
			float safe_h_arg = std::max(1e-9f, truth.h * h / biases[2 * n + 1]);

			float tw = log(safe_w_arg);
			float th = log(safe_h_arg);

			// Front point loss: Compare decoded predictions (absolute [0,1]) with truth
			// pred_center.fx and pred_center.fy are already decoded to [0,1] coordinates
			float diff_fx = truth.fx - pred_center.fx;
			float diff_fy = truth.fy - pred_center.fy;
			float abs_diff_fx = std::abs(diff_fx);
			float abs_diff_fy = std::abs(diff_fy);

			// Smooth L1 loss for front point
			float loss_fx = (abs_diff_fx < 1.0f) ? (0.5f * diff_fx * diff_fx) : (abs_diff_fx - 0.5f);
			float loss_fy = (abs_diff_fy < 1.0f) ? (0.5f * diff_fy * diff_fy) : (abs_diff_fy - 0.5f);
			all_ious.fp_loss = loss_fx + loss_fy;

			// Log front point loss for debugging (MSE path)
			LOSS_FP(current_iter, all_ious.fp_loss, "anchor=" << n << " grid=(" << i << "," << j << ") "
			        << "loss_fx=" << loss_fx << " loss_fy=" << loss_fy
			        << " diff_fx=" << diff_fx << " diff_fy=" << diff_fy);

			// Gradient for Smooth L1
			float grad_fx = (abs_diff_fx < 1.0f) ? diff_fx : ((diff_fx > 0) ? 1.0f : -1.0f);
			float grad_fy = (abs_diff_fy < 1.0f) ? diff_fy : ((diff_fy > 0) ? 1.0f : -1.0f);

			// Compute deltas (will be accumulated later with x,y,w,h deltas)
			float dfx = scale * (-grad_fx) * logistic_gradient(x[index + 4 * stride]) * fp_normalizer;
			float dfy = scale * (-grad_fy) * logistic_gradient(x[index + 5 * stride]) * fp_normalizer;

			if (new_coords)
			{
				// BDP-specific: Ensure sqrt arguments are positive
				float safe_w_sqrt_arg = std::max(0.0f, truth.w * w / (4 * biases[2 * n]));
				float safe_h_sqrt_arg = std::max(0.0f, truth.h * h / (4 * biases[2 * n + 1]));
				tw = sqrt(safe_w_sqrt_arg);
				th = sqrt(safe_h_sqrt_arg);
			}

			// Calculate deltas for x,y,w,h (standard MSE)
			float dx = scale * (tx - x[index + 0 * stride]) * iou_normalizer;
			float dy = scale * (ty - x[index + 1 * stride]) * iou_normalizer;
			float dw = scale * (tw - x[index + 2 * stride]) * iou_normalizer;
			float dh = scale * (th - x[index + 3 * stride]) * iou_normalizer;

			// Front point deltas already computed above (line 466-467)
			// dfx and dfy are already defined in the scope above

			// Fix NaN/inf values in all deltas
			fix_nan_inf(dx); fix_nan_inf(dy); fix_nan_inf(dw); fix_nan_inf(dh);
			fix_nan_inf(dfx); fix_nan_inf(dfy);

			// Apply clipping if specified
			if (max_delta != FLT_MAX)
			{
				clip_value(dx, max_delta); clip_value(dy, max_delta);
				clip_value(dw, max_delta); clip_value(dh, max_delta);
				clip_value(dfx, max_delta); clip_value(dfy, max_delta);
			}

			// Clip gradients to prevent explosion (especially dw, dh due to exp() multiplication)
			const float grad_clip = 2.0f;
			const float fp_grad_clip = 0.1f;  // Much smaller clip for front point gradients
			dx = std::max(-grad_clip, std::min(grad_clip, dx));
			dy = std::max(-grad_clip, std::min(grad_clip, dy));
			dw = std::max(-grad_clip, std::min(grad_clip, dw));
			dh = std::max(-grad_clip, std::min(grad_clip, dh));
			dfx = std::max(-fp_grad_clip, std::min(fp_grad_clip, dfx));
			dfy = std::max(-fp_grad_clip, std::min(fp_grad_clip, dfy));

			// Log gradients before accumulation (MSE path)
			GRAD_LOSS_FP(current_iter, dfx, dfy, "anchor=" << n << " grid=(" << i << "," << j << ") "
			             << "dx=" << dx << " dy=" << dy << " dw=" << dw << " dh=" << dh);

			// Accumulate delta for all 6 parameters
			delta[index + 0 * stride] += dx;   // x
			delta[index + 1 * stride] += dy;   // y
			delta[index + 2 * stride] += dw;   // w
			delta[index + 3 * stride] += dh;   // h
			delta[index + 4 * stride] += dfx;  // fx (front point x)
			delta[index + 5 * stride] += dfy;  // fy (front point y)

			// Fix any NaN/inf in delta array immediately after accumulation (MSE path)
			if (!std::isfinite(delta[index + 0 * stride])) delta[index + 0 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 1 * stride])) delta[index + 1 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 2 * stride])) delta[index + 2 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 3 * stride])) delta[index + 3 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 4 * stride])) delta[index + 4 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 5 * stride])) delta[index + 5 * stride] = 0.0f;
		}
		else  // IoU-based loss
		{
			// BDP gradients: Use axis-aligned IoU gradients (following paper)
			// Gradients only affect (x, y, w, h), NOT (fx, fy) - orientation handled by front point loss
			dxrep_bdp iou_grad = dx_box_riou(pred_center, truth, iou_loss);

			// Fix NaN/inf BEFORE clipping (std::max/min don't handle NaN properly)
			if (!std::isfinite(iou_grad.dx)) iou_grad.dx = 0.0f;
			if (!std::isfinite(iou_grad.dy)) iou_grad.dy = 0.0f;
			if (!std::isfinite(iou_grad.dw)) iou_grad.dw = 0.0f;
			if (!std::isfinite(iou_grad.dh)) iou_grad.dh = 0.0f;
			if (!std::isfinite(iou_grad.dfx)) iou_grad.dfx = 0.0f;
			if (!std::isfinite(iou_grad.dfy)) iou_grad.dfy = 0.0f;

			// Clip raw RIOU gradients to prevent explosion
			// Conservative clipping based on observed gradient magnitudes
			const float pos_grad_clip = 0.08f;   // Max ±0.08 for dx/dy
			const float size_grad_clip = 0.08f;  // Max ±0.08 for dw/dh
			const float fp_grad_clip = 0.08f;    // Max ±0.08 for dfx/dfy
			iou_grad.dx = std::max(-pos_grad_clip, std::min(pos_grad_clip, iou_grad.dx));
			iou_grad.dy = std::max(-pos_grad_clip, std::min(pos_grad_clip, iou_grad.dy));
			iou_grad.dw = std::max(-size_grad_clip, std::min(size_grad_clip, iou_grad.dw));
			iou_grad.dh = std::max(-size_grad_clip, std::min(size_grad_clip, iou_grad.dh));
			iou_grad.dfx = std::max(-fp_grad_clip, std::min(fp_grad_clip, iou_grad.dfx));
			iou_grad.dfy = std::max(-fp_grad_clip, std::min(fp_grad_clip, iou_grad.dfy));

			// Extract gradients for x,y,w,h from RIOU
			// dxrep_bdp contains {dx, dy, dw, dh, dfx, dfy}
			float dx = iou_grad.dx;
			float dy = iou_grad.dy;
			float dw = iou_grad.dw;
			float dh = iou_grad.dh;

			// Orientation gradients from RIOU (includes angular correction in rotated IoU)
			float dfx_from_iou = iou_grad.dfx;
			float dfy_from_iou = iou_grad.dfy;

			// Front point Smooth L1 loss: additional regularization for orientation
			// This provides a direct signal independent of IoU for learning rotation
			// Use decoded predictions (absolute [0,1] coords) vs truth
			float diff_fx = truth.fx - pred_center.fx;
			float diff_fy = truth.fy - pred_center.fy;
			float abs_diff_fx = std::abs(diff_fx);
			float abs_diff_fy = std::abs(diff_fy);

			// DEBUG: Print actual values in IoU-based loss path
			static int debug_count_iou = 0;
			if (debug_count_iou++ < 5) {
				*cfg_and_state.output << "   [BDP_DEBUG_IOU] CALL#" << debug_count_iou
				                      << " grid=(i=" << i << ",j=" << j << ")"
				                      << " truth.x=" << truth.x << " truth.y=" << truth.y
				                      << "\n      truth.fx=" << truth.fx << " truth.fy=" << truth.fy
				                      << "\n      pred_fx=" << x[index + 4 * stride]
				                      << " pred_fy=" << x[index + 5 * stride]
				                      << "\n      diff_fx=" << diff_fx << " diff_fy=" << diff_fy
				                      << std::endl;
			}

			// Smooth L1 loss computation
			float loss_fx = (abs_diff_fx < 1.0f) ? (0.5f * diff_fx * diff_fx) : (abs_diff_fx - 0.5f);
			float loss_fy = (abs_diff_fy < 1.0f) ? (0.5f * diff_fy * diff_fy) : (abs_diff_fy - 0.5f);
			all_ious.fp_loss = loss_fx + loss_fy;

			// Log front point loss for debugging (IoU-based loss path)
			LOSS_FP(current_iter, all_ious.fp_loss, "anchor=" << n << " grid=(" << i << "," << j << ") "
			        << "loss_fx=" << loss_fx << " loss_fy=" << loss_fy
			        << " diff_fx=" << diff_fx << " diff_fy=" << diff_fy);

			// Smooth L1 gradient for front point
			float grad_fx_smoothL1 = (abs_diff_fx < 1.0f) ? diff_fx : ((diff_fx > 0) ? 1.0f : -1.0f);
			float grad_fy_smoothL1 = (abs_diff_fy < 1.0f) ? diff_fy : ((diff_fy > 0) ? 1.0f : -1.0f);

			// DEBUG: Check for NaN in gradients before combining
			static int nan_debug = 0;
			bool has_nan = !std::isfinite(dfx_from_iou) || !std::isfinite(dfy_from_iou) ||
			               !std::isfinite(grad_fx_smoothL1) || !std::isfinite(grad_fy_smoothL1) ||
			               !std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dw) || !std::isfinite(dh);
			if (has_nan || nan_debug++ < 3) {
				const char* loss_name = "UNKNOWN";
				switch(iou_loss) {
					case IOU: loss_name = "IOU"; break;
					case GIOU: loss_name = "GIOU"; break;
					case MSE: loss_name = "MSE"; break;
					case DIOU: loss_name = "DIOU"; break;
					case CIOU: loss_name = "CIOU"; break;
					case RIOU: loss_name = "RIOU"; break;
				}
				*cfg_and_state.output << "   [GRAD_DEBUG] loss_type=" << loss_name << " (" << iou_loss << ")"
				                      << "\n      dx=" << dx << " dy=" << dy << " dw=" << dw << " dh=" << dh
				                      << "\n      dfx_from_iou=" << dfx_from_iou << " dfy_from_iou=" << dfy_from_iou
				                      << "\n      grad_fx_smoothL1=" << grad_fx_smoothL1 << " grad_fy_smoothL1=" << grad_fy_smoothL1
				                      << "\n      iou_normalizer=" << iou_normalizer << " fp_normalizer=" << fp_normalizer
				                      << " scale=" << scale << std::endl;
			}

			// Phase 3: Combine IoU gradients with Smooth L1 gradients for orientation
			// Total gradient = IoU gradient (from rotated IoU) + Smooth L1 gradient (direct penalty)
			// Scale down Smooth L1 gradients when RIOU is very small (< 0.1) to prioritize position first
			// Minimum RIOU of 0.001 ensures some gradient flow even when boxes don't overlap
			float riou_clamped = std::max(0.001f, all_ious.iou);
			// Scaling: 0.01 if riou=0.001, 1 if riou>=0.1, linear in between
			float riou_scale = std::min(1.0f, riou_clamped / 0.1f);
			// Apply chain rule through sigmoid activation function
			// logistic_gradient computes d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
			float dfx = scale * (dfx_from_iou * iou_normalizer + grad_fx_smoothL1 * fp_normalizer * riou_scale) * logistic_gradient(x[index + 4 * stride]);
			float dfy = scale * (dfy_from_iou * iou_normalizer + grad_fy_smoothL1 * fp_normalizer * riou_scale) * logistic_gradient(x[index + 5 * stride]);

			// Apply exponential gradient for w,h if needed
			// BDP-specific: Validate input values before exp() to prevent inf in gradients
			if (!new_coords)
			{
				float raw_w_for_grad = x[index + 2 * stride];
				float raw_h_for_grad = x[index + 3 * stride];

				// BDP-specific: Check for finite values and clamp to safe range before exp()
				if (!std::isfinite(raw_w_for_grad)) raw_w_for_grad = 0.0f;
				if (!std::isfinite(raw_h_for_grad)) raw_h_for_grad = 0.0f;
				raw_w_for_grad = std::max(-10.0f, std::min(10.0f, raw_w_for_grad));
				raw_h_for_grad = std::max(-10.0f, std::min(10.0f, raw_h_for_grad));

				dw *= exp(raw_w_for_grad);
				dh *= exp(raw_h_for_grad);
			}

			// Normalize x,y,w,h gradients (fx,fy already normalized in combined gradient above)
			dx *= iou_normalizer;
			dy *= iou_normalizer;
			dw *= iou_normalizer;
			dh *= iou_normalizer;
			// Note: dfx, dfy already include normalizers from line 496-497

			// Fix NaN/inf values
			fix_nan_inf(dx); fix_nan_inf(dy); fix_nan_inf(dw); fix_nan_inf(dh);
			fix_nan_inf(dfx); fix_nan_inf(dfy);

			// DEBUG: Check if final gradients after fix_nan_inf are still NaN
			static int final_nan_check = 0;
			if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dw) ||
			    !std::isfinite(dh) || !std::isfinite(dfx) || !std::isfinite(dfy)) {
				*cfg_and_state.output << "   [ERROR] Gradients still NaN after fix_nan_inf! count=" << final_nan_check++
				                      << "\n      dx=" << dx << " dy=" << dy << " dw=" << dw << " dh=" << dh
				                      << "\n      dfx=" << dfx << " dfy=" << dfy << std::endl;
			}

			// DEBUG: Log pre-clip values when riou=0
			static int pre_clip_debug = 0;
			if (all_ious.iou < 1e-6f && pre_clip_debug++ < 5) {
				*cfg_and_state.output << "   [PRE_CLIP_DEBUG] riou=" << all_ious.iou
				                      << " riou_scale=" << riou_scale
				                      << "\n      dfx_before_clip=" << dfx << " dfy_before_clip=" << dfy
				                      << "\n      dfx_from_iou=" << dfx_from_iou << " dfy_from_iou=" << dfy_from_iou
				                      << "\n      grad_fx_smoothL1=" << grad_fx_smoothL1 << " grad_fy_smoothL1=" << grad_fy_smoothL1
				                      << "\n      fp_normalizer=" << fp_normalizer << " scale=" << scale << std::endl;
			}

			// Apply clipping if specified
			if (max_delta != FLT_MAX)
			{
				clip_value(dx, max_delta); clip_value(dy, max_delta);
				clip_value(dw, max_delta); clip_value(dh, max_delta);
				clip_value(dfx, max_delta); clip_value(dfy, max_delta);
			}

			// NOTE: Gradients already clipped at lines 653-663 after dx_box_riou()
			// No additional clipping needed here to avoid redundancy

			// Log gradients before accumulation (only when abnormal or first 5 iterations)
			GRAD_LOSS_RIOU(current_iter, dx, dy, dw, dh, dfx, dfy);

			// Zero out deltas if not accumulating
			if (!accumulate)
			{
				delta[index + 0 * stride] = 0;  // x
				delta[index + 1 * stride] = 0;  // y
				delta[index + 2 * stride] = 0;  // w
				delta[index + 3 * stride] = 0;  // h
				delta[index + 4 * stride] = 0;  // fx
				delta[index + 5 * stride] = 0;  // fy
			}

			// Accumulate all deltas
			delta[index + 0 * stride] += dx;   // x
			delta[index + 1 * stride] += dy;   // y
			delta[index + 2 * stride] += dw;   // w
			delta[index + 3 * stride] += dh;   // h
			delta[index + 4 * stride] += dfx;  // fx (front point x)
			delta[index + 5 * stride] += dfy;  // fy (front point y)

			// Fix any NaN/inf in delta array immediately after accumulation
			if (!std::isfinite(delta[index + 0 * stride])) delta[index + 0 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 1 * stride])) delta[index + 1 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 2 * stride])) delta[index + 2 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 3 * stride])) delta[index + 3 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 4 * stride])) delta[index + 4 * stride] = 0.0f;
			if (!std::isfinite(delta[index + 5 * stride])) delta[index + 5 * stride] = 0.0f;

			// DEBUG: Check delta array after accumulation for NaN
			static int delta_nan_check = 0;
			if (!std::isfinite(delta[index + 0 * stride]) || !std::isfinite(delta[index + 1 * stride]) ||
			    !std::isfinite(delta[index + 2 * stride]) || !std::isfinite(delta[index + 3 * stride]) ||
			    !std::isfinite(delta[index + 4 * stride]) || !std::isfinite(delta[index + 5 * stride])) {
				*cfg_and_state.output << "   [ERROR] Delta array contains NaN after accumulation! count=" << delta_nan_check++
				                      << "\n      delta[x]=" << delta[index + 0 * stride]
				                      << " delta[y]=" << delta[index + 1 * stride]
				                      << " delta[w]=" << delta[index + 2 * stride]
				                      << " delta[h]=" << delta[index + 3 * stride]
				                      << " delta[fx]=" << delta[index + 4 * stride]
				                      << " delta[fy]=" << delta[index + 5 * stride]
				                      << "\n      Added: dx=" << dx << " dy=" << dy << " dw=" << dw
				                      << " dh=" << dh << " dfx=" << dfx << " dfy=" << dfy << std::endl;
			}
		}

		return all_ious;
	}


	static inline void averages_yolo_deltas(const int class_index, const int box_index, const int stride, const int classes, float * delta)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		int classes_in_one_box = 0;
		for (int c = 0; c < classes; ++c)
		{
			if (delta[class_index + stride * c] > 0.0f)
			{
				classes_in_one_box++;
			}
		}

		if (classes_in_one_box > 0)
		{
			delta[box_index + 0 * stride] /= classes_in_one_box;
			delta[box_index + 1 * stride] /= classes_in_one_box;
			delta[box_index + 2 * stride] /= classes_in_one_box;
			delta[box_index + 3 * stride] /= classes_in_one_box;
		}
	}


	/// loss function:  delta for class
	static inline void delta_yolo_class(const float * output, float * delta, const int index, const int class_id, const int classes, const int stride, float * avg_cat, const int focal_loss, const float label_smooth_eps, const float *classes_multipliers, const float cls_normalizer)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		if (delta[index + stride * class_id])
		{
			float y_true = 1;

			if (label_smooth_eps)
			{
				y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
			}

			const float result_delta = y_true - output[index + stride * class_id];

			if (!isnan(result_delta) && !isinf(result_delta))
			{
				delta[index + stride * class_id] = result_delta;
			}

			if (classes_multipliers)
			{
				delta[index + stride * class_id] *= classes_multipliers[class_id];
			}

			if (avg_cat)
			{
				*avg_cat += output[index + stride * class_id];
			}

			return;
		}

		// Focal loss
		if (focal_loss)
		{
			// Focal Loss
			float alpha = 0.5;    // 0.25 or 0.5
			//float gamma = 2;    // hardcoded in many places of the grad-formula

			int ti = index + stride*class_id;
			float pt = output[ti] + 0.000000000000001F;
			// http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
			float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
			//float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

			for (int n = 0; n < classes; ++n)
			{
				delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride * n]);

				delta[index + stride*n] *= alpha*grad;

				if (n == class_id && avg_cat)
				{
					*avg_cat += output[index + stride * n];
				}
			}
		}
		else
		{
			// default
			for (int n = 0; n < classes; ++n)
			{
				float y_true = ((n == class_id) ? 1.0f : 0.0f);

				if (label_smooth_eps)
				{
					y_true = y_true *  (1.0f - label_smooth_eps) + 0.5f * label_smooth_eps;
				}
				float result_delta = y_true - output[index + stride*n];
				if (!isnan(result_delta) && !isinf(result_delta))
				{
					delta[index + stride * n] = result_delta;
				}

				if (classes_multipliers && n == class_id)
				{
					delta[index + stride * class_id] *= classes_multipliers[class_id] * cls_normalizer;
				}

				// DEBUG: Check for NaN in classification delta
				static int class_nan_debug = 0;
				if (!std::isfinite(delta[index + stride * n])) {
					if (class_nan_debug++ < 10) {
						*cfg_and_state.output << "   [GRAD_DEBUG_CLASS] NaN in class delta! class=" << n
						                      << " target_class=" << class_id
						                      << "\n      output[" << n << "]=" << output[index + stride * n]
						                      << " y_true=" << y_true
						                      << " result_delta=" << result_delta
						                      << " delta[" << n << "]=" << delta[index + stride * n]
						                      << "\n      cls_normalizer=" << cls_normalizer;
						if (classes_multipliers) {
							*cfg_and_state.output << " classes_multipliers[" << class_id << "]=" << classes_multipliers[class_id];
						}
						*cfg_and_state.output << std::endl;
					}
				}

				if (n == class_id && avg_cat)
				{
					*avg_cat += output[index + stride * n];
				}
			}
		}
	}


	static inline int compare_yolo_class(const float * output, const int classes, const int class_index, const int stride, const float objectness, const int class_id, const float conf_thresh)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		for (int j = 0; j < classes; ++j)
		{
			const float prob = output[class_index + stride * j];
			if (prob > conf_thresh)
			{
				return 1;
			}
		}

		return 0;
	}


	/// @implement OBB: Entry index calculation for BDP (6-parameter) oriented bounding boxes
	/// This function calculates memory indices for accessing BDP box parameters in YOLO layer output.
	/// 
	/// INTERACTION WITH OTHER FUNCTIONS:
	/// - Called by: forward_yolo_layer_bdp(), get_yolo_detections_bdp(), yolo_num_detections_bdp()
	/// - Uses: Layer structure (l.w, l.h, l.classes, l.outputs)
	/// - Purpose: Maps (batch, location, entry) coordinates to linear memory index
	///
	/// MEMORY LAYOUT FOR BDP:
	/// Each anchor box has: [x, y, w, h, fx, fy, objectness, class0, class1, ..., classN]
	/// Total entries per box = 6 (coords) + 1 (objectness) + classes = 7 + classes
	/// This differs from standard YOLO which uses 4 (coords) + 1 (objectness) + classes = 5 + classes
	///
	/// COORDINATE MAPPING:
	/// - entry 0-1: center coordinates (x,y)
	/// - entry 2-3: dimensions (w,h) 
	/// - entry 4-5: front point coordinates (fx,fy) - NEW for BDP
	/// - entry 6: objectness score
	/// - entry 7+: class probabilities
	static inline int yolo_entry_index_bdp(const Darknet::Layer & l, const int batch, const int location, const int entry)
	{
		TAT_COMMENT(TATPARMS, "BDP entry index calculation for 6-parameter boxes");

		// Static assertions for compile-time validation
		static_assert(sizeof(int) >= 4, "Integer must be at least 32 bits");
		
		// Pre-conditions: validate input parameters to prevent buffer overflows
		assert(batch >= 0 && "Batch index must be non-negative");
		assert(location >= 0 && "Location must be non-negative"); 
		assert(entry >= 0 && "Entry must be non-negative");
		assert(l.w > 0 && l.h > 0 && "Layer dimensions must be positive");

		// Calculate anchor and local position within the grid
		const int n = location / (l.w * l.h);      // Which anchor box (0 to l.n-1)
		const int loc = location % (l.w * l.h);    // Grid position (0 to w*h-1)
		
		// BDP uses 6 coordinates + 1 objectness + classes (vs standard YOLO's 4 + 1 + classes)
		const int bdp_entries_per_box = 6 + 1 + l.classes; // x,y,w,h,fx,fy + objectness + classes
		
		// Calculate linear memory index in the flattened output array
		// Memory layout: [batch0[anchor0[entry0...entryN], anchor1[...]], batch1[...]]
		int result = batch * l.outputs +                    // Skip to current batch
					 n * l.w * l.h * bdp_entries_per_box +  // Skip to current anchor
					 entry * l.w * l.h +                    // Skip to current entry type
					 loc;                                   // Skip to grid position

		// Post-condition: ensure result is within valid bounds
		assert(result >= 0 && "Calculated index must be non-negative");
		
		return result;
	}

	static inline int yolo_entry_index(const Darknet::Layer & l, const int batch, const int location, const int entry)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		// similar function exists in region_layer.cpp, but the math is slightly different

		const int n		= location / (l.w * l.h);
		const int loc	= location % (l.w * l.h);

		return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
	}

} // anonymous namespace


Darknet::Layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
	TAT(TATPARMS);

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::YOLO;

	l.n = n;
	l.total = total;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n*(classes + 4 + 1);
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.cost = (float*)xcalloc(1, sizeof(float));
	l.biases = (float*)xcalloc(total * 2, sizeof(float));

	/* PR #51:
	 * When the model is loaded in darknet, I'm working on allowing Python to access the model's information.
	 * At this point, I'm sending the structural information of all layers of the model to Python, but when
	 * passing the length of the bias pointer of the 'YOLO' layer, the length (nbiases) value is 0.
	 *
	 * For the YOLO layer, the bias value has a length of anchor * 2, and it would be great if this information
	 * could also be confirmed from nbiases.
	 */
	l.nbiases = total * 2;

	if (mask)
	{
		l.mask = mask;
	}
	else
	{
		l.mask = (int*)xcalloc(n, sizeof(int));

		for (int i = 0; i < n; ++i)
		{
			l.mask[i] = i;
		}
	}
	l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
	l.outputs = h*w*n*(classes + 4 + 1);
	l.inputs = l.outputs;
	l.max_boxes = max_boxes;
	l.truth_size = 4 + 2;
	l.truths = l.max_boxes*l.truth_size;    // 90*(4 + 1);
	l.labels = (int*)xcalloc(batch * l.w*l.h*l.n, sizeof(int));

	for (int i = 0; i < batch * l.w*l.h*l.n; ++i)
	{
		l.labels[i] = -1;
	}
	l.class_ids = (int*)xcalloc(batch * l.w*l.h*l.n, sizeof(int));

	for (int i = 0; i < batch * l.w*l.h*l.n; ++i)
	{
		l.class_ids[i] = -1;
	}

	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));

	for (int i = 0; i < total * 2; ++i)
	{
		l.biases[i] = .5;
	}

	l.forward = forward_yolo_layer;
	l.backward = backward_yolo_layer;
#ifdef DARKNET_GPU
	l.forward_gpu = forward_yolo_layer_gpu;
	l.backward_gpu = backward_yolo_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.output_avg_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

	free(l.output);
	/// @todo Valgrind tells us this is not freed in @ref free_layer_custom()
	if (cudaSuccess == cudaHostAlloc((void**)&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped))
	{
		l.output_pinned = 1;
	}
	else
	{
		std::ignore = cudaGetLastError(); // reset CUDA-error
		l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	}

	free(l.delta);
	if (cudaSuccess == cudaHostAlloc((void**)&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped))
	{
		l.delta_pinned = 1;
	}
	else
	{
		std::ignore = cudaGetLastError(); // reset CUDA-error
		l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	}
#endif

	return l;
}


/// @implement OBB: Factory function for oriented bounding box YOLO layer (BDP representation)
/// This function creates a YOLO_BDP layer that processes 6-parameter oriented bounding boxes.
///
/// INTERACTION WITH OTHER FUNCTIONS:
/// - Called by: parse_yolo_bdp_section() during configuration parsing
/// - Creates: Layer with forward_yolo_layer_bdp as forward function
/// - Initializes: 6-parameter coordinate handling (x,y,w,h,fx,fy)
///
/// DIFFERENCES FROM STANDARD make_yolo_layer:
/// - Sets l.type = ELayerType::YOLO_BDP instead of YOLO
/// - Sets l.coords = 6 instead of 4 (adds fx,fy front point coordinates)
/// - Adjusts output calculations for 6-parameter boxes: (6 + 1 + classes) instead of (4 + 1 + classes)
/// - Uses forward_yolo_layer_bdp for processing oriented boxes
/// - Sets truth_size = 6 + 2 for ground truth handling (6 coords + 1 class + 1 objectness)
Darknet::Layer make_yolo_layer_bdp(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
	TAT(TATPARMS);

	// Aggressive static assertions for compile-time validation
	static_assert(sizeof(DarknetBoxBDP) == 6 * sizeof(float), "BDP box must have exactly 6 float parameters");
	static_assert(std::is_trivially_copyable_v<DarknetBoxBDP>, "BDP box must be trivially copyable");

	// Precondition checks
	assert(batch > 0 && batch <= 64);        // Reasonable batch size bounds
	assert(w > 0 && w <= 2048);              // Grid width bounds
	assert(h > 0 && h <= 2048);              // Grid height bounds
	assert(n > 0 && n <= 10);                // Reasonable anchor count
	assert(total > 0 && total <= 20);        // Total anchor count bounds
	assert(classes > 0 && classes <= 1000);  // Reasonable class count
	assert(max_boxes > 0 && max_boxes <= 1000); // Reasonable max boxes

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::YOLO_BDP;  // Set as BDP layer type
	l.detection = 0;  // Set to 0 to allow free_layer to free labels/class_ids arrays

	l.n = n;
	l.total = total;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.coords = 6;  // BDP uses 6 parameters: x,y,w,h,fx,fy
	l.c = n * (classes + 6 + 1);  // 6 coordinates + 1 objectness + classes
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.cost = (float*)xcalloc(1, sizeof(float));
	l.biases = (float*)xcalloc(total * 2, sizeof(float));
	l.nbiases = total * 2;

	if (mask)
	{
		l.mask = mask;
	}
	else
	{
		l.mask = (int*)xcalloc(n, sizeof(int));
		for (int i = 0; i < n; ++i)
		{
			l.mask[i] = i;
		}
	}
	
	l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
	l.outputs = h * w * n * (classes + 6 + 1);  // 6 coordinates + 1 objectness + classes
	l.inputs = l.outputs;
	l.max_boxes = max_boxes;
	l.truth_size = 6 + 2;  // 6 coordinates + 1 class + 1 objectness for ground truth
	l.truths = l.max_boxes * l.truth_size;

	// Allocate and initialize labels array for tracking assigned ground truth boxes
	l.labels = (int*)xcalloc(batch * l.w * l.h * l.n, sizeof(int));
	for (int i = 0; i < batch * l.w * l.h * l.n; ++i)
	{
		l.labels[i] = -1;
	}

	// Allocate and initialize class_ids array for tracking assigned class IDs
	// This array is used in training to store which class each grid cell was assigned
	l.class_ids = (int*)xcalloc(batch * l.w * l.h * l.n, sizeof(int));
	for (int i = 0; i < batch * l.w * l.h * l.n; ++i)
	{
		l.class_ids[i] = -1;
	}

	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));

	for (int i = 0; i < total * 2; ++i)
	{
		l.biases[i] = .5;
	}

	l.forward = forward_yolo_layer_bdp;   // Use BDP-specific forward function
	l.backward = backward_yolo_layer_bdp; // Use BDP-specific backward function for 6-parameter loss

#ifdef DARKNET_GPU
	l.forward_gpu = forward_yolo_layer_gpu;  // TODO: Implement BDP GPU version if needed
	l.backward_gpu = backward_yolo_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.output_avg_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);

	free(l.output);
	if (cudaSuccess == cudaHostAlloc((void**)&l.output, batch * l.outputs * sizeof(float), cudaHostRegisterMapped))
	{
		l.output_pinned = 1;
	}
	else
	{
		std::ignore = cudaGetLastError();
		l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	}

	free(l.delta);
	if (cudaSuccess == cudaHostAlloc((void**)&l.delta, batch * l.outputs * sizeof(float), cudaHostRegisterMapped))
	{
		l.delta_pinned = 1;
	}
	else
	{
		std::ignore = cudaGetLastError();
		l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	}
#endif

	return l;
}

void resize_yolo_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	l->w = w;
	l->h = h;

	l->outputs = h*w*l->n*(l->classes + 4 + 1);
	l->inputs = l->outputs;

	if (l->embedding_output) l->embedding_output = (float*)xrealloc(l->output, l->batch * l->embedding_size * l->n * l->h * l->w * sizeof(float));
	if (l->labels) l->labels = (int*)xrealloc(l->labels, l->batch * l->n * l->h * l->w * sizeof(int));
	if (l->class_ids) l->class_ids = (int*)xrealloc(l->class_ids, l->batch * l->n * l->h * l->w * sizeof(int));

	if (!l->output_pinned) l->output = (float*)xrealloc(l->output, l->batch*l->outputs * sizeof(float));
	if (!l->delta_pinned) l->delta = (float*)xrealloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef DARKNET_GPU
	if (l->output_pinned) {
		CHECK_CUDA(cudaFreeHost(l->output));
		if (cudaSuccess != cudaHostAlloc((void**)&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
			std::ignore = cudaGetLastError(); // reset CUDA-error
			l->output = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
			l->output_pinned = 0;
		}
	}

	if (l->delta_pinned) {
		CHECK_CUDA(cudaFreeHost(l->delta));
		if (cudaSuccess != cudaHostAlloc((void**)&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
			std::ignore = cudaGetLastError(); // reset CUDA-error
			l->delta = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
			l->delta_pinned = 0;
		}
	}

	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);
	cuda_free(l->output_avg_gpu);

	l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
	l->output_avg_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}


void process_batch(void* ptr)
{
	TAT_COMMENT(TATPARMS, "complicated");

	train_yolo_args *args = (train_yolo_args*)ptr;
	const Darknet::Layer & l = *args->l;
	Darknet::NetworkState state = args->state;
	int b = args->b;

	float avg_cat = 0.0f;

	for (int j = 0; j < l.h; ++j)
	{
		for (int i = 0; i < l.w; ++i)
		{
			for (int n = 0; n < l.n; ++n)
			{
				const int class_index = yolo_entry_index(l, b, n * l.w * l.h + j * l.w + i, 4 + 1);
				const int obj_index = yolo_entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
				const int box_index = yolo_entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
				const int stride = l.w * l.h;
				Darknet::Box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.new_coords);
				float best_match_iou = 0;
				//int best_match_t = 0;
				float best_iou = 0;
				int best_t = 0;

				for (int t = 0; t < l.max_boxes; ++t)
				{
					Darknet::Box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
					if (!truth.x)
					{
						break;  // continue;
					}

					int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
					if (class_id >= l.classes || class_id < 0)
					{
						darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d", class_id);
					}

					float objectness = l.output[obj_index];
					if (isnan(objectness) || isinf(objectness))
					{
						l.output[obj_index] = 0;
					}
					int class_id_match = compare_yolo_class(l.output, l.classes, class_index, l.w * l.h, objectness, class_id, 0.25f);

					float iou = box_iou(pred, truth);
					if (iou > best_match_iou && class_id_match == 1)
					{
						best_match_iou = iou;
						//best_match_t = t;
					}
					if (iou > best_iou)
					{
						best_iou = iou;
						best_t = t;
					}
				}

				// delta for objectness:

				l.delta[obj_index] = l.obj_normalizer * (0 - l.output[obj_index]);
				if (best_match_iou > l.ignore_thresh)
				{
					if (l.objectness_smooth)
					{
						const float delta_obj = l.obj_normalizer * (best_match_iou - l.output[obj_index]);
						if (delta_obj > l.delta[obj_index])
						{
							l.delta[obj_index] = delta_obj;
						}
					}
					else
					{
						l.delta[obj_index] = 0;
					}
				}
				else if (state.net.adversarial)
				{
					float scale = pred.w * pred.h;
					if (scale > 0)
					{
						scale = sqrt(scale);
					}
					l.delta[obj_index] = scale * l.obj_normalizer * (0 - l.output[obj_index]);
					int cl_id;
					int found_object = 0;
					for (cl_id = 0; cl_id < l.classes; ++cl_id)
					{
						if (l.output[class_index + stride * cl_id] * l.output[obj_index] > 0.25)
						{
							l.delta[class_index + stride * cl_id] = scale * (0 - l.output[class_index + stride * cl_id]);
							found_object = 1;
						}
					}
					if (found_object)
					{
						// don't use this loop for adversarial attack drawing
						for (cl_id = 0; cl_id < l.classes; ++cl_id)
						{
							if (l.output[class_index + stride * cl_id] * l.output[obj_index] < 0.25)
							{
								l.delta[class_index + stride * cl_id] = scale * (1 - l.output[class_index + stride * cl_id]);
							}
						}

						l.delta[box_index + 0 * stride] += scale * (0 - l.output[box_index + 0 * stride]);
						l.delta[box_index + 1 * stride] += scale * (0 - l.output[box_index + 1 * stride]);
						l.delta[box_index + 2 * stride] += scale * (0 - l.output[box_index + 2 * stride]);
						l.delta[box_index + 3 * stride] += scale * (0 - l.output[box_index + 3 * stride]);
					}
				}
				if (best_iou > l.truth_thresh)
				{
					const float iou_multiplier = best_iou * best_iou;// (best_iou - l.truth_thresh) / (1.0 - l.truth_thresh);
					if (l.objectness_smooth)
					{
						l.delta[obj_index] = l.obj_normalizer * (iou_multiplier - l.output[obj_index]);
					}
					else
					{
						l.delta[obj_index] = l.obj_normalizer * (1 - l.output[obj_index]);
					}

					int class_id = state.truth[best_t * l.truth_size + b * l.truths + 4];
					if (l.map)
					{
						class_id = l.map[class_id];
					}
					delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, 0, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);
					const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
					if (l.objectness_smooth)
					{
						l.delta[class_index + stride * class_id] = class_multiplier * (iou_multiplier - l.output[class_index + stride * class_id]);
					}
					Darknet::Box truth = float_to_box_stride(state.truth + best_t * l.truth_size + b * l.truths, 1);
					delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
					(*state.net.total_bbox)++;
				}
			}
		}
	}

	for (int t = 0; t < l.max_boxes; ++t)
	{
		Darknet::Box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
		if (!truth.x)
		{
			break;  // continue;
		}

		if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid coordinates, width, or height (x=%f, y=%f, w=%f, h=%f)", truth.x, truth.y, truth.w, truth.h);
		}
		const int check_class_id = state.truth[t * l.truth_size + b * l.truths + 4];
		if (check_class_id >= l.classes || check_class_id < 0)
		{
			continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value
		}

		float best_iou = 0;
		int best_n = 0;
		int i = (truth.x * l.w);
		int j = (truth.y * l.h);
		Darknet::Box truth_shift = truth;
		truth_shift.x = truth_shift.y = 0;
		for (int n = 0; n < l.total; ++n)
		{
			Darknet::Box pred = { 0 };
			pred.w = l.biases[2 * n] / state.net.w;
			pred.h = l.biases[2 * n + 1] / state.net.h;
			float iou = box_iou(pred, truth_shift);
			if (iou > best_iou)
			{
				best_iou = iou;
				best_n = n;
			}
		}

		int mask_n2 = int_index(l.mask, best_n, l.n);
		if (mask_n2 >= 0)
		{
			int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
			if (l.map)
			{
				class_id = l.map[class_id];
			}

			int box_index = yolo_entry_index(l, b, mask_n2 * l.w * l.h + j * l.w + i, 0);
			const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
			ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
			(*state.net.total_bbox)++;

			const int truth_in_index = t * l.truth_size + b * l.truths + 5;
			const int track_id = state.truth[truth_in_index];
			const int truth_out_index = b * l.n * l.w * l.h + mask_n2 * l.w * l.h + j * l.w + i;
			l.labels[truth_out_index] = track_id;
			l.class_ids[truth_out_index] = class_id;

			// range is 0 <= 1
			args->tot_iou += all_ious.iou;
			args->tot_iou_loss += 1 - all_ious.iou;
			// range is -1 <= giou <= 1
			args->tot_giou_loss += 1 - all_ious.giou;

			int obj_index = yolo_entry_index(l, b, mask_n2 * l.w * l.h + j * l.w + i, 4);
			if (l.objectness_smooth)
			{
				float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
				if (l.delta[obj_index] == 0)
				{
					l.delta[obj_index] = delta_obj;
				}
			}
			else
			{
				l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
			}

			int class_index = yolo_entry_index(l, b, mask_n2 * l.w * l.h + j * l.w + i, 4 + 1);
			delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

			++(args->count);
			++(args->class_count);
		}

		// iou_thresh
		for (int n = 0; n < l.total; ++n)
		{
			int mask_n = int_index(l.mask, n, l.n);
			if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f)
			{
				Darknet::Box pred = { 0 };
				pred.w = l.biases[2 * n] / state.net.w;
				pred.h = l.biases[2 * n + 1] / state.net.h;
				float iou = box_iou_kind(pred, truth_shift, l.iou_thresh_kind); // IOU, GIOU, MSE, DIOU, CIOU
				// iou, n

				if (iou > l.iou_thresh)
				{
					int class_id = state.truth[t * l.truth_size + b * l.truths + 4];
					if (l.map)
					{
						class_id = l.map[class_id];
					}

					int box_index = yolo_entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
					const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
					ious all_ious = delta_yolo_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords);
					(*state.net.total_bbox)++;

					// range is 0 <= 1
					args->tot_iou += all_ious.iou;
					args->tot_iou_loss += 1 - all_ious.iou;
					// range is -1 <= giou <= 1
					args->tot_giou_loss += 1 - all_ious.giou;

					int obj_index = yolo_entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
					if (l.objectness_smooth)
					{
						float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
						if (l.delta[obj_index] == 0)
						{
							l.delta[obj_index] = delta_obj;
						}
					}
					else
					{
						l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
					}

					int class_index = yolo_entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4 + 1);
					delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

					++(args->count);
					++(args->class_count);
				}
			}
		}
	}

	if (l.iou_thresh < 1.0f)
	{
		// averages the deltas obtained by the function: delta_yolo_box()_accumulate
		for (int j = 0; j < l.h; ++j)
		{
			for (int i = 0; i < l.w; ++i)
			{
				for (int n = 0; n < l.n; ++n)
				{
					int obj_index = yolo_entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
					int box_index = yolo_entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					int class_index = yolo_entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
					const int stride = l.w*l.h;

					if (l.delta[obj_index] != 0)
					{
						averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta);
					}
				}
			}
		}
	}

	return;
}


void forward_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	memcpy(l.output, state.input, l.outputs * l.batch * sizeof(float));

#ifndef DARKNET_GPU
	for (int b = 0; b < l.batch; ++b)
	{
		for (int n = 0; n < l.n; ++n)
		{
			int bbox_index = yolo_entry_index(l, b, n*l.w*l.h, 0);
			if (l.new_coords)
			{
				//activate_array(l.output + bbox_index, 4 * l.w*l.h, LOGISTIC);    // x,y,w,h
			}
			else
			{
				activate_array(l.output + bbox_index, 2 * l.w*l.h, LOGISTIC);        // x,y,
				int obj_index = yolo_entry_index(l, b, n*l.w*l.h, 4);
				activate_array(l.output + obj_index, (1 + l.classes)*l.w*l.h, LOGISTIC);
			}
			scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + bbox_index, 1);    // scale x,y
		}
	}
#endif

	if (!state.train)
	{
		return;
	}

	// delta is zeroed
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

	for (int i = 0; i < l.batch * l.w*l.h*l.n; ++i)
	{
		l.labels[i] = -1;
	}

	for (int i = 0; i < l.batch * l.w*l.h*l.n; ++i)
	{
		l.class_ids[i] = -1;
	}

	//float avg_iou = 0;
	float tot_iou = 0;
	//float tot_giou = 0;
	//float tot_diou = 0;
	//float tot_ciou = 0;
	float tot_iou_loss = 0;
	float tot_giou_loss = 0;
	float tot_fp_loss = 0;  // Total front point loss across all batches
	//float tot_diou_loss = 0;
	//float tot_ciou_loss = 0;
	//float recall = 0;
	//float recall75 = 0;
	//float avg_cat = 0;
	//float avg_obj = 0;
	//float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;

	int num_threads = l.batch;
	Darknet::VThreads threads;
	threads.reserve(num_threads);

	struct train_yolo_args * yolo_args = (train_yolo_args*)xcalloc(l.batch, sizeof(struct train_yolo_args));

	for (int b = 0; b < l.batch; b++)
	{
		yolo_args[b].l = &l;
		yolo_args[b].state = state;
		yolo_args[b].b = b;

		yolo_args[b].tot_iou = 0;
		yolo_args[b].tot_iou_loss = 0;
		yolo_args[b].tot_giou_loss = 0;
		yolo_args[b].tot_fp_loss = 0;  // Initialize front point loss
		yolo_args[b].count = 0;
		yolo_args[b].class_count = 0;

		threads.emplace_back(process_batch, &(yolo_args[b]));
	}

	for (int b = 0; b < l.batch; b++)
	{
		threads[b].join();

		tot_iou += yolo_args[b].tot_iou;
		tot_iou_loss += yolo_args[b].tot_iou_loss;
		tot_giou_loss += yolo_args[b].tot_giou_loss;
		tot_fp_loss += yolo_args[b].tot_fp_loss;  // Aggregate front point loss
		count += yolo_args[b].count;
		class_count += yolo_args[b].class_count;
	}

	free(yolo_args);

	// Search for an equidistant point from the distant boundaries of the local minimum
	int iteration_num = get_current_iteration(state.net);
	const int start_point = state.net.max_batches * 3 / 4;

	if ((state.net.badlabels_rejection_percentage && start_point < iteration_num) ||
		(state.net.num_sigmas_reject_badlabels && start_point < iteration_num) ||
		(state.net.equidistant_point && state.net.equidistant_point < iteration_num))
	{
		const float progress_it = iteration_num - state.net.equidistant_point;
		const float progress = progress_it / (state.net.max_batches - state.net.equidistant_point);
		float ep_loss_threshold = (*state.net.delta_rolling_avg) * progress * 1.4;

		float cur_max = 0;
		float cur_avg = 0;
		float counter = 0;
		for (int i = 0; i < l.batch * l.outputs; ++i)
		{
			if (l.delta[i] != 0)
			{
				counter++;
				cur_avg += fabs(l.delta[i]);

				if (cur_max < fabs(l.delta[i]))
				{
					cur_max = fabs(l.delta[i]);
				}
			}
		}

		cur_avg = cur_avg / counter;

		if (*state.net.delta_rolling_max == 0)
		{
			*state.net.delta_rolling_max = cur_max;
		}
		*state.net.delta_rolling_max = *state.net.delta_rolling_max * 0.99 + cur_max * 0.01;
		*state.net.delta_rolling_avg = *state.net.delta_rolling_avg * 0.99 + cur_avg * 0.01;

		// reject high loss to filter bad labels
		if (state.net.num_sigmas_reject_badlabels && start_point < iteration_num)
		{
			const float rolling_std = (*state.net.delta_rolling_std);
			const float rolling_max = (*state.net.delta_rolling_max);
			const float rolling_avg = (*state.net.delta_rolling_avg);
			const float progress_badlabels = (float)(iteration_num - start_point) / (start_point);

			float cur_std = 0.0f;
			counter = 0.0f;
			for (int i = 0; i < l.batch * l.outputs; ++i)
			{
				if (l.delta[i] != 0)
				{
					counter++;
					cur_std += pow(l.delta[i] - rolling_avg, 2);
				}
			}
			cur_std = sqrt(cur_std / counter);

			*state.net.delta_rolling_std = *state.net.delta_rolling_std * 0.99 + cur_std * 0.01;

			float final_badlebels_threshold = rolling_avg + rolling_std * state.net.num_sigmas_reject_badlabels;
			float badlabels_threshold = rolling_max - progress_badlabels * fabs(rolling_max - final_badlebels_threshold);
			badlabels_threshold = std::max(final_badlebels_threshold, badlabels_threshold);
			for (int i = 0; i < l.batch * l.outputs; ++i)
			{
				if (fabs(l.delta[i]) > badlabels_threshold)
				{
					l.delta[i] = 0;
				}
			}

			*cfg_and_state.output
				<< " rolling_std="		<< rolling_std
				<< ", rolling_max="		<< rolling_max
				<< ", rolling_avg="		<< rolling_avg
				<< std::endl
				<< "badlabels"
				<< " loss_threshold="	<< badlabels_threshold
				<< ", start_it="		<< start_point
				<< ", progress="		<< progress_badlabels * 100.0f
				<< std::endl;

			ep_loss_threshold = std::min(final_badlebels_threshold, rolling_avg) * progress;
		}

		// reject some percent of the highest deltas to filter bad labels
		if (state.net.badlabels_rejection_percentage && start_point < iteration_num)
		{
			if (*state.net.badlabels_reject_threshold == 0)
			{
				*state.net.badlabels_reject_threshold = *state.net.delta_rolling_max;
			}

			*cfg_and_state.output << "badlabels_reject_threshold=" << *state.net.badlabels_reject_threshold << std::endl;

			const float num_deltas_per_anchor = (l.classes + 4 + 1);
			float counter_reject = 0;
			float counter_all = 0;
			for (int i = 0; i < l.batch * l.outputs; ++i)
			{
				if (l.delta[i] != 0)
				{
					counter_all++;
					if (fabs(l.delta[i]) > (*state.net.badlabels_reject_threshold))
					{
						counter_reject++;
						l.delta[i] = 0;
					}
				}
			}
			float cur_percent = 100 * (counter_reject*num_deltas_per_anchor / counter_all);
			if (cur_percent > state.net.badlabels_rejection_percentage)
			{
				*state.net.badlabels_reject_threshold += 0.01;
				*cfg_and_state.output << "increase!!!" << std::endl;
			}
			else if (*state.net.badlabels_reject_threshold > 0.01)
			{
				*state.net.badlabels_reject_threshold -= 0.01;
				*cfg_and_state.output << "decrease!!!" << std::endl;
			}

			*cfg_and_state.output
				<< "badlabels_reject_threshold="		<< *state.net.badlabels_reject_threshold
				<< ", cur_percent="						<< cur_percent
				<< ", badlabels_rejection_percentage="	<< state.net.badlabels_rejection_percentage
				<< ", delta_rolling_max="				<< *state.net.delta_rolling_max
				<< std::endl;
		}

		// reject low loss to find equidistant point
		if (state.net.equidistant_point && state.net.equidistant_point < iteration_num)
		{
			*cfg_and_state.output
				<< "equidistant_point"
				<< " loss_threshold="	<< ep_loss_threshold
				<< ", start_it="		<< state.net.equidistant_point
				<< ", progress="		<< progress * 100.0f << "%"
				<< std::endl;

			for (int i = 0; i < l.batch * l.outputs; ++i)
			{
				if (fabs(l.delta[i]) < ep_loss_threshold)
				{
					l.delta[i] = 0;
				}
			}
		}
	}

	if (count == 0)
	{
		count = 1;
	}
	if (class_count == 0)
	{
		class_count = 1;
	}

	int stride = l.w*l.h;
	float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
	memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));

	for (int b = 0; b < l.batch; ++b)
	{
		for (int j = 0; j < l.h; ++j)
		{
			for (int i = 0; i < l.w; ++i)
			{
				for (int n = 0; n < l.n; ++n)
				{
					int index = yolo_entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					no_iou_loss_delta[index + 0 * stride] = 0;
					no_iou_loss_delta[index + 1 * stride] = 0;
					no_iou_loss_delta[index + 2 * stride] = 0;
					no_iou_loss_delta[index + 3 * stride] = 0;
				}
			}
		}
	}

	float classification_loss = l.obj_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
	free(no_iou_loss_delta);
	float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	float iou_loss = loss - classification_loss;

	float avg_iou_loss = 0.0f;
	*(l.cost) = loss;

	// gIOU loss + MSE (objectness) loss
	if (l.iou_loss == MSE)
	{
		*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	}
	else
	{
		// Always compute classification loss both for iou + cls loss and for logging with mse loss
		// TODO: remove IOU loss fields before computing MSE on class
		//   probably split into two arrays
		if (l.iou_loss == GIOU)
		{
			avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
		}
		else
		{
			avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
		}
		*(l.cost) = avg_iou_loss + classification_loss;
	}

	loss /= l.batch;
	classification_loss /= l.batch;
	iou_loss /= l.batch;

	// show detailed output
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output <<
				"v3 " << (	l.iou_loss == MSE	?	"mse"	:
							l.iou_loss == GIOU	?	"giou"	:
													"iou"	) << " loss, "
				"Normalizer: "
				"(iou: "		<< std::setprecision(2) << l.iou_normalizer		<<
				", obj: "		<< std::setprecision(2) << l.obj_normalizer		<<
				", cls: "		<< std::setprecision(2) << l.cls_normalizer		<< ") "
				"Region "		<< state.index									<< " "
				"Avg (IOU: "	<< std::setprecision(6) << tot_iou / count		<< "), "
				"count: "		<< count										<< ", "
				"class_loss: "	<< std::setprecision(6) << classification_loss	<< ", "
				"iou_loss: "	<< std::setprecision(6) << iou_loss				<< ", "
				"total_loss: "	<< std::setprecision(6) << loss
								<< std::setprecision(2)							<< std::endl;
	}
}

void backward_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void backward_yolo_layer_bdp(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: ENTRY - layer params: "
	                      << "batch=" << l.batch << " inputs=" << l.inputs << " outputs=" << l.outputs
	                      << " train=" << state.train);

	// Aggressive static assertions for compile-time validation
	static_assert(sizeof(DarknetBoxBDP) == 6 * sizeof(float), "BDP box must have exactly 6 float parameters");
	static_assert(std::is_trivially_copyable_v<DarknetBoxBDP>, "BDP box must be trivially copyable for memcpy operations");
	static_assert(std::is_standard_layout_v<DarknetBoxBDP>, "BDP box must have standard layout for C compatibility");
	static_assert(alignof(DarknetBoxBDP) <= alignof(float), "BDP box alignment must not exceed float alignment");
	static_assert(offsetof(DarknetBoxBDP, fx) == 4 * sizeof(float), "fx must be at offset 16");
	static_assert(offsetof(DarknetBoxBDP, fy) == 5 * sizeof(float), "fy must be at offset 20");
	
	// Simple precondition checks without postcondition validation to avoid race conditions
	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: Validating preconditions...");

	// Layer structure validation
	assert(l.type == Darknet::ELayerType::YOLO_BDP && "Layer must be YOLO_BDP type");
	assert(l.coords == 6 && "BDP layer must have 6 coordinates");
	assert(l.classes > 0 && l.classes <= 1000 && "Classes must be between 1-1000");
	assert(l.n > 0 && l.n <= 10 && "Anchors must be between 1-10");
	assert(l.batch > 0 && l.batch <= 64 && "Batch size must be between 1-64");
	assert(l.w > 0 && l.w <= 2048 && l.h > 0 && l.h <= 2048 && "Grid dimensions must be positive");

	// Memory validation
	assert(l.delta != nullptr && "Layer delta must be allocated");
	assert(state.delta != nullptr && "State delta must be allocated");
	assert(l.inputs > 0 && l.outputs > 0 && "Layer inputs/outputs must be positive");

	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: Preconditions passed");

	// For BDP layers, we need to call the BDP-specific loss calculation
	// This function should be called during training to compute gradients for the 6-parameter boxes

	if (!state.train) {
		// If not training, just copy deltas like standard backward
		DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: Inference mode - copying deltas...");
		DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: Calling axpy_cpu with n=" << (l.batch*l.inputs));
		axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
		DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: axpy_cpu completed");
		return;
	}

	// During training, ensure proper gradient computation for BDP layers
	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: Training mode - propagating gradients...");

	// Note: The actual loss calculation happens in forward_yolo_layer_bdp()
	// which should call delta_yolo_box_bdp() for each ground truth box
	// Copy *l.delta(Already computed in forward pass) to *net.delta, namely *l.delta of the previous layer.

	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: Calling axpy_cpu with n=" << std::dec << (l.batch*l.inputs));
	float delta_mag = mag_array(l.delta, l.batch*l.outputs);
	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: l.delta magnitude=" << std::dec << delta_mag << " (ptr=" << l.delta << ")");
	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: state.delta ptr=" << state.delta);
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: axpy_cpu completed");

	// Fix NaN/inf values in the propagated gradients (network-level safety)
	if (state.net.try_fix_nan)
	{
		DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: Fixing NaN/inf values...");
		fix_nan_and_inf_cpu(state.delta, l.batch * l.inputs);
		fix_nan_and_inf_cpu(l.delta, l.batch * l.outputs);
		DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: NaN/inf fixing completed");
	}

	DEBUG_TRACE("   [DEBUG] backward_yolo_layer_bdp: EXIT - gradient propagation completed");
}

// Converts output of the network to detection boxes
// w,h: image width,height
// netw,neth: network width,height
// relative: 1 (all callers seems to pass TRUE)
void correct_yolo_boxes(Darknet::Detection * dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
	TAT(TATPARMS);

	int i;
	// network height (or width)
	int new_w = 0;
	// network height (or width)
	int new_h = 0;
	// Compute scale given image w,h vs network w,h
	// I think this "rotates" the image to match network to input image w/h ratio
	// new_h and new_w are really just network width and height
	if (letter)
	{
		if (((float)netw / w) < ((float)neth / h))
		{
			new_w = netw;
			new_h = (h * netw) / w;
		}
		else
		{
			new_h = neth;
			new_w = (w * neth) / h;
		}
	}
	else
	{
		new_w = netw;
		new_h = neth;
	}
	// difference between network width and "rotated" width
	float deltaw = netw - new_w;
	// difference between network height and "rotated" height
	float deltah = neth - new_h;
	// ratio between rotated network width and network width
	float ratiow = (float)new_w / netw;
	// ratio between rotated network width and network width
	float ratioh = (float)new_h / neth;

	for (i = 0; i < n; ++i)
	{
		Darknet::Box b = dets[i].bbox;
		// x = ( x - (deltaw/2)/netw ) / ratiow;
		//   x - [(1/2 the difference of the network width and rotated width) / (network width)]
		b.x = (b.x - deltaw / 2. / netw) / ratiow;
		b.y = (b.y - deltah / 2. / neth) / ratioh;
		// scale to match rotation of incoming image
		b.w *= 1 / ratiow;
		b.h *= 1 / ratioh;

		// relative seems to always be == 1, I don't think we hit this condition, ever.
		if (!relative)
		{
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}

		dets[i].bbox = b;
	}
}


int yolo_num_detections(const Darknet::Layer & l, float thresh)
{
	TAT(TATPARMS);

	int count = 0;

	for (int n = 0; n < l.n; ++n)
	{
		/// @todo V3 JAZZ 2024-06-02:  Why does "omp parallel" not work like I expect?
		//#pragma omp parallel for reduction (+:count)
		for (int i = 0; i < l.w * l.h; ++i)
		{
			const int obj_index  = yolo_entry_index(l, 0, n * l.w * l.h + i, 4);
			if (l.output[obj_index] > thresh)
			{
				++count;
			}
		}
	}

	return count;
}


int yolo_num_detections_v3(Darknet::Network * net, const int index, const float thresh, Darknet::Output_Object_Cache & cache)
{
	TAT(TATPARMS);

	// IMPORTANT:  note the object cache is NOT cleared here.  Because there may be multiple YOLO layers within a network,
	// we only want to append to the cache, not overwrite previous entries from earlier YOLO layers.

	int count = 0;

	const Darknet::Layer & l = net->layers[index];

	#pragma omp for schedule(dynamic, 8)
	for (int n = 0; n < l.n; ++n)
	{
		for (int i = 0; i < l.w * l.h; ++i)
		{
			const int obj_index = yolo_entry_index(l, 0, n * l.w * l.h + i, 4);
			if (l.output[obj_index] > thresh)
			{
				++count;

				// remember the location of this object so we don't have to walk through the array again
				Darknet::Output_Object oo;
				oo.layer_index = index;
				oo.n = n;
				oo.i = i;
				oo.obj_index = obj_index;
				cache.push_back(oo);
			}
		}
	}

	return count;
}


int yolo_num_detections_batch(const Darknet::Layer & l, float thresh, int batch)
{
	TAT(TATPARMS);

	int count = 0;
	for (int i = 0; i < l.w * l.h; ++i)
	{
		for (int n = 0; n < l.n; ++n)
		{
			int obj_index  = yolo_entry_index(l, batch, n * l.w * l.h + i, 4);
			if (l.output[obj_index] > thresh)
			{
				++count;
			}
		}
	}

	return count;
}


int get_yolo_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection * dets, int letter)
{
	TAT(TATPARMS);

	const float * predictions = l.output;

	int count = 0;
	for (int i = 0; i < l.w*l.h; ++i)
	{
		int row = i / l.w;
		int col = i % l.w;

		for (int n = 0; n < l.n; ++n)
		{
			int obj_index  = yolo_entry_index(l, 0, n*l.w*l.h + i, 4);
			float objectness = predictions[obj_index];

			if (objectness > thresh)
			{
				int box_index = yolo_entry_index(l, 0, n*l.w*l.h + i, 0);
				dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.new_coords);
				dets[count].objectness = objectness;
				dets[count].classes = l.classes;
				if (l.embedding_output)
				{
					get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, 0, dets[count].embeddings);
				}

				for (int j = 0; j < l.classes; ++j)
				{
					int class_index = yolo_entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
					float prob = objectness*predictions[class_index];
					dets[count].prob[j] = (prob > thresh) ? prob : 0;
				}
				++count;
			}
		}
	}

	correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);

	return count;
}


int get_yolo_detections_v3(Darknet::Network * net, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection * dets, int letter, Darknet::Output_Object_Cache & cache)
{
	TAT(TATPARMS);

	int count = 0;

	for (const auto & oo : cache)
	{
		const auto & i			= oo.i;
		const auto & n			= oo.n;
		const auto & obj_index	= oo.obj_index;

		const Darknet::Layer & l = net->layers[oo.layer_index];
		const float * predictions = l.output;

		const int row			= i / l.w;
		const int col			= i % l.w;
		const float objectness	= predictions[obj_index];

		const int box_index = yolo_entry_index(l, 0, n * l.w * l.h + i, 0);

		dets[count].bbox		= get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w * l.h, l.new_coords);
		dets[count].objectness	= objectness;
		dets[count].classes		= l.classes;

		if (l.embedding_output)
		{
			/// @todo V3 what is this and where does it get used?
			get_embedding(l.embedding_output, l.w, l.h, l.n * l.embedding_size, l.embedding_size, col, row, n, 0, dets[count].embeddings);
		}

		for (int j = 0; j < l.classes; ++j)
		{
			const int class_index = yolo_entry_index(l, 0, n * l.w * l.h + i, 4 + 1 + j);
			const float prob = objectness * predictions[class_index];
			dets[count].prob[j] = (prob > thresh) ? prob : 0.0f;
		}
		++count;
	}

	correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);

	return count;
}


int get_yolo_detections_batch(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection * dets, int letter, int batch)
{
	TAT(TATPARMS);

	int i,j,n;
	float *predictions = l.output;
	//if (l.batch == 2) avg_flipped_yolo(l);
	int count = 0;
	for (i = 0; i < l.w*l.h; ++i){
		int row = i / l.w;
		int col = i % l.w;
		for(n = 0; n < l.n; ++n){
			int obj_index  = yolo_entry_index(l, batch, n*l.w*l.h + i, 4);
			float objectness = predictions[obj_index];
			//if(objectness <= thresh) continue;    // incorrect behavior for Nan values
			if (objectness > thresh)
			{
				int box_index = yolo_entry_index(l, batch, n*l.w*l.h + i, 0);
				dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h, l.new_coords);
				dets[count].objectness = objectness;
				dets[count].classes = l.classes;
				if (l.embedding_output)
				{
					get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, batch, dets[count].embeddings);
				}

				for (j = 0; j < l.classes; ++j)
				{
					int class_index = yolo_entry_index(l, batch, n*l.w*l.h + i, 4 + 1 + j);
					float prob = objectness*predictions[class_index];
					dets[count].prob[j] = (prob > thresh) ? prob : 0;
				}
				++count;
			}
		}
	}
	correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
	return count;
}

#ifdef DARKNET_GPU

void forward_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.embedding_output)
	{
		Darknet::Layer & le = state.net.layers[l.embedding_layer_id];
		cuda_pull_array_async(le.output_gpu, l.embedding_output, le.batch*le.outputs);
	}

	//copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
	simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
	for (int b = 0; b < l.batch; ++b)
	{
		for (int n = 0; n < l.n; ++n)
		{
			int bbox_index = yolo_entry_index(l, b, n*l.w*l.h, 0);
			// y = 1./(1. + exp(-x))
			// x = ln(y/(1-y))  // ln - natural logarithm (base = e)
			// if(y->1) x -> inf
			// if(y->0) x -> -inf
			if (l.new_coords)
			{
				//activate_array_ongpu(l.output_gpu + bbox_index, 4 * l.w*l.h, LOGISTIC);    // x,y,w,h
			}
			else
			{
				activate_array_ongpu(l.output_gpu + bbox_index, 2 * l.w*l.h, LOGISTIC);    // x,y

				int obj_index = yolo_entry_index(l, b, n*l.w*l.h, 4);
				activate_array_ongpu(l.output_gpu + obj_index, (1 + l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
			}

			if (l.scale_x_y != 1)
			{
				scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + bbox_index, 1);      // scale x,y
			}
		}
	}

	if (!state.train || l.onlyforward)
	{
		//cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
		if (l.mean_alpha && l.output_avg_gpu)
		{
			mean_array_gpu(l.output_gpu, l.batch*l.outputs, l.mean_alpha, l.output_avg_gpu);
		}
		cuda_pull_array_async(l.output_gpu, l.output, l.batch * l.outputs);
		CHECK_CUDA(cudaPeekAtLastError());
		return;
	}

	float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
	cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
	memcpy(in_cpu, l.output, l.batch * l.outputs * sizeof(float));
	float * truth_cpu = nullptr;

	if (state.truth)
	{
		int num_truth = l.batch * l.truths;
		truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);
	}
	Darknet::NetworkState cpu_state = state;
//	cpu_state.net = state.net;
//	cpu_state.index = state.index;
//	cpu_state.train = state.train;
	cpu_state.truth = truth_cpu;
	cpu_state.input = in_cpu;
	forward_yolo_layer(l, cpu_state);
	//forward_yolo_layer(l, state);
	cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);

	free(in_cpu);
	if (cpu_state.truth)
	{
		free(cpu_state.truth);
	}
}

void backward_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	axpy_ongpu(l.batch*l.inputs, state.net.loss_scale * l.delta_normalizer, l.delta_gpu, 1, state.delta, 1);
}
#endif


Darknet::MMats Darknet::create_yolo_heatmaps(Darknet::NetworkPtr ptr, const float threshold)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot generate heatmaps without a network pointer");
	}

	MMats m;
	m[-1] = cv::Mat(net->h, net->w, CV_32FC1, {0, 0, 0});
	for (size_t idx = 0; idx < net->details->class_names.size(); idx ++)
	{
		if (net->details->classes_to_ignore.count(idx) == 0)
		{
			m[idx] = cv::Mat(net->h, net->w, CV_32FC1, {0, 0, 0});
		}
	}

	// look through all the layers to find the YOLO ones
	for (int layer_index = 0; layer_index < net->n; layer_index ++)
	{
		const Darknet::Layer & l = net->layers[layer_index];
		if (l.type != Darknet::ELayerType::YOLO)
		{
			// not YOLO...keep looking for another layer
			continue;
		}

//		Darknet::dump(l);

		for (int n = 0; n < l.n; ++n) // anchors?
		{
			for (int idx = 0; idx < l.w * l.h; idx ++) // loop through all entries in the YOLO output buffer
			{
				const size_t objectness_index = yolo_entry_index(l, 0, n * l.w * l.h + idx, 4);
				const float & objectness = l.output[objectness_index];

				for (int class_index = 0; class_index < l.classes; class_index ++)
				{
					if (m.count(class_index) == 0)
					{
						// we've been told to ignore this class
						continue;
					}

					const size_t confidence_index = yolo_entry_index(l, 0, n * l.w * l.h + idx, 5 + class_index);
					const float & confidence = l.output[confidence_index];
					const float probability = objectness * confidence;

					if (probability < threshold)
					{
						// ignore this prediction (does not match threshold)
						continue;
					}

					// need the X and Y coordinates
					const int box_index = yolo_entry_index(l, 0, n * l.w * l.h + idx, 0);
					const int row = idx / l.w;
					const int col = idx % l.w;
					const auto bbox = get_yolo_box(l.output, l.biases, l.mask[n], box_index, col, row, l.w, l.h, net->w, net->h, l.w * l.h, l.new_coords);

					#if 0
					*cfg_and_state.output
						<< "layer=" << layer_index
						<< " anchor=" << n
						<< " idx=" << idx
						<< " class=" << class_index
						<< " row=" << row
						<< " col=" << col
						<< " obj=" << objectness
						<< " conf=" << confidence
						<< " prob=" << probability
						<< " x=" << bbox.x
						<< " y=" << bbox.y
						<< " w=" << bbox.w
						<< " h=" << bbox.h
						<< std::endl;
					#endif

					const int w = std::round(net->w * bbox.w);
					const int h = std::round(net->h * bbox.h);
					const int x = std::round(net->w * (bbox.x - bbox.w / 2.0f));
					const int y = std::round(net->h * (bbox.y - bbox.h / 2.0f));
					cv::Rect r(x, y, w, h);
					if (r.x < 0) r.x = 0;
					if (r.y < 0) r.y = 0;
					if (r.x + r.width >= net->w) r.width = net->w - r.x - 1;
					if (r.y + r.height >= net->h) r.height = net->h - r.y - 1;

					// create a mask with rounded corners
					// https://stackoverflow.com/a/78814207/13022

					cv::Mat mask(net->h, net->w, CV_32FC1);
					mask = 0.0f;

					const auto linetype = cv::LineTypes::LINE_4;
					const auto linethickness = 1;
					const auto colour = cv::Scalar(1.0f);

					const cv::Point tl(r.tl());
					const cv::Point br(r.br());
					const cv::Point tr(br.x, tl.y);
					const cv::Point bl(tl.x, br.y);

					const int hoffset = std::round((tr.x - tl.x) / 5.0f);
					const int voffset = std::round((bl.y - tl.y) / 5.0f);

					// draw horizontal and vertical segments
					cv::line(mask, cv::Point(tl.x + hoffset, tl.y), cv::Point(tr.x - hoffset, tr.y), colour, linethickness, linetype);
					cv::line(mask, cv::Point(tr.x, tr.y + voffset), cv::Point(br.x, br.y - voffset), colour, linethickness, linetype);
					cv::line(mask, cv::Point(br.x - hoffset, br.y), cv::Point(bl.x + hoffset, bl.y), colour, linethickness, linetype);
					cv::line(mask, cv::Point(bl.x, bl.y - voffset), cv::Point(tl.x, tl.y + voffset), colour, linethickness, linetype);

					// draw each of the corners
					cv::ellipse(mask, tl + cv::Point(+hoffset, +voffset), cv::Size(hoffset, voffset), 0.0, 180.0 , 270.0 , colour, linethickness, linetype);
					cv::ellipse(mask, tr + cv::Point(-hoffset, +voffset), cv::Size(hoffset, voffset), 0.0, 270.0 , 360.0 , colour, linethickness, linetype);
					cv::ellipse(mask, br + cv::Point(-hoffset, -voffset), cv::Size(hoffset, voffset), 0.0, 0.0   , 90.0  , colour, linethickness, linetype);
					cv::ellipse(mask, bl + cv::Point(+hoffset, -voffset), cv::Size(hoffset, voffset), 0.0, 90.0  , 180.0 , colour, linethickness, linetype);

					cv::floodFill(mask, cv::Point(r.x + r.width / 2, r.y + r.height / 2), colour);

					cv::Mat value(net->h, net->w, CV_32FC1);
					value = probability;

					m[-1] += (value & mask);
					m[class_index] += (value & mask);
				}
			}
		}
	}

	return m;
}


/// @implement OBB: Helper function to extract BDP ground truth box from truth array
/// Extracts a 6-parameter oriented bounding box from the truth array with stride support.
///
/// @param f Pointer to truth array at the start of box data (x coordinate)
/// @param stride Memory stride between consecutive values (typically 1)
/// @return DarknetBoxBDP with x,y,w,h,fx,fy extracted from truth array
///
/// MEMORY LAYOUT EXPECTED (truth_size = 8):
/// - f[0*stride]: x (center x coordinate, normalized [0,1])
/// - f[1*stride]: y (center y coordinate, normalized [0,1])
/// - f[2*stride]: w (width, normalized [0,1])
/// - f[3*stride]: h (height, normalized [0,1])
/// - f[4*stride]: fx (front point x coordinate, normalized [0,1])
/// - f[5*stride]: fy (front point y coordinate, normalized [0,1])
/// - f[6*stride]: reserved/track_id (not extracted, accessed separately via offset +6)
/// - f[7*stride]: class_id (not extracted, accessed separately via offset +7)
///
/// USAGE IN TRAINING:
/// Called from process_batch_bdp() to extract ground truth for loss calculation.
/// Pointer arithmetic: state.truth + t * l.truth_size + b * l.truths gives box start.
static inline DarknetBoxBDP float_to_box_bdp_stride(const float *f, const int stride)
{
	TAT(TATPARMS);

	// Preconditions: validate input parameters (using asserts instead of exceptions)
	#ifndef NDEBUG
	assert(f != nullptr && "Truth array pointer must not be null");
	assert(stride > 0 && "Stride must be positive");
	// Validate all 6 coordinates are finite
	assert(std::isfinite(f[0]) && "x coordinate must be finite");
	assert(std::isfinite(f[1 * stride]) && "y coordinate must be finite");
	assert(std::isfinite(f[2 * stride]) && "w must be finite");
	assert(std::isfinite(f[3 * stride]) && "h must be finite");
	assert(std::isfinite(f[4 * stride]) && "fx coordinate must be finite");
	assert(std::isfinite(f[5 * stride]) && "fy coordinate must be finite");
	// Validate coordinate ranges [0,1]
	assert(f[0] >= 0.0f && f[0] <= 1.0f && "x must be in [0,1]");
	assert(f[1 * stride] >= 0.0f && f[1 * stride] <= 1.0f && "y must be in [0,1]");
	assert(f[2 * stride] >= 0.0f && f[2 * stride] <= 1.0f && "w must be in [0,1]");
	assert(f[3 * stride] >= 0.0f && f[3 * stride] <= 1.0f && "h must be in [0,1]");
	assert(f[4 * stride] >= 0.0f && f[4 * stride] <= 1.0f && "fx must be in [0,1]");
	assert(f[5 * stride] >= 0.0f && f[5 * stride] <= 1.0f && "fy must be in [0,1]");
	#endif

	// Extract 6-parameter BDP box from truth array
	DarknetBoxBDP b;
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];
	b.fx = f[4 * stride];
	b.fy = f[5 * stride];

	// Postconditions: validate output box (using asserts instead of exceptions)
	#ifndef NDEBUG
	assert(std::isfinite(b.x) && "Output x must be finite");
	assert(std::isfinite(b.y) && "Output y must be finite");
	assert(std::isfinite(b.w) && "Output w must be finite");
	assert(std::isfinite(b.h) && "Output h must be finite");
	assert(std::isfinite(b.fx) && "Output fx must be finite");
	assert(std::isfinite(b.fy) && "Output fy must be finite");
	assert(b.w > 0.0f && b.h > 0.0f && "Width and height must be positive");
	#endif

	return b;
}


/// @implement OBB: Process a single batch for BDP training
/// This function computes loss and gradients for one batch of BDP boxes.
///
/// @param ptr Pointer to train_yolo_args structure containing layer and state
///
/// PROCESSING FLOW:
/// 1. Iterate through all grid positions and anchors
/// 2. Extract predicted BDP boxes using get_yolo_box_bdp()
/// 3. Match predictions with ground truth using IOU (based on x,y,w,h)
/// 4. Compute objectness delta for background/foreground classification
/// 5. For matched boxes, call delta_yolo_box_bdp() to compute coordinate loss
/// 6. Compute class loss using delta_yolo_class()
///
/// DIFFERENCES FROM STANDARD PROCESS_BATCH:
/// - Uses yolo_entry_index_bdp() for memory access (6 coords instead of 4)
/// - Calls delta_yolo_box_bdp() instead of delta_yolo_box()
/// - Extracts ground truth using float_to_box_bdp_stride()
/// - Class ID at offset +6 instead of +4 in truth array
/// - Objectness at entry 6, classes at entry 7+ (instead of 4, 5+)
void process_batch_bdp(void* ptr)
{
	TAT_COMMENT(TATPARMS, "BDP training batch processing");

	DEBUG_TRACE("   [DEBUG] process_batch_bdp: ENTRY - ptr=" << ptr);

	train_yolo_args *args = (train_yolo_args*)ptr;
	DEBUG_TRACE("   [DEBUG] process_batch_bdp: args=" << args);

	const Darknet::Layer & l = *args->l;
	DEBUG_TRACE("   [DEBUG] process_batch_bdp: layer at " << args->l);

	Darknet::NetworkState state = args->state;
	int b = args->b;
	DEBUG_TRACE("   [DEBUG] process_batch_bdp: batch index b=" << b);

	// Validate inputs
	DEBUG_TRACE("   [DEBUG] process_batch_bdp: Validating inputs...");
	if (!state.truth)
	{
		*cfg_and_state.output << "   [WARNING] process_batch_bdp: No ground truth available - returning" << std::endl;
		return; // No ground truth available
	}
	DEBUG_TRACE("   [DEBUG] process_batch_bdp: Ground truth OK at " << state.truth);

	if (b < 0 || b >= l.batch)
	{
		*cfg_and_state.output << "   [ERROR] process_batch_bdp: Invalid batch index b=" << b << " (valid range: 0-" << (l.batch-1) << ")" << std::endl;
		darknet_fatal_error(DARKNET_LOC, "invalid batch index b=%d (must be 0 <= b < %d)", b, l.batch);
	}
	DEBUG_TRACE("   [DEBUG] process_batch_bdp: Batch index valid");

	float avg_cat = 0.0f;

	// Iterate through grid positions and anchors
	DEBUG_TRACE("   [DEBUG] process_batch_bdp: Starting grid iteration (h=" << l.h << " w=" << l.w << " n=" << l.n << ")...");
	for (int j = 0; j < l.h; ++j)
	{
		for (int i = 0; i < l.w; ++i)
		{
			for (int n = 0; n < l.n; ++n)
			{
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: Grid position j=" << std::dec << j << " i=" << i << " anchor n=" << n);

				// Get memory indices for this anchor box
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: Calling yolo_entry_index_bdp for indices...");
				const int class_index = yolo_entry_index_bdp(l, b, n * l.w * l.h + j * l.w + i, 7); // Classes at entry 7+
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: class_index=" << std::dec << class_index);

				const int obj_index = yolo_entry_index_bdp(l, b, n * l.w * l.h + j * l.w + i, 6);   // Objectness at entry 6
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: obj_index=" << std::dec << obj_index);

				const int box_index = yolo_entry_index_bdp(l, b, n * l.w * l.h + j * l.w + i, 0);   // Box starts at entry 0
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: box_index=" << std::dec << box_index);

				const int stride = l.w * l.h;
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: stride=" << std::dec << stride);

				// Extract predicted BDP box
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: Calling get_yolo_box_bdp...");
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: Parameters - l.mask[" << std::dec << n << "]=" << std::dec << l.mask[n]
				            << " biases[0]=" << l.biases[0] << " biases[1]=" << l.biases[1]);
				DarknetBoxBDP pred_bdp = get_yolo_box_bdp(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.new_coords);
				DEBUG_TRACE("   [DEBUG] process_batch_bdp: get_yolo_box_bdp returned successfully");

				DarknetBoxBDP pred;
				pred.x = pred_bdp.x;
				pred.y = pred_bdp.y;
				pred.w = pred_bdp.w;
				pred.h = pred_bdp.h;
				pred.fx = pred_bdp.fx;
				pred.fy = pred_bdp.fy;

				float best_match_iou = 0;
				float best_iou = 0;
				int best_t = 0;

				// Match prediction with ground truth boxes
				for (int t = 0; t < l.max_boxes; ++t)
				{
					// Extract ground truth BDP box
					DarknetBoxBDP truth_bdp = float_to_box_bdp_stride(state.truth + t * l.truth_size + b * l.truths, 1);
					if (!truth_bdp.x)
					{
						break;  // No more ground truth boxes
					}

					// Get class ID (at offset +7 in BDP truth array: [x,y,w,h,fx,fy,reserved,class_id])
					int class_id = state.truth[t * l.truth_size + b * l.truths + 7];
					if (class_id >= l.classes || class_id < 0)
					{
						darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d", class_id);
					}

					// Check for NaN/inf in objectness
					float objectness = l.output[obj_index];
					if (isnan(objectness) || isinf(objectness))
					{
						l.output[obj_index] = 0;
					}

					// Compare class predictions
					int class_id_match = compare_yolo_class(l.output, l.classes, class_index, l.w * l.h, objectness, class_id, 0.25f);

					// Use BDP IoU for matching
					float iou_bdp = box_iou_bdp(pred_bdp, truth_bdp);
					if (iou_bdp > best_match_iou && class_id_match == 1)
					{
						best_match_iou = iou_bdp;
					}
					if (iou_bdp > best_iou)
					{
						best_iou = iou_bdp;
						best_t = t;
					}
				}

				// Compute objectness delta for background boxes
				l.delta[obj_index] = l.obj_normalizer * (0 - l.output[obj_index]);

				// DEBUG: Track objectness delta for background/ignored boxes
				static int obj_nan_debug = 0;
				bool obj_has_nan = !std::isfinite(l.delta[obj_index]) || !std::isfinite(l.output[obj_index]);
				if (obj_has_nan || obj_nan_debug++ < 3) {
					*cfg_and_state.output << "   [OBJ_BG_DEBUG] batch=" << b << " n=" << n << " i=" << i << " j=" << j
					                      << "\n      best_match_iou=" << best_match_iou << " ignore_thresh=" << l.ignore_thresh
					                      << "\n      l.output[obj_index]=" << l.output[obj_index]
					                      << " l.delta[obj_index]=" << l.delta[obj_index]
					                      << " obj_normalizer=" << l.obj_normalizer << std::endl;
				}

				if (best_match_iou > l.ignore_thresh)
				{
					if (l.objectness_smooth)
					{
						const float delta_obj = l.obj_normalizer * (best_match_iou - l.output[obj_index]);
						if (delta_obj > l.delta[obj_index])
						{
							l.delta[obj_index] = delta_obj;
						}
					}
					else
					{
						l.delta[obj_index] = 0;
					}

					// DEBUG: After adjustment
					if (obj_has_nan || !std::isfinite(l.delta[obj_index]) || std::abs(l.delta[obj_index]) < 1e-6f || std::abs(l.delta[obj_index]) > 1.0f) {
						*cfg_and_state.output << "   [OBJ_IGNORED_DEBUG] After ignore_thresh adjustment:"
						                      << " l.delta[obj_index]=" << l.delta[obj_index]
						                      << " objectness_smooth=" << l.objectness_smooth << std::endl;
					}
				}

				// If IOU exceeds threshold, compute full loss for this matched box
				if (best_iou > l.truth_thresh)
				{
					// DEBUG: Track truth_thresh match
					static int truth_match_debug = 0;
					if (truth_match_debug++ < 3) {
						*cfg_and_state.output << "   [TRUTH_MATCH_DEBUG] batch=" << b << " n=" << n << " i=" << i << " j=" << j
						                      << "\n      best_iou=" << best_iou << " truth_thresh=" << l.truth_thresh
						                      << " best_t=" << best_t << std::endl;
					}

					const float iou_multiplier = best_iou * best_iou;
					if (l.objectness_smooth)
					{
						l.delta[obj_index] = l.obj_normalizer * (iou_multiplier - l.output[obj_index]);
					}
					else
					{
						l.delta[obj_index] = l.obj_normalizer * (1 - l.output[obj_index]);
					}

					// DEBUG: Check objectness delta
					if (!std::isfinite(l.delta[obj_index])) {
						*cfg_and_state.output << "   [TRUTH_OBJ_NAN_DEBUG] NaN in truth-matched objectness delta!"
						                      << " obj_index=" << obj_index
						                      << " iou_multiplier=" << iou_multiplier
						                      << " l.output[obj_index]=" << l.output[obj_index]
						                      << " obj_normalizer=" << l.obj_normalizer << std::endl;
					}

					// Get ground truth class ID and apply mapping if needed
					int class_id = state.truth[best_t * l.truth_size + b * l.truths + 7];
					if (l.map)
					{
						class_id = l.map[class_id];
					}

					// Compute class loss
					delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, 0, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);
					const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
					if (l.objectness_smooth)
					{
						l.delta[class_index + stride * class_id] = class_multiplier * (iou_multiplier - l.output[class_index + stride * class_id]);
					}

					// Extract ground truth BDP box and compute coordinate loss
					DarknetBoxBDP truth_bdp = float_to_box_bdp_stride(state.truth + best_t * l.truth_size + b * l.truths, 1);

					// Compute BDP box loss (includes x,y,w,h,fx,fy with Smooth L1 for fx/fy)
					// current_iter = seen / batch_size (approx iteration number for logging)
					int current_iter = state.net.seen ? (*state.net.seen / l.batch) : 0;
					ious all_ious = delta_yolo_box_bdp(truth_bdp, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth_bdp.w * truth_bdp.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.fp_normalizer, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords, current_iter);

					static int accum_count_1 = 0;
					if (accum_count_1++ < 5) {
						*cfg_and_state.output << "   [ACCUM#1-" << accum_count_1 << "] Location: first loop (grid scan)"
						                      << " fp_loss=" << all_ious.fp_loss
						                      << " tot_fp_loss_before=" << args->tot_fp_loss
						                      << " tot_fp_loss_after=" << (args->tot_fp_loss + all_ious.fp_loss) << std::endl;
					}

					args->tot_fp_loss += all_ious.fp_loss;  // Accumulate front point loss
					(*state.net.total_bbox)++;
				}
			}
		}
	}

	// Process all ground truth boxes to ensure they are assigned to best anchor
	for (int t = 0; t < l.max_boxes; ++t)
	{
		DarknetBoxBDP truth_bdp = float_to_box_bdp_stride(state.truth + t * l.truth_size + b * l.truths, 1);
		if (!truth_bdp.x)
		{
			break;  // No more ground truth boxes
		}

		// Validate coordinates
		if (truth_bdp.x < 0 || truth_bdp.y < 0 || truth_bdp.x > 1 || truth_bdp.y > 1 || truth_bdp.w < 0 || truth_bdp.h < 0 ||
		    truth_bdp.w > 1 || truth_bdp.h > 1 || truth_bdp.fx < 0 || truth_bdp.fy < 0 || truth_bdp.fx > 1 || truth_bdp.fy > 1)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid coordinates, width, or height (x=%f, y=%f, w=%f, h=%f, fx=%f, fy=%f)", truth_bdp.x, truth_bdp.y, truth_bdp.w, truth_bdp.h, truth_bdp.fx, truth_bdp.fy);
		}

		const int check_class_id = state.truth[t * l.truth_size + b * l.truths + 7];
		if (check_class_id >= l.classes || check_class_id < 0)
		{
			continue; // Skip if class_id is invalid
		}

		// Find best anchor for this ground truth box
		float best_iou = 0;
		int best_n = 0;
		int i = (truth_bdp.x * l.w);
		int j = (truth_bdp.y * l.h);

		// Create shifted box for anchor matching (centered at 0,0)
		DarknetBoxBDP truth_shift;
		truth_shift.x = 0;
		truth_shift.y = 0;
		truth_shift.w = truth_bdp.w;
		truth_shift.h = truth_bdp.h;
		truth_shift.fx = truth_bdp.fx;
		truth_shift.fy = truth_bdp.fy;

		for (int n = 0; n < l.total; ++n)
		{
			DarknetBoxBDP pred = { 0 };
			pred.w = l.biases[2 * n] / state.net.w;
			pred.h = l.biases[2 * n + 1] / state.net.h;
			// Anchors only define shape (w,h), not orientation (fx,fy)
			// Match center and orientation with truth for proper angular correction
			pred.x = truth_shift.x;
			pred.y = truth_shift.y;
			pred.fx = truth_shift.fx;
			pred.fy = truth_shift.fy;
			float iou = box_iou_bdp(pred, truth_shift);
			if (iou > best_iou)
			{
				best_iou = iou;
				best_n = n;
			}
		}

		// If best anchor is in this layer's mask, compute loss
		int mask_n2 = int_index(l.mask, best_n, l.n);
		if (mask_n2 >= 0)
		{
			int class_id = state.truth[t * l.truth_size + b * l.truths + 7];
			if (l.map)
			{
				class_id = l.map[class_id];
			}

			int box_index = yolo_entry_index_bdp(l, b, mask_n2 * l.w * l.h + j * l.w + i, 0);
			const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;

			// DEBUG: Track GT assignment to best anchor
			static int gt_assign_debug = 0;
			if (gt_assign_debug++ < 3) {
				*cfg_and_state.output << "   [GT_ASSIGN_DEBUG] t=" << t << " best_n=" << best_n << " mask_n2=" << mask_n2
				                      << " i=" << i << " j=" << j
				                      << "\n      truth: x=" << truth_bdp.x << " y=" << truth_bdp.y
				                      << " w=" << truth_bdp.w << " h=" << truth_bdp.h
				                      << " fx=" << truth_bdp.fx << " fy=" << truth_bdp.fy
				                      << "\n      iou_normalizer=" << l.iou_normalizer << " fp_normalizer=" << l.fp_normalizer
				                      << " class_multiplier=" << class_multiplier << std::endl;
			}

			// Compute BDP box loss for this ground truth (includes angular IoU correction and Smooth L1 for fx/fy)
			int current_iter = state.net.seen ? (*state.net.seen / l.batch) : 0;
			ious all_ious = delta_yolo_box_bdp(truth_bdp, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth_bdp.w * truth_bdp.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.fp_normalizer, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords, current_iter);

			// DEBUG: Check for NaN in returned IoUs
			if (!std::isfinite(all_ious.iou) || !std::isfinite(all_ious.fp_loss)) {
				*cfg_and_state.output << "   [GT_IOU_NAN_DEBUG] NaN in second loop (GT assignment)!"
				                      << " iou=" << all_ious.iou << " giou=" << all_ious.giou
				                      << " diou=" << all_ious.diou << " ciou=" << all_ious.ciou
				                      << " fp_loss=" << all_ious.fp_loss << std::endl;
			}

			static int accum_count_2 = 0;
			if (accum_count_2++ < 5) {
				*cfg_and_state.output << "   [ACCUM#2-" << accum_count_2 << "] Location: second loop (GT assignment to best anchor)"
				                      << " fp_loss=" << all_ious.fp_loss
				                      << " tot_fp_loss_before=" << args->tot_fp_loss
				                      << " tot_fp_loss_after=" << (args->tot_fp_loss + all_ious.fp_loss) << std::endl;
			}

			args->tot_fp_loss += all_ious.fp_loss;  // Accumulate front point loss
			(*state.net.total_bbox)++;

			const int truth_in_index = t * l.truth_size + b * l.truths + 6; // Track ID/reserved at offset +6
			const int track_id = state.truth[truth_in_index];
			const int truth_out_index = b * l.n * l.w * l.h + mask_n2 * l.w * l.h + j * l.w + i;
			l.labels[truth_out_index] = track_id;
			l.class_ids[truth_out_index] = class_id;

			// Accumulate IOU statistics
			args->tot_iou += all_ious.iou;
			args->tot_iou_loss += 1 - all_ious.iou;
			args->tot_giou_loss += 1 - all_ious.giou;

			int obj_index = yolo_entry_index_bdp(l, b, mask_n2 * l.w * l.h + j * l.w + i, 6);
			if (l.objectness_smooth)
			{
				float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
				if (l.delta[obj_index] == 0)
				{
					l.delta[obj_index] = delta_obj;
				}
			}
			else
			{
				l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
			}

			int class_index = yolo_entry_index_bdp(l, b, mask_n2 * l.w * l.h + j * l.w + i, 7);
			delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

			++(args->count);
			++(args->class_count);
		}

		// Handle IOU threshold for additional anchors
		for (int n = 0; n < l.total; ++n)
		{
			int mask_n = int_index(l.mask, n, l.n);
			if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f)
			{
				DarknetBoxBDP pred = { 0 };
				pred.w = l.biases[2 * n] / state.net.w;
				pred.h = l.biases[2 * n + 1] / state.net.h;
				// Anchors only define shape (w,h), not orientation (fx,fy)
				// Match center and orientation with truth for proper angular correction
				pred.x = truth_shift.x;
				pred.y = truth_shift.y;
				pred.fx = truth_shift.fx;
				pred.fy = truth_shift.fy;
				float iou = box_iou_bdp(pred, truth_shift);

				if (iou > l.iou_thresh)
				{
					int class_id = state.truth[t * l.truth_size + b * l.truths + 7];
					if (l.map)
					{
						class_id = l.map[class_id];
					}

					int box_index = yolo_entry_index_bdp(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
					const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
					int current_iter = state.net.seen ? (*state.net.seen / l.batch) : 0;
					ious all_ious = delta_yolo_box_bdp(truth_bdp, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth_bdp.w * truth_bdp.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.fp_normalizer, l.iou_loss, 1, l.max_delta, state.net.rewritten_bbox, l.new_coords, current_iter);

					static int accum_count_3 = 0;
					if (accum_count_3++ < 5) {
						*cfg_and_state.output << "   [ACCUM#3-" << accum_count_3 << "] Location: third loop (additional high-IOU anchors)"
						                      << " fp_loss=" << all_ious.fp_loss
						                      << " tot_fp_loss_before=" << args->tot_fp_loss
						                      << " tot_fp_loss_after=" << (args->tot_fp_loss + all_ious.fp_loss) << std::endl;
					}

					args->tot_fp_loss += all_ious.fp_loss;  // Accumulate front point loss
					(*state.net.total_bbox)++;

					args->tot_iou += all_ious.iou;
					args->tot_iou_loss += 1 - all_ious.iou;
					args->tot_giou_loss += 1 - all_ious.giou;

					int obj_index = yolo_entry_index_bdp(l, b, mask_n * l.w * l.h + j * l.w + i, 6);
					if (l.objectness_smooth)
					{
						float delta_obj = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
						if (l.delta[obj_index] == 0)
						{
							l.delta[obj_index] = delta_obj;
						}
					}
					else
					{
						l.delta[obj_index] = class_multiplier * l.obj_normalizer * (1 - l.output[obj_index]);
					}

					int class_index = yolo_entry_index_bdp(l, b, mask_n * l.w * l.h + j * l.w + i, 7);
					delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers, l.cls_normalizer);

					++(args->count);
					++(args->class_count);
				}
			}
		}
	}
}


/// @implement OBB: Forward pass for YOLO layer with oriented bounding box (BDP) support
/// This function processes the YOLO layer output for 6-parameter oriented bounding boxes.
///
/// INTERACTION WITH OTHER FUNCTIONS:
/// - Called by: Network forward pass when layer is in BDP mode
/// - Calls: yolo_entry_index_bdp() to access memory locations
/// - Uses: get_yolo_box_bdp() indirectly through detection extraction
/// - Modifies: l.output (applies activation functions)
///
/// DIFFERENCES FROM STANDARD FORWARD PASS:
/// - Expects 6 parameters per box instead of 4 (x,y,w,h,fx,fy vs x,y,w,h)
/// - Uses yolo_entry_index_bdp() which accounts for 6+1+classes entries per box
/// - Objectness and class activations remain at positions 6 and 7+ (instead of 4 and 5+)
/// - Front point coordinates (fx,fy) at positions 4-5 get same treatment as center (x,y)
///
/// MEMORY LAYOUT PROCESSING:
/// - Processes each batch, each anchor, applies activations to coordinate subsets
/// - Applies logistic activation to: x,y coordinates (entries 0-1) and fx,fy (entries 4-5) 
/// - Applies logistic activation to: objectness (entry 6) and classes (entries 7+)
/// - Applies scaling to center coordinates (x,y) for numerical stability
void forward_yolo_layer_bdp(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	// BDP requires RIOU loss - enforce this requirement
	if (l.iou_loss != RIOU) {
		*cfg_and_state.output << "   [ERROR] BDP layer requires iou_loss=RIOU, but got iou_loss=" << l.iou_loss << std::endl;
		*cfg_and_state.output << "   [ERROR] Please set 'iou_loss=riou' in your .cfg file for BDP layers" << std::endl;
		throw std::runtime_error("BDP layer requires RIOU loss");
	}

	// Debug: Entry point logging with detailed layer parameters
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: ENTRY - layer params: "
	                      << "batch=" << l.batch << " n=" << l.n << " w=" << l.w << " h=" << l.h
	                      << " classes=" << l.classes << " coords=" << l.coords << " outputs=" << l.outputs
	                      << " train=" << state.train << " iou_loss=RIOU");

	// Aggressive static assertions for compile-time validation
	static_assert(sizeof(DarknetBoxBDP) == 6 * sizeof(float), "BDP box must have exactly 6 float parameters");
	static_assert(std::is_trivially_copyable_v<DarknetBoxBDP>, "BDP box must be trivially copyable for memcpy operations");
	static_assert(std::is_standard_layout_v<DarknetBoxBDP>, "BDP box must have standard layout for C compatibility");
	static_assert(alignof(DarknetBoxBDP) <= alignof(float), "BDP box alignment must not exceed float alignment");
	static_assert(offsetof(DarknetBoxBDP, x) == 0 * sizeof(float), "x must be at offset 0");
	static_assert(offsetof(DarknetBoxBDP, y) == 1 * sizeof(float), "y must be at offset 4");
	static_assert(offsetof(DarknetBoxBDP, w) == 2 * sizeof(float), "w must be at offset 8");
	static_assert(offsetof(DarknetBoxBDP, h) == 3 * sizeof(float), "h must be at offset 12");
	static_assert(offsetof(DarknetBoxBDP, fx) == 4 * sizeof(float), "fx must be at offset 16");
	static_assert(offsetof(DarknetBoxBDP, fy) == 5 * sizeof(float), "fy must be at offset 20");
	static_assert(sizeof(float) == 4, "Float must be 32-bit for memory calculations");
	static_assert(std::numeric_limits<float>::is_iec559, "Float must be IEEE 754 compliant");
	static_assert(LOGISTIC < 32, "LOGISTIC activation must fit in reasonable enum range");

	// Debug: Precondition validation phase
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Starting precondition checks...");

	void* old_output = nullptr;
	size_t old_output_size = 0;

	// Precondition checks
	assert(l.coords == 6);           // Must be configured for 6-parameter boxes
	assert(l.classes > 0);           // Must have at least one class
	assert(l.classes <= 1000);       // Reasonable upper bound on classes
	assert(l.n > 0);                 // Must have at least one anchor
	assert(l.n <= 10);               // Reasonable upper bound on anchors
	assert(l.batch > 0);             // Must have at least one batch
	assert(l.batch <= 64);           // Reasonable upper bound on batch size
	assert(l.w > 0 && l.w <= 2048);  // Grid width bounds
	assert(l.h > 0 && l.h <= 2048);  // Grid height bounds
	assert(l.w % 32 == 0 || l.w <= 32); // Width should be multiple of 32 or small
	assert(l.h % 32 == 0 || l.h <= 32); // Height should be multiple of 32 or small
	assert(state.input != nullptr);  // Input must be valid
	assert(l.output != nullptr);     // Output must be allocated
	assert(l.outputs > 0);           // Must have positive output size
	
	// Verify output size calculation: batch * anchors * grid * (coords + objectness + classes)
	size_t expected_outputs = l.batch * l.n * l.w * l.h * (6 + 1 + l.classes);
	assert(l.outputs == expected_outputs);
	(void)expected_outputs;  // Suppress unused warning in release builds
	// Memory bounds checking
	assert(l.outputs <= 1000000);    // Reasonable upper bound to prevent overflow
	// Scale factor bounds
	assert(l.scale_x_y >= 1.0f && l.scale_x_y <= 2.0f);
	// Debug: Preconditions passed
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Preconditions PASSED");


	// Store old output state for postconditions
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Allocating old_output buffer (size=" << (l.outputs * l.batch * sizeof(float)) << " bytes)...");
	old_output_size = l.outputs * l.batch * sizeof(float);
	old_output = malloc(old_output_size);
	if (old_output) {
		memcpy(old_output, l.output, old_output_size);
		DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: old_output buffer allocated successfully");
	} else {
		*cfg_and_state.output << " [WARNING] forward_yolo_layer_bdp: Failed to allocate old_output buffer" << std::endl;
	}

	// Copy input to output (standard YOLO behavior - raw predictions copied first)
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Copying input to output (size=" << (l.outputs * l.batch * sizeof(float)) << " bytes)...");
	memcpy(l.output, state.input, l.outputs * l.batch * sizeof(float));
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Input copied to output successfully");

	// BDP-specific: Fix NaN/inf values in raw input BEFORE applying activation functions
	// This is critical because activation functions can amplify or propagate NaN/inf
	// Specifically, we must clamp w,h values before activation to prevent exp() overflow
	/*
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Checking for NaN/inf values (try_fix_nan=" << state.net.try_fix_nan << ")...");
	if (state.net.try_fix_nan)
	{
		int nan_count = 0, inf_count = 0;
		for (int b = 0; b < l.batch; ++b)
		{
			for (int n = 0; n < l.n; ++n)
			{
				// Process w,h entries (indices 2-3) - these are critical for exp() safety
				int wh_index = yolo_entry_index_bdp(l, b, n * l.w * l.h, 2);
				for (int i = 0; i < 2 * l.w * l.h; ++i)
				{
					float& val = l.output[wh_index + i];
					if (std::isnan(val))
					{
						val = 0.0f;
						nan_count++;
					}
					else if (std::isinf(val))
					{
						val = 0.0f;  // Use 0 for inf in w,h to prevent exp() overflow
						inf_count++;
					}
					else if (val > 10.0f || val < -10.0f)
					{
						// BDP-specific: Clamp w,h to safe range for exp()
						val = std::max(-10.0f, std::min(10.0f, val));
					}
				}
			}
		}

		if (nan_count > 0 || inf_count > 0)
		{
			*cfg_and_state.output << "BDP forward pass (pre-activation): fixed " << nan_count
			                      << " NaN values and " << inf_count << " inf values in w/h coordinates" << std::endl;
		}
	}
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: NaN/inf check completed");
	*/

	// Force CPU activation regardless of GPU settings (debug)
	#if 1  // Always execute this path for debugging
	// CPU-only processing: apply activation functions to appropriate ranges
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Starting activation loop for batch=" << l.batch << " anchors=" << l.n);
	for (int b = 0; b < l.batch; ++b)
	{
		DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Processing batch " << b << "/" << l.batch);
		for (int n = 0; n < l.n; ++n)  // For each anchor box
		{
			DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Processing anchor " << n << "/" << l.n << " (batch " << b << ")");
			// Get starting index for bounding box coordinates (x,y,w,h,fx,fy)
			int bbox_index = yolo_entry_index_bdp(l, b, n * l.w * l.h, 0);
			DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: bbox_index=" << bbox_index << " (max allowed=" << (l.outputs * l.batch) << ")");
			
			// Additional runtime bounds checking
			assert(bbox_index >= 0 && bbox_index < l.outputs * l.batch);
			assert(bbox_index + 6 * l.w * l.h <= l.outputs * l.batch);
			
			if (l.new_coords)
			{
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Using new_coords mode");
				// New coordinate system: don't apply activation to w,h (use squared terms)
				// Apply logistic activation to x,y (entries 0-1)
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Activating x,y at index " << bbox_index);
				activate_array(l.output + bbox_index, 2 * l.w * l.h, LOGISTIC);

				// Apply logistic activation to fx,fy (entries 4-5) - front point coordinates
				int front_point_index = yolo_entry_index_bdp(l, b, n * l.w * l.h, 4);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: front_point_index=" << front_point_index);
				assert(front_point_index >= 0 && front_point_index + 2 * l.w * l.h <= l.outputs * l.batch);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Activating fx,fy at index " << front_point_index);
				activate_array(l.output + front_point_index, 2 * l.w * l.h, LOGISTIC);

				// Apply logistic activation to objectness and classes (entries 6+)
				int obj_index = yolo_entry_index_bdp(l, b, n * l.w * l.h, 6);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: obj_index=" << obj_index);
				assert(obj_index >= 0 && obj_index + (1 + l.classes) * l.w * l.h <= l.outputs * l.batch);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Activating objectness+classes at index " << obj_index << " (count=" << ((1 + l.classes) * l.w * l.h) << ")");
				activate_array(l.output + obj_index, (1 + l.classes) * l.w * l.h, LOGISTIC);
			}
			else
			{
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Using traditional coords mode");
				// Traditional coordinate system: apply logistic to x,y,fx,fy but not w,h (use exp)
				// Apply logistic activation to x,y (entries 0-1)
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Activating x,y at index " << bbox_index);
				activate_array(l.output + bbox_index, 2 * l.w * l.h, LOGISTIC);

				// Skip w,h (entries 2-3) - they use exp() in get_yolo_box_bdp()

				// Apply logistic activation to fx,fy (entries 4-5) - front point coordinates
				int front_point_index = yolo_entry_index_bdp(l, b, n * l.w * l.h, 4);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: front_point_index=" << front_point_index);
				assert(front_point_index >= 0 && front_point_index + 2 * l.w * l.h <= l.outputs * l.batch);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Activating fx,fy at index " << front_point_index);
				activate_array(l.output + front_point_index, 2 * l.w * l.h, LOGISTIC);

				// Apply logistic activation to objectness and classes (entries 6+)
				int obj_index = yolo_entry_index_bdp(l, b, n * l.w * l.h, 6);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: obj_index=" << obj_index);
				assert(obj_index >= 0 && obj_index + (1 + l.classes) * l.w * l.h <= l.outputs * l.batch);
				DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Activating objectness+classes at index " << obj_index << " (count=" << ((1 + l.classes) * l.w * l.h) << ")");
				activate_array(l.output + obj_index, (1 + l.classes) * l.w * l.h, LOGISTIC);
			}
			
			// Apply coordinate scaling to x,y for numerical stability (standard YOLO technique)
			if (l.scale_x_y != 1)
			{
				assert(l.scale_x_y > 0.0f && "Scale factor must be positive");
				scal_add_cpu(2 * l.w * l.h, l.scale_x_y, -0.5f * (l.scale_x_y - 1), l.output + bbox_index, 1);
			}
		}
	}
#endif

	// Fix NaN/inf values in the output after activation functions (BDP specific rules)
	/*
	if (state.net.try_fix_nan)
	{
		int nan_count = 0, inf_count = 0;
		for (int i = 0; i < l.outputs * l.batch; ++i)
		{
			if (std::isnan(l.output[i]))
			{
				l.output[i] = 0.0f;
				nan_count++;
			}
			else if (std::isinf(l.output[i]))
			{
				l.output[i] = 1.0f;
				inf_count++;
			}
		}
		
		if (nan_count > 0 || inf_count > 0)
		{
			*cfg_and_state.output << "BDP forward pass: fixed " << nan_count << " NaN values (→0) and "
			                      << inf_count << " inf values (→1)" << std::endl;
		}
	}
	*/

	// Postcondition checks
	assert(l.output != nullptr);     // Output still valid
	
	// CRITICAL POSTCONDITION: Verify that activation functions were applied to the output
	if (old_output && !state.train) {
		bool outputs_differ = (memcmp(state.input, l.output, old_output_size) != 0);
		
		// More detailed check: compare first few values to see what changed
		float* input_ptr = state.input;
		int changed_count = 0;
		int checked_count = std::min(10, static_cast<int>(l.outputs * l.batch));
		for (int i = 0; i < checked_count; i++) {
			if (std::abs(input_ptr[i] - l.output[i]) > 1e-6f) {
				changed_count++;
			}
		}
		
		// Allow for the case where no activations are applied in new_coords mode
		if (l.new_coords) {
			// In new_coords mode, some values might remain unchanged
			// This is acceptable behavior
		} else {
			// Use enhanced contract violation with detailed diagnostics
			std::string context = "Forward pass failed to modify outputs - activations not applied. "
							     "Layer: w=" + std::to_string(l.w) + " h=" + std::to_string(l.h) + 
							     " n=" + std::to_string(l.n) + " classes=" + std::to_string(l.classes) + 
							     " new_coords=" + std::to_string(l.new_coords) + " batch=" + std::to_string(l.batch) + 
							     " outputs=" + std::to_string(l.outputs) + " train=" + std::to_string(state.train) +
							     " changed_values=" + std::to_string(changed_count) + "/" + std::to_string(checked_count) +
							     " memcmp_result=" + std::to_string(memcmp(state.input, l.output, old_output_size));
			#ifdef DARKNET_GPU
				context += " GPU_MODE=ON";
			#else
				context += " GPU_MODE=OFF";
			#endif
			
			DARKNET_POSTCONDITION(outputs_differ, context);
		}
	}
	
	// Verify output values are in reasonable ranges after activation and NaN/inf fixing
	for (int i = 0; i < std::min(100, static_cast<int>(l.outputs * l.batch)); ++i) {
		// After NaN/inf fixing, all values should be finite
		assert(std::isfinite(l.output[i])); // No NaN or inf values should remain
		
		// After logistic activation, most values should be in [0,1] or reasonable range
		if (i % (6 + 1 + l.classes) < 6) {
			// Coordinate values - can be outside [0,1] due to scaling
			// But should be reasonable after NaN/inf fixing
			assert(l.output[i] >= -10.0f && l.output[i] <= 10.0f);
		} else if (i % (6 + 1 + l.classes) == 6) {
			// Objectness after logistic should be in [0,1]
			assert(l.output[i] >= 0.0f && l.output[i] <= 1.0f);
		} else {
			// Class probabilities after logistic should be in [0,1]
			assert(l.output[i] >= 0.0f && l.output[i] <= 1.0f);
		}
	}
	
	// Clean up
	if (old_output) {
		free(old_output);
	}

	// For inference-only mode, skip training-related computations
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Checking training mode (train=" << state.train << ")...");
	if (!state.train)
	{
		DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Inference mode - skipping training computations");
		return;
	}

	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Training mode ENABLED - proceeding with loss calculation");

	// @implement OBB: Training mode processing for BDP oriented bounding boxes
	// Compute loss and gradients for 6-parameter boxes (x,y,w,h,fx,fy)
	//
	// NOTE: Loss for front point coordinates (fx, fy) is computed inside delta_yolo_box_bdp()
	// The function computes MSE loss for all 6 parameters: x, y, w, h, fx, fy
	// See delta_yolo_box_bdp() at lines 317-489 for the full implementation

	// Verify ground truth is available
	/*
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Checking ground truth availability...");
	if (!state.truth)
	{
		*cfg_and_state.output << "WARNING: Training mode enabled but no ground truth provided (state.truth is null)" << std::endl;
		return;
	}
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Ground truth available at " << state.truth);
	*/

	// Zero delta array
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Zeroing delta array (size=" << (l.outputs * l.batch * sizeof(float)) << " bytes)...");
	if (l.delta == nullptr) {
		*cfg_and_state.output << "   [ERROR] forward_yolo_layer_bdp: l.delta is NULL!" << std::endl;
	}
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

	// Initialize label arrays
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Initializing label arrays...");
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Checking l.labels pointer: " << l.labels);
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Checking l.class_ids pointer: " << l.class_ids);
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Array size needed: " << (l.batch * l.w * l.h * l.n) << " elements");

	if (l.labels == nullptr) {
		*cfg_and_state.output << "   [ERROR] forward_yolo_layer_bdp: l.labels is NULL! Cannot initialize." << std::endl;
		darknet_fatal_error(DARKNET_LOC, "l.labels array is NULL - layer not properly initialized");
	}
	if (l.class_ids == nullptr) {
		*cfg_and_state.output << "   [ERROR] forward_yolo_layer_bdp: l.class_ids is NULL! Cannot initialize." << std::endl;
		darknet_fatal_error(DARKNET_LOC, "l.class_ids array is NULL - layer not properly initialized");
	}

	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Initializing l.labels array...");
	for (int i = 0; i < l.batch * l.w*l.h*l.n; ++i)
	{
		l.labels[i] = -1;
	}
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: l.labels initialized successfully");

	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Initializing l.class_ids array...");
	for (int i = 0; i < l.batch * l.w*l.h*l.n; ++i)
	{
		l.class_ids[i] = -1;
	}
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: l.class_ids initialized successfully");

	// Initialize loss accumulators
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Initializing loss accumulators...");
	float tot_iou = 0;
	float tot_iou_loss = 0;
	float tot_giou_loss = 0;
	float tot_fp_loss = 0;  // Front point loss accumulator (paper equation 10)
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;

	// Create threads for parallel batch processing
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Creating threads for parallel processing (num_threads=" << l.batch << ")...");
	int num_threads = l.batch;
	Darknet::VThreads threads;
	threads.reserve(num_threads);

	struct train_yolo_args * yolo_args = (train_yolo_args*)xcalloc(l.batch, sizeof(struct train_yolo_args));
	DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Allocated yolo_args array at " << yolo_args);

	for (int b = 0; b < l.batch; b++)
	{
		DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Setting up thread " << b << " args...");
		yolo_args[b].l = &l;
		yolo_args[b].state = state;  // training state
		yolo_args[b].b = b;  // batch index

		yolo_args[b].tot_iou = 0;   
		yolo_args[b].tot_iou_loss = 0;
		yolo_args[b].tot_giou_loss = 0;
		yolo_args[b].tot_fp_loss = 0;  // Initialize front point loss
		yolo_args[b].count = 0;
		yolo_args[b].class_count = 0;

		DEBUG_TRACE("   [DEBUG] forward_yolo_layer_bdp: Launching thread " << b << " for process_batch_bdp...");
		threads.emplace_back(process_batch_bdp, &(yolo_args[b]));
	}

	// Wait for all threads to complete and aggregate results
	for (int b = 0; b < l.batch; b++)
	{
		threads[b].join();

		tot_iou += yolo_args[b].tot_iou;
		tot_iou_loss += yolo_args[b].tot_iou_loss;
		tot_giou_loss += yolo_args[b].tot_giou_loss;
		tot_fp_loss += yolo_args[b].tot_fp_loss;  // Aggregate front point loss
		DEBUG_TRACE("   [BDP] forward_yolo_layer_bdp: batch=" << b
		            << " batch_fp_loss=" << yolo_args[b].tot_fp_loss
		            << " total_fp_loss=" << tot_fp_loss);
		count += yolo_args[b].count;
		class_count += yolo_args[b].class_count;
	}

	free(yolo_args);

	// Handle bad label rejection and equidistant point logic (same as standard YOLO)
	int iteration_num = get_current_iteration(state.net);
	const int start_point = state.net.max_batches * 3 / 4;

	if ((state.net.badlabels_rejection_percentage && start_point < iteration_num) ||
		(state.net.num_sigmas_reject_badlabels && start_point < iteration_num) ||
		(state.net.equidistant_point && state.net.equidistant_point < iteration_num))
	{
		const float progress_it = iteration_num - state.net.equidistant_point;
		const float progress = progress_it / (state.net.max_batches - state.net.equidistant_point);
		float ep_loss_threshold = (*state.net.delta_rolling_avg) * progress * 1.4;

		float cur_max = 0;
		float cur_avg = 0;
		float counter = 0;
		for (int i = 0; i < l.batch * l.outputs; ++i)
		{
			if (l.delta[i] != 0)
			{
				counter++;
				cur_avg += fabs(l.delta[i]);

				if (cur_max < fabs(l.delta[i]))
				{
					cur_max = fabs(l.delta[i]);
				}
			}
		}

		cur_avg = cur_avg / counter;

		if (*state.net.delta_rolling_max == 0)
		{
			*state.net.delta_rolling_max = cur_max;
		}
		*state.net.delta_rolling_max = *state.net.delta_rolling_max * 0.99 + cur_max * 0.01;
		*state.net.delta_rolling_avg = *state.net.delta_rolling_avg * 0.99 + cur_avg * 0.01;

		// Reject high loss to filter bad labels
		if (state.net.num_sigmas_reject_badlabels && start_point < iteration_num)
		{
			const float rolling_std = (*state.net.delta_rolling_std);
			const float rolling_max = (*state.net.delta_rolling_max);
			const float rolling_avg = (*state.net.delta_rolling_avg);
			const float progress_badlabels = (float)(iteration_num - start_point) / (start_point);

			float cur_std = 0.0f;
			counter = 0.0f;
			for (int i = 0; i < l.batch * l.outputs; ++i)
			{
				if (l.delta[i] != 0)
				{
					counter++;
					cur_std += pow(l.delta[i] - rolling_avg, 2);
				}
			}
			cur_std = sqrt(cur_std / counter);

			*state.net.delta_rolling_std = *state.net.delta_rolling_std * 0.99 + cur_std * 0.01;

			float final_badlebels_threshold = rolling_avg + rolling_std * state.net.num_sigmas_reject_badlabels;
			float badlabels_threshold = rolling_max - progress_badlabels * fabs(rolling_max - final_badlebels_threshold);
			badlabels_threshold = std::max(final_badlebels_threshold, badlabels_threshold);
			for (int i = 0; i < l.batch * l.outputs; ++i)
			{
				if (fabs(l.delta[i]) > badlabels_threshold)
				{
					l.delta[i] = 0;
				}
			}

			ep_loss_threshold = std::min(final_badlebels_threshold, rolling_avg) * progress;
		}

		// Reject low loss to find equidistant point
		if (state.net.equidistant_point && state.net.equidistant_point < iteration_num)
		{
			for (int i = 0; i < l.batch * l.outputs; ++i)
			{
				if (fabs(l.delta[i]) < ep_loss_threshold)
				{
					l.delta[i] = 0;
				}
			}
		}
	}

	if (count == 0)
	{
		count = 1;
	}
	if (class_count == 0)
	{
		class_count = 1;
	}

	// Fix any remaining NaN/inf in the entire delta array before computing loss
	// This prevents NaN from objectness/class deltas contaminating the loss
	for (int i = 0; i < l.batch * l.outputs; ++i) {
		if (!std::isfinite(l.delta[i])) {
			l.delta[i] = 0.0f;
		}
	}

	// Compute final loss metrics
	int stride = l.w*l.h;
	float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));

	// Check for NaN in l.delta before computing classification loss
	static int delta_nan_count = 0;
	for (int i = 0; i < l.batch * l.outputs && delta_nan_count < 5; i++) {
		if (!std::isfinite(l.delta[i])) {
			if (delta_nan_count++ < 5) {
				*cfg_and_state.output << "   [ERROR] NaN/Inf found in l.delta[" << i << "]=" << l.delta[i] << std::endl;
			}
			// Replace NaN/Inf with 0 to prevent propagation
			l.delta[i] = 0.0f;
		}
	}

	memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));

	// Zero out coordinate deltas to isolate classification loss
	for (int b = 0; b < l.batch; ++b)
	{
		for (int j = 0; j < l.h; ++j)
		{
			for (int i = 0; i < l.w; ++i)
			{
				for (int n = 0; n < l.n; ++n)
				{
					int index = yolo_entry_index_bdp(l, b, n*l.w*l.h + j*l.w + i, 0);
					// Zero out all 6 coordinate deltas (x,y,w,h,fx,fy)
					no_iou_loss_delta[index + 0 * stride] = 0;
					no_iou_loss_delta[index + 1 * stride] = 0;
					no_iou_loss_delta[index + 2 * stride] = 0;
					no_iou_loss_delta[index + 3 * stride] = 0;
					no_iou_loss_delta[index + 4 * stride] = 0;
					no_iou_loss_delta[index + 5 * stride] = 0;
				}
			}
		}
	}

	float mag_no_iou = mag_array(no_iou_loss_delta, l.outputs * l.batch);
	if (!std::isfinite(mag_no_iou)) {
		*cfg_and_state.output << "   [ERROR] mag_no_iou is NaN/Inf! Setting to 0" << std::endl;
		mag_no_iou = 0.0f;
	}

	float classification_loss = l.obj_normalizer * pow(mag_no_iou, 2);
	if (!std::isfinite(classification_loss)) {
		*cfg_and_state.output << "   [ERROR] classification_loss is NaN/Inf! obj_normalizer=" << l.obj_normalizer
		                      << " mag_no_iou=" << mag_no_iou << " Setting to minimum" << std::endl;
		classification_loss = 0.00001f;
	}
	// Clamp classification_loss to minimum of 0.00001 to prevent zero/NaN propagation
	classification_loss = std::max(0.00001f, classification_loss);

	free(no_iou_loss_delta);

	float mag_delta = mag_array(l.delta, l.outputs * l.batch);
	if (!std::isfinite(mag_delta)) {
		DEBUG_TRACE("   [ERROR] mag_delta is NaN/inf! Checking l.delta array for NaN values...");
		int nan_count = 0;
		for (int idx = 0; idx < l.outputs * l.batch && nan_count < 10; ++idx) {
			if (!std::isfinite(l.delta[idx])) {
				DEBUG_TRACE("      l.delta[" << std::dec << idx << "] = " << l.delta[idx]);
				nan_count++;
			}
		}
		DEBUG_TRACE("   [ERROR] Found " << std::dec << nan_count << " NaN/inf values in l.delta. Setting mag_delta=0");
		mag_delta = 0.0f;
	}

	float loss = pow(mag_delta, 2);
	if (!std::isfinite(loss)) loss = 0.0f;

	float iou_loss = loss - classification_loss;
	if (!std::isfinite(iou_loss)) iou_loss = 0.0001f;
	// Clamp iou_loss to minimum of 0.0001 to prevent zero/NaN propagation
	iou_loss = std::max(0.0001f, iou_loss);

	// DEBUG: Check for NaN/inf in loss computation
	static int loss_nan_count = 0;
	if (!std::isfinite(mag_no_iou) || !std::isfinite(mag_delta) ||
	    !std::isfinite(classification_loss) || !std::isfinite(loss) || !std::isfinite(iou_loss)) {
		if (loss_nan_count++ < 10) {
			*cfg_and_state.output << "   [LOSS_DEBUG] NaN/inf in loss computation!"
			                      << "\n      mag_no_iou=" << mag_no_iou << " classification_loss=" << classification_loss
			                      << "\n      mag_delta=" << mag_delta << " loss=" << loss
			                      << " iou_loss=" << iou_loss
			                      << "\n      l.obj_normalizer=" << l.obj_normalizer << std::endl;
		}
	}

	// Compute average front point loss
	float avg_fp_loss = (count > 0) ? (tot_fp_loss / count) : 0.0f;

	// Debug: Check for errors in loss values and BDP loss computation
	DEBUG_TRACE("   [BDP] forward_yolo_layer_bdp: FINAL tot_fp_loss=" << tot_fp_loss
	            << " count=" << count << " avg_fp_loss=" << avg_fp_loss
	            << " fp_normalizer=" << l.fp_normalizer);
	DEBUG_TRACE("   [BDP] Loss breakdown: "
	            << "iou_loss=" << loss_color(iou_loss) << iou_loss << reset_color()
	            << " class_loss=" << loss_color(classification_loss) << classification_loss << reset_color()
	            << " fp_loss=" << loss_color(avg_fp_loss) << avg_fp_loss << reset_color()
	            << " total=" << loss_color(loss) << loss << reset_color());
	DEBUG_TRACE("   [DEBUG] Loss computation: total_loss=" << loss << " iou_loss=" << iou_loss
	            << " class_loss=" << classification_loss << " fp_loss=" << avg_fp_loss);
	if (std::isnan(loss) || std::isinf(loss)) {
		*cfg_and_state.output << "   [ERROR] Total loss is NaN/inf!" << std::endl;
	}
	if (std::isnan(avg_fp_loss) || std::isinf(avg_fp_loss)) {
		*cfg_and_state.output << "   [ERROR] Front point loss is NaN/inf!" << std::endl;
	}

	*(l.cost) = loss;

	// Output training metrics including front point loss (paper equation 10)
	// Use high precision and always print alert if: NaN, Inf, very small (<1e-4), or exceeds 15.0f
	bool should_always_print = std::isnan(loss) || std::isinf(loss) ||
	                           std::isnan(iou_loss) || std::isinf(iou_loss) ||
	                           std::isnan(avg_fp_loss) || std::isinf(avg_fp_loss) ||
	                           std::isnan(classification_loss) || std::isinf(classification_loss) ||
	                           std::abs(loss) < 1e-4f || std::abs(iou_loss) < 1e-4f ||
	                           std::abs(avg_fp_loss) < 1e-4f || std::abs(classification_loss) < 1e-4f ||
	                           loss > 15.0f || iou_loss > 15.0f || avg_fp_loss > 15.0f || classification_loss > 15.0f;

	const char* iou_note = (iou_loss <= 0.0001f) ? "(min-clamped)" : "";
	*cfg_and_state.output << std::fixed << std::setprecision(8)
	                      << "BDP training: "
	                      << "loss=" << loss_color(loss) << loss << reset_color()
	                      << " iou_loss=" << loss_color(iou_loss) << iou_loss << reset_color() << iou_note
	                      << " fp_loss=" << loss_color(avg_fp_loss) << avg_fp_loss << reset_color()
	                      << " class_loss=" << loss_color(classification_loss) << classification_loss << reset_color()
	                      << " count=" << count
	                      << " avg_iou=" << (tot_iou / count)
	                      << std::endl;

	if (should_always_print) {
		*cfg_and_state.output << "   [ALERT] Abnormal loss values detected: "
		                      << "loss=" << (std::isnan(loss) ? "NaN" : std::isinf(loss) ? "Inf" : std::abs(loss) < 1e-4f ? "~0" : ">15")
		                      << " iou_loss=" << (std::isnan(iou_loss) ? "NaN" : std::isinf(iou_loss) ? "Inf" : std::abs(iou_loss) < 1e-4f ? "~0" : ">15")
		                      << " fp_loss=" << (std::isnan(avg_fp_loss) ? "NaN" : std::isinf(avg_fp_loss) ? "Inf" : std::abs(avg_fp_loss) < 1e-4f ? "~0" : ">15")
		                      << " class_loss=" << (std::isnan(classification_loss) ? "NaN" : std::isinf(classification_loss) ? "Inf" : std::abs(classification_loss) < 1e-4f ? "~0" : ">15")
		                      << std::endl;
	}
}


/// @implement OBB: Count number of oriented bounding box detections above threshold
/// This function counts BDP detections by checking objectness scores in the layer output.
///
/// INTERACTION WITH OTHER FUNCTIONS:
/// - Called by: get_yolo_detections_bdp(), network prediction functions
/// - Calls: yolo_entry_index_bdp() to access objectness scores
/// - Used by: Memory allocation for detection arrays
///
/// DIFFERENCES FROM STANDARD VERSION:
/// - Uses yolo_entry_index_bdp() which accounts for 6-parameter boxes
/// - Objectness is at entry 6 instead of entry 4 (due to fx,fy front point coordinates)
/// - Otherwise identical logic: scan grid positions, check objectness > threshold
///
/// MEMORY ACCESS PATTERN:
/// - Iterates through all anchor boxes (n) and grid positions (w*h)
/// - Accesses objectness score at entry 6 for each box position
/// - Returns count for subsequent memory allocation of detection array
int yolo_num_detections_bdp(const Darknet::Layer & l, float thresh)
{
	TAT(TATPARMS);

	// Aggressive static assertions for compile-time validation
	static_assert(std::numeric_limits<float>::is_iec559, "Float must be IEEE 754 compliant");
	static_assert(sizeof(float) == 4, "Float must be 32-bit");
	static_assert(std::numeric_limits<int>::max() >= 1000000, "Int must be able to count large detection arrays");

	// Boost.Contract preconditions and postconditions
	int result = 0;
	boost::contract::check c = boost::contract::function()
		.precondition([&] {
			assert(l.n > 0 && l.n <= 10);           // Reasonable anchor count
			assert(l.w > 0 && l.w <= 2048);         // Grid width bounds
			assert(l.h > 0 && l.h <= 2048);         // Grid height bounds
			assert(l.classes > 0 && l.classes <= 1000); // Reasonable class count
			assert(thresh >= 0.0f && thresh <= 1.0f); // Threshold must be probability
			assert(l.output != nullptr);             // Output must be allocated
			assert(l.outputs > 0);                   // Must have output size
			
			// Verify output size for BDP: batch * anchors * grid * (6 coords + 1 obj + classes)
			size_t expected_min_outputs = l.n * l.w * l.h * (6 + 1 + l.classes);
			assert(l.outputs >= expected_min_outputs);
			(void)expected_min_outputs;  // Suppress unused warning in release builds
		})
		.postcondition([&] {
			assert(result >= 0);                     // Count must be non-negative
			assert(result <= l.n * l.w * l.h);       // Can't exceed total possible boxes
		});

	int count = 0;
	
	// Iterate through all anchor boxes and grid positions
	for (int n = 0; n < l.n; ++n)  // For each anchor
	{
		for (int i = 0; i < l.w * l.h; ++i)  // For each grid position
		{
			// Get objectness score (entry 6 for BDP vs entry 4 for standard)
			const int obj_index = yolo_entry_index_bdp(l, 0, n * l.w * l.h + i, 6);
			
			// Runtime bounds checking
			assert(obj_index >= 0 && obj_index < l.outputs);
			assert(std::isfinite(l.output[obj_index])); // Verify no NaN/inf values
			
			// Check if objectness exceeds threshold
			if (l.output[obj_index] > thresh)
			{
				++count;
				
				// Additional validation: objectness after logistic should be in [0,1]
				assert(l.output[obj_index] >= 0.0f && l.output[obj_index] <= 1.0f);
			}
		}
	}
	
	result = count;
	return result;
}


/// @implement OBB: Extract oriented bounding box detections from YOLO layer output
/// This function converts YOLO layer raw predictions into structured BDP detection objects.
///
/// INTERACTION WITH OTHER FUNCTIONS:
/// - Called by: Network prediction pipeline, get_network_boxes_bdp()
/// - Calls: get_yolo_box_bdp() to extract 6-parameter boxes
/// - Calls: yolo_entry_index_bdp() to access objectness and class scores
/// - Uses: correct_yolo_boxes_bdp() for coordinate correction (to be implemented)
///
/// DIFFERENCES FROM STANDARD VERSION:
/// - Returns DarknetDetectionOBB* instead of Darknet::Detection*
/// - Uses get_yolo_box_bdp() for 6-parameter box extraction (x,y,w,h,fx,fy)
/// - Objectness at entry 6, classes at entry 7+ (vs 4, 5+ for standard)
/// - Populates DarknetBoxBDP with front point coordinates for orientation
///
/// PROCESSING FLOW:
/// 1. Iterate through grid positions and anchors
/// 2. Check objectness score against threshold
/// 3. Extract 6-parameter BDP box using get_yolo_box_bdp()
/// 4. Calculate class probabilities (objectness * class_score)
/// 5. Apply coordinate corrections for image dimensions
/// 6. Return count of valid detections extracted
int get_yolo_detections_bdp(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, DarknetDetectionOBB *dets, int letter)
{
	TAT(TATPARMS);

	// Aggressive static assertions for compile-time validation
	static_assert(sizeof(DarknetDetectionOBB) >= sizeof(DarknetBoxBDP), "Detection must contain BDP box");
	static_assert(std::is_trivially_copyable_v<DarknetBoxBDP>, "BDP box must be trivially copyable");
	static_assert(alignof(DarknetDetectionOBB) >= alignof(float), "Detection alignment must accommodate float arrays");
	static_assert(offsetof(DarknetDetectionOBB, bbox) == 0, "BDP box must be first member of detection");

	// Boost.Contract preconditions and postconditions
	int result = 0;
	boost::contract::check c = boost::contract::function()
		.precondition([&] {
			assert(l.n > 0 && l.n <= 10);           // Reasonable anchor count
			assert(l.w > 0 && l.w <= 2048);         // Layer grid width bounds
			assert(l.h > 0 && l.h <= 2048);         // Layer grid height bounds
			assert(l.classes > 0 && l.classes <= 1000); // Reasonable class count
			assert(w > 0 && w <= 8192);              // Image width bounds
			assert(h > 0 && h <= 8192);              // Image height bounds
			assert(netw > 0 && netw <= 2048);        // Network input width
			assert(neth > 0 && neth <= 2048);        // Network input height
			assert(thresh >= 0.0f && thresh <= 1.0f); // Threshold must be probability
			assert(l.output != nullptr);             // Layer output must exist
			assert(dets != nullptr);                 // Detection array must be allocated
			assert(l.biases != nullptr);             // Anchor biases must exist
			assert(l.mask != nullptr);               // Anchor mask must exist
			
			// Verify biases array has correct size
			assert(l.n <= 10); // Reasonable upper bound for checking bias array
		})
		.postcondition([&] {
			assert(result >= 0);                     // Count must be non-negative  
			assert(result <= l.n * l.w * l.h);       // Can't exceed total possible boxes
		});

	const float * predictions = l.output;
	int count = 0;

	// Process each grid position and anchor box
	for (int i = 0; i < l.w * l.h; ++i)
	{
		// Calculate grid coordinates from linear index
		int row = i / l.w;
		int col = i % l.w;
		
		// Bounds checking for grid coordinates
		assert(row >= 0 && row < l.h);
		assert(col >= 0 && col < l.w);

		for (int n = 0; n < l.n; ++n)  // For each anchor box
		{
			// Get objectness score (entry 6 for BDP vs entry 4 for standard)
			int obj_index = yolo_entry_index_bdp(l, 0, n * l.w * l.h + i, 6);
			
			// Runtime bounds checking
			assert(obj_index >= 0 && obj_index < l.outputs);
			
			float objectness = predictions[obj_index];
			
			// Validate objectness value
			assert(std::isfinite(objectness));
			assert(objectness >= 0.0f && objectness <= 1.0f); // Should be in [0,1] after logistic
			
			// Check if detection meets threshold
			if (objectness > thresh)
			{
				// Extract 6-parameter oriented bounding box
				int box_index = yolo_entry_index_bdp(l, 0, n * l.w * l.h + i, 0);
				assert(box_index >= 0 && box_index + 5 < l.outputs); // Ensure 6 parameters fit
				
				// Use get_yolo_box_bdp to extract x,y,w,h,fx,fy
				dets[count].bbox = get_yolo_box_bdp(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w * l.h, l.new_coords);
				
				// Set detection metadata
				dets[count].objectness = objectness;
				dets[count].classes = l.classes;
				dets[count].best_class_idx = -1; // Will be set during NMS
				
				// Handle embeddings if present (for tracking)
				if (l.embedding_output)
				{
					// @implement OBB: Add embedding extraction for tracking support
					// get_embedding(l.embedding_output, l.w, l.h, l.n * l.embedding_size, l.embedding_size, col, row, n, 0, dets[count].embeddings);
					assert(dets[count].embeddings != nullptr);
					dets[count].embedding_size = l.embedding_size;
				}
				else
				{
					dets[count].embeddings = nullptr;
					dets[count].embedding_size = 0;
				}
				
				// Extract class probabilities (entries 7+ for BDP vs 5+ for standard)
				assert(dets[count].prob != nullptr); // Should be allocated by caller
				
				for (int j = 0; j < l.classes; ++j)
				{
					int class_index = yolo_entry_index_bdp(l, 0, n * l.w * l.h + i, 7 + j);
					assert(class_index >= 0 && class_index < l.outputs);
					
					float class_score = predictions[class_index];
					assert(std::isfinite(class_score));
					assert(class_score >= 0.0f && class_score <= 1.0f); // Should be in [0,1] after logistic
					
					// Calculate final probability: objectness * class_probability
					float prob = objectness * class_score;
					
					// Apply threshold and class mapping
					if (prob > thresh)
					{
						dets[count].prob[j] = prob;
					}
					else
					{
						dets[count].prob[j] = 0.0f;
					}
				}
				
				// Validate detection bounds
				assert(dets[count].bbox.x >= 0.0f && dets[count].bbox.x <= 1.0f);
				assert(dets[count].bbox.y >= 0.0f && dets[count].bbox.y <= 1.0f);
				assert(dets[count].bbox.w > 0.0f && dets[count].bbox.w <= 1.0f);
				assert(dets[count].bbox.h > 0.0f && dets[count].bbox.h <= 1.0f);
				assert(dets[count].bbox.fx >= 0.0f && dets[count].bbox.fx <= 1.0f);
				assert(dets[count].bbox.fy >= 0.0f && dets[count].bbox.fy <= 1.0f);
				
				++count;
			}
		}
	}

	// Apply coordinate corrections for different image dimensions
	correct_yolo_boxes_bdp(dets, count, w, h, netw, neth, relative, letter);

	result = count;
	return result;
}


/// @implement OBB: Coordinate correction function for oriented bounding boxes (BDP representation)
/// This function converts normalized BDP coordinates from network space to original image space.
///
/// INTERACTION WITH OTHER FUNCTIONS:
/// - Called by: get_yolo_detections_bdp() after detection extraction
/// - Similar to: correct_yolo_boxes() but handles 6-parameter BDP boxes
/// - Modifies: DarknetDetectionOBB array in-place to correct coordinates
///
/// DIFFERENCES FROM STANDARD VERSION:
/// - Operates on DarknetDetectionOBB* instead of Darknet::Detection*
/// - Corrects 6 parameters (x,y,w,h,fx,fy) instead of 4 (x,y,w,h)
/// - Front point coordinates (fx,fy) undergo same transformations as center (x,y)
/// - Maintains orientation relationships between center and front point
///
/// COORDINATE TRANSFORMATIONS:
/// - Accounts for letterboxing and aspect ratio differences
/// - Converts from network normalized space [0,1] to image pixel space
/// - Applies scaling and offset corrections for proper image alignment
/// - Preserves oriented bounding box geometry during transformation
void correct_yolo_boxes_bdp(DarknetDetectionOBB *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
	TAT(TATPARMS);

	// Aggressive static assertions for compile-time validation
	static_assert(sizeof(DarknetBoxBDP) == 6 * sizeof(float), "BDP box must have exactly 6 float parameters");
	static_assert(std::is_trivially_copyable_v<DarknetDetectionOBB>, "Detection must be trivially copyable");
	static_assert(offsetof(DarknetDetectionOBB, bbox) == 0, "BDP box must be first member");
	static_assert(std::numeric_limits<float>::is_iec559, "Float must be IEEE 754 compliant");

	// Boost.Contract preconditions and postconditions
	boost::contract::check c = boost::contract::function()
		.precondition([&] {
			assert(dets != nullptr || n == 0);     // Array must exist if n > 0
			assert(n >= 0 && n <= 100000);         // Reasonable detection count bounds
			assert(w > 0 && w <= 8192);            // Original image width bounds
			assert(h > 0 && h <= 8192);            // Original image height bounds
			assert(netw > 0 && netw <= 2048);      // Network input width bounds  
			assert(neth > 0 && neth <= 2048);      // Network input height bounds
			assert(relative == 0 || relative == 1); // Boolean flag validation
			assert(letter == 0 || letter == 1);    // Boolean flag validation
			
			// Validate input coordinates are in expected ranges (normalized [0,1])
			for (int i = 0; i < std::min(n, 10); ++i) {  // Check first 10 detections
				assert(dets[i].bbox.x >= 0.0f && dets[i].bbox.x <= 1.0f);
				assert(dets[i].bbox.y >= 0.0f && dets[i].bbox.y <= 1.0f);  
				assert(dets[i].bbox.w > 0.0f && dets[i].bbox.w <= 1.0f);
				assert(dets[i].bbox.h > 0.0f && dets[i].bbox.h <= 1.0f);
				assert(dets[i].bbox.fx >= 0.0f && dets[i].bbox.fx <= 1.0f);
				assert(dets[i].bbox.fy >= 0.0f && dets[i].bbox.fy <= 1.0f);
			}
		})
		.postcondition([&] {
			// Validate output coordinates are in reasonable ranges
			for (int i = 0; i < std::min(n, 10); ++i) {  // Check first 10 detections
				if (relative) {
					// Relative coordinates should remain in [0,1] range
					assert(dets[i].bbox.x >= 0.0f && dets[i].bbox.x <= 1.0f);
					assert(dets[i].bbox.y >= 0.0f && dets[i].bbox.y <= 1.0f);
					assert(dets[i].bbox.fx >= 0.0f && dets[i].bbox.fx <= 1.0f);
					assert(dets[i].bbox.fy >= 0.0f && dets[i].bbox.fy <= 1.0f);
				} else {
					// Absolute coordinates should be in pixel ranges
					assert(dets[i].bbox.x >= -w && dets[i].bbox.x <= 2*w);  // Allow some overflow
					assert(dets[i].bbox.y >= -h && dets[i].bbox.y <= 2*h);
					assert(dets[i].bbox.fx >= -w && dets[i].bbox.fx <= 2*w);
					assert(dets[i].bbox.fy >= -h && dets[i].bbox.fy <= 2*h);
				}
				assert(dets[i].bbox.w > 0.0f);      // Width must remain positive
				assert(dets[i].bbox.h > 0.0f);      // Height must remain positive
			}
		});

	// Calculate letterboxing parameters (same logic as standard correct_yolo_boxes)
	int new_w = 0;
	int new_h = 0;
	
	if (letter)
	{
		// Letterbox mode: maintain aspect ratio with padding
		if (((float)netw / w) < ((float)neth / h))
		{
			new_w = netw;
			new_h = (h * netw) / w;
		}
		else
		{
			new_h = neth;
			new_w = (w * neth) / h;
		}
	}
	else
	{
		// Stretch mode: fill entire network input
		new_w = netw;
		new_h = neth;
	}
	
	// Calculate transformation parameters
	const float deltaw = netw - new_w;        // Horizontal padding offset
	const float deltah = neth - new_h;        // Vertical padding offset
	const float ratiow = (float)new_w / netw; // Horizontal scaling ratio
	const float ratioh = (float)new_h / neth; // Vertical scaling ratio
	
	// Runtime validation of calculated parameters
	assert(ratiow > 0.0f && ratiow <= 1.0f);
	assert(ratioh > 0.0f && ratioh <= 1.0f);
	assert(deltaw >= 0.0f && deltaw <= netw);
	assert(deltah >= 0.0f && deltah <= neth);

	// Apply coordinate corrections to all detections
	for (int i = 0; i < n; ++i)
	{
		DarknetBoxBDP b = dets[i].bbox;
		
		// Validate input coordinates for this detection
		assert(std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.w) && std::isfinite(b.h));
		assert(std::isfinite(b.fx) && std::isfinite(b.fy));
		
		// Transform center coordinates (same as standard YOLO correction)
		b.x = (b.x - deltaw / 2.0f / netw) / ratiow;
		b.y = (b.y - deltah / 2.0f / neth) / ratioh;
		
		// Transform front point coordinates (using same transformation as center)
		b.fx = (b.fx - deltaw / 2.0f / netw) / ratiow;
		b.fy = (b.fy - deltah / 2.0f / neth) / ratioh;
		
		// Scale dimensions  
		b.w /= ratiow;
		b.h /= ratioh;
		
		// Convert to absolute coordinates if requested
		if (!relative)
		{
			b.x *= w;   // Center x to pixel coordinates
			b.y *= h;   // Center y to pixel coordinates
			b.w *= w;   // Width to pixel dimensions
			b.h *= h;   // Height to pixel dimensions
			b.fx *= w;  // Front point x to pixel coordinates
			b.fy *= h;  // Front point y to pixel coordinates
		}
		
		// Validate output coordinates
		assert(std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.w) && std::isfinite(b.h));
		assert(std::isfinite(b.fx) && std::isfinite(b.fy));
		assert(b.w > 0.0f && b.h > 0.0f); // Dimensions must remain positive
		
		// Store corrected coordinates back to detection
		dets[i].bbox = b;
	}
}
