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
			if (is_abnormal || cfg_and_state.is_verbose) { \
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
			if (is_abnormal || cfg_and_state.is_verbose) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[LOSS_FP iter=" << (iter) << "] " \
									  << "fp_loss=" << value_color(fp_val) << fp_val << reset_color() \
									  << " " << __VA_ARGS__ << std::endl; \
			} \
		}

	#define GRAD_LOSS_RIOU(iter, dx_val, dy_val, dw_val, dh_val, dfx_val, dfy_val) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool grad_has_nan = !std::isfinite(dx_val) || !std::isfinite(dy_val) || !std::isfinite(dw_val) || \
						   !std::isfinite(dh_val) || !std::isfinite(dfx_val) || !std::isfinite(dfy_val); \
			bool grad_all_zero = (std::abs(dx_val) < 1e-8f && std::abs(dy_val) < 1e-8f && \
							 std::abs(dw_val) < 1e-8f && std::abs(dh_val) < 1e-8f && \
							 std::abs(dfx_val) < 1e-8f && std::abs(dfy_val) < 1e-8f); \
			bool grad_very_large = std::abs(dx_val) > 2.0f || std::abs(dy_val) > 2.0f || std::abs(dw_val) > 2.0f || \
							  std::abs(dh_val) > 2.0f || std::abs(dfx_val) > 2.0f || std::abs(dfy_val) > 2.0f; \
			bool is_abnormal = grad_has_nan || grad_all_zero || grad_very_large; \
			if (is_abnormal || cfg_and_state.is_verbose) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[GRAD_LOSS_RIOU iter=" << (iter) << "] anchor=" << n << " grid=(" << i << "," << j << ") " \
									  << "dx=" << value_color(dx_val) << dx_val << reset_color() \
									  << " dy=" << value_color(dy_val) << dy_val << reset_color() \
									  << " dw=" << value_color(dw_val) << dw_val << reset_color() \
									  << " dh=" << value_color(dh_val) << dh_val << reset_color() \
									  << " dfx=" << value_color(dfx_val) << dfx_val << reset_color() \
									  << " dfy=" << value_color(dfy_val) << dfy_val << reset_color(); \
				if (grad_has_nan) *cfg_and_state.output << " [NaN!]"; \
				if (grad_all_zero) *cfg_and_state.output << " [ALL_ZERO!]"; \
				if (grad_very_large) *cfg_and_state.output << " [VERY_LARGE!]"; \
				*cfg_and_state.output << std::endl; \
			} \
		}

	#define GRAD_LOSS_CLASS(iter, class_val, class_id_val) \
		if ((iter) <= BDP_DEBUG_MAX_ITER) { \
			bool grad_has_nan = !std::isfinite(class_val); \
			bool grad_all_zero = std::abs(class_val) < 1e-8f; \
			bool grad_very_large = std::abs(class_val) > 2.0f; \
			bool is_abnormal = grad_has_nan || grad_all_zero || grad_very_large; \
			if (is_abnormal || (iter) <= 5) { \
				*cfg_and_state.output << std::fixed << std::setprecision(6) \
									  << "[GRAD_LOSS_CLASS iter=" << (iter) << "] anchor=" << n << " grid=(" << i << "," << j << ") " \
									  << "class_id=" << (class_id_val) \
									  << " grad=" << value_color(class_val) << class_val << reset_color(); \
				if (grad_has_nan) *cfg_and_state.output << " [NaN!]"; \
				if (grad_all_zero) *cfg_and_state.output << " [ALL_ZERO!]"; \
				if (grad_very_large) *cfg_and_state.output << " [VERY_LARGE!]"; \
				*cfg_and_state.output << std::endl; \
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

	static inline void fix_nan_inf(float & val)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		// First check using standard library functions
		if (std::isnan(val) or std::isinf(val))
		{
			val = 0.0f;
		}

		// Additional manual binary check as safety (std::isnan/std::isinf may be broken)
		// NaN/Inf have all exponent bits set (exponent = 0xFF)
		// We use uint32_t to inspect the raw bit pattern of the float (32 bits)
		// Float layout: [sign:1bit][exponent:8bits][mantissa:23bits]
		// memcpy copies the float's bytes into uint32_t without conversion
		uint32_t bits;
		memcpy(&bits, &val, sizeof(float));
		uint32_t exponent = (bits >> 23) & 0xFF;  // Extract 8 exponent bits
		if (exponent == 0xFF)  // All 1s = NaN or Inf
		{
			val = 0.0f;
		}

		return;
	}

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
