#include "darknet_internal.hpp"

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();


	static inline cv::Rect2f darknet_box_to_cv_rect(const Darknet::Box & b)
	{
		TAT_REVIEWED(TATPARMS, "2025-04-26 inlined");

		const float w = b.w;
		const float h = b.h;
		const float x = b.x - (b.w * 0.5f);
		const float y = b.y - (b.h * 0.5f);

		return cv::Rect2f(x, y, w, h);
	}


	static inline dbox derivative(const Darknet::Box & a, const Darknet::Box & b)
	{
		TAT_REVIEWED(TATPARMS, "2024-03-19 inlined");
		// this function is only used in this file

		dbox d;
		d.dx = a.x < b.x ? 1.0 : -1.0;
		d.dy = a.y < b.y ? 1.0 : -1.0;
		d.dw = a.w < b.w ? 1.0 : -1.0;
		d.dh = a.h < b.h ? 1.0 : -1.0;

		return d;
	}


	/// where c is the smallest box that fully encompases a and b
	static inline boxabs box_c(const Darknet::Box & a, const Darknet::Box & b)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12 inlined");
		// this function is only used in this file

		boxabs ba;
		ba.top		= fmin(a.y - a.h / 2.0f, b.y - b.h / 2.0f);
		ba.bot		= fmax(a.y + a.h / 2.0f, b.y + b.h / 2.0f);
		ba.left		= fmin(a.x - a.w / 2.0f, b.x - b.w / 2.0f);
		ba.right	= fmax(a.x + a.w / 2.0f, b.x + b.w / 2.0f);

		return ba;
	}


	/// representation from x, y, w, h to top, left, bottom, right
	static inline boxabs to_tblr(const Darknet::Box & a)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12 inlined");
		// this function is only used in this file

		boxabs tblr;
		tblr.top	= a.y - (a.h / 2.0f);
		tblr.bot	= a.y + (a.h / 2.0f);
		tblr.left	= a.x - (a.w / 2.0f);
		tblr.right	= a.x + (a.w / 2.0f);

		return tblr;
	}


	static inline float overlap(const float x1, const float w1, const float x2, const float w2)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12 inlined");
		// this function is only used in this file

		const float l1 = x1 - w1 / 2.0f;
		const float l2 = x2 - w2 / 2.0f;
		const float left = std::max(l1, l2);

		const float r1 = x1 + w1 / 2.0f;
		const float r2 = x2 + w2 / 2.0f;
		const float right = std::min(r1, r2);

		return right - left;
	}


	static inline float box_intersection(const Darknet::Box & a, const Darknet::Box & b)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12 inlined");
		// this function is only used in this file

		const float w = overlap(a.x, a.w, b.x, b.w);
		if (w <= 0.0f)
		{
			return 0.0f;
		}

		const float h = overlap(a.y, a.h, b.y, b.h);
		if (h <= 0.0f)
		{
			return 0.0f;
		}

		return w * h;
	}


	static inline float box_union(const Darknet::Box & a, const Darknet::Box & b, const float intersection)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12 inlined");
		// this function is only used in this file

		const float u = a.w * a.h + b.w * b.h - intersection;

		return u;
	}


	static inline float box_union(const Darknet::Box & a, const Darknet::Box & b)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12 inlined");
		// this function is only used in this file

		const float i = box_intersection(a, b);
		const float u = a.w * a.h + b.w * b.h - i;

		return u;
	}


	static inline float box_diounms(const Darknet::Box & a, const Darknet::Box & b, const float beta1)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12");
		// this function is only used in this file

		const boxabs ba = box_c(a, b);
		const float w = ba.right - ba.left;
		const float h = ba.bot - ba.top;
		const float c = w * w + h * h;
		const float iou = box_iou(a, b);
		if (c == 0.0f)
		{
			return iou;
		}

		float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
		float diou_term = pow(d / c, beta1);

		return iou - diou_term;
	}


	static inline dbox dintersect(const Darknet::Box & a, const Darknet::Box & b)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12");
		// this function is only used in this file

		const float w = overlap(a.x, a.w, b.x, b.w);
		const float h = overlap(a.y, a.h, b.y, b.h);
		const dbox dover = derivative(a, b);

		dbox di;
		di.dw = dover.dw * h;
		di.dx = dover.dx * h;
		di.dh = dover.dh * w;
		di.dy = dover.dy * w;

		return di;
	}


	static inline dbox dunion(const Darknet::Box & a, const Darknet::Box & b)
	{
		TAT_REVIEWED(TATPARMS, "2024-05-12");
		// this function is only used in this file

		const dbox di = dintersect(a, b);

		dbox du;
		du.dw = a.h - di.dw;
		du.dh = a.w - di.dh;
		du.dx = -di.dx;
		du.dy = -di.dy;

		return du;
	}


	struct sortable_bbox
	{
		int index;
		int class_id;
		float **probs;
	};


	static inline int nms_comparator(const void *pa, const void *pb)
	{
		TAT(TATPARMS);
		// this is only called from 1 place

		sortable_bbox a = *(sortable_bbox *)pa;
		sortable_bbox b = *(sortable_bbox *)pb;

		float diff = a.probs[a.index][b.class_id] - b.probs[b.index][b.class_id];

		if(diff < 0.0f)
		{
			return 1;
		}

		if(diff > 0.0f)
		{
			return -1;
		}

		return 0;
	}


	static inline void sort_box_detections(Darknet::Detection * dets, const int total)
	{
		TAT(TATPARMS);

		if (total > 1)
		{
			// We want to sort from high probability to low probability.  The default sort behaviour would be to
			// sort from low to high.  We reverse the sort order by comparing RHS to LHS instead of LHS to RHS.

			std::sort(/** @todo try this again in 2026? std::execution::par_unseq,*/ dets, dets + total,
					[](const Darknet::Detection & lhs, const Darknet::Detection & rhs) -> bool
					{
						if (rhs.sort_class < 0)
						{
							return rhs.objectness < lhs.objectness;
						}

						return rhs.prob[rhs.sort_class] < lhs.prob[rhs.sort_class];
					});
		}
	}

	/// @implement OBB: Sort oriented bounding box detections by probability
	static inline void sort_box_detections_bdp(DarknetDetectionOBB * dets, const int total)
	{
		TAT(TATPARMS);

		if (total > 1)
		{
			// Sort from high probability to low probability
			std::sort(dets, dets + total,
					[](const DarknetDetectionOBB & lhs, const DarknetDetectionOBB & rhs) -> bool
					{
						if (rhs.sort_class < 0)
						{
							return rhs.objectness < lhs.objectness;
						}

						return rhs.prob[rhs.sort_class] < lhs.prob[rhs.sort_class];
					});
		}
	}

} // anonymous namespace


Darknet::Box float_to_box(const float * f)
{
	TAT_REVIEWED(TATPARMS, "2024-05-12");
	// this function is used in several places

	Darknet::Box b;
	b.x = f[0];
	b.y = f[1];
	b.w = f[2];
	b.h = f[3];

	return b;
}


Darknet::Box float_to_box_stride(const float *f, const int stride)
{
	TAT_REVIEWED(TATPARMS, "2024-05-12");
	// this function is used in several places

	Darknet::Box b;
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];

	return b;
}


float box_iou_kind(const Darknet::Box & a, const Darknet::Box & b, const IOU_LOSS iou_kind)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");
	// this function is used in several places

	//IOU, GIOU, MSE, DIOU, CIOU
	switch(iou_kind)
	{
		case IOU:	return box_iou(a, b);
		case GIOU:	return box_giou(a, b);
		case DIOU:	return box_diou(a, b);
		case CIOU:	return box_ciou(a, b);
		default:	break;
	}

	return box_iou(a, b);
}


float box_iou(const Darknet::Box & a, const Darknet::Box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	/* This function is used in many places.  It is the function at the very top of the list when looking at the TAT
	 * results while training.  Speed is quick, but the total amount of times it gets called is the reason it is the
	 * function with the longest time.
	 */

#if 1
	const float I = box_intersection(a, b);
	if (I == 0.0f)
	{
		return 0.0f;
	}

	const float U = box_union(a, b, I);

#else

	const auto ra = darknet_box_to_cv_rect(a);
	const auto rb = darknet_box_to_cv_rect(b);
	const float I = (ra & rb).area();

	if (I == 0.0f)
	{
		return 0.0f;
	}

	const float U = ra.area() + rb.area() - I;
#endif

	// if intersection is non-zero, then union will of course be non-zero, so no need to worry about divide-by-zero
	return I / U;

}


float Darknet::iou(const cv::Rect2f & lhs, const cv::Rect2f & rhs)
{
	TAT_REVIEWED(TATPARMS, "2025-05-04");

	float intersection_over_union = 0.0f;

	const float intersection = (lhs & rhs).area();

	// if the intersection is zero, then don't bother with the rest, we know the answer will be zero
	if (intersection > 0.0f)
	{
		intersection_over_union = intersection / (lhs.area() + rhs.area() - intersection);
	}

	return intersection_over_union;
}


float Darknet::iou(const cv::Rect & lhs, const cv::Rect & rhs)
{
	TAT_REVIEWED(TATPARMS, "2024-09-07");

	float intersection_over_union = 0.0f;

#if 1
	// see: https://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap/9325084

	const auto & tl1 = lhs.tl();	// blue_triangle
	const auto & tl2 = rhs.tl();	// orange_triangle
	const auto & br1 = lhs.br();	// blue_circle
	const auto & br2 = rhs.br();	// orange_circle

	const float intersection =
			std::max(0, std::min(br1.x, br2.x) - std::max(tl1.x, tl2.x)) *
			std::max(0, std::min(br1.y, br2.y) - std::max(tl1.y, tl2.y));

	// if the intersection is zero, then don't bother with the rest, we know the answer will be zero
	if (intersection > 0.0f)
	{
		intersection_over_union = intersection / (lhs.area() + rhs.area() - intersection);
	}

#else

	// 2025-05-04:  This next implementation produces the exact same results, but tests show that it is slightly slower.

	const float r_intersection = (lhs & rhs).area();
	if (r_intersection > 0.0f)
	{
		// if intersection is non-zero, then union will also be non-zero,
		// so no need to worry about divide-by-zero
		const float r_union = lhs.area() + rhs.area() - r_intersection;
		intersection_over_union = r_intersection / r_union;
	}
#endif

	return intersection_over_union;
}


float box_giou(const Darknet::Box & a, const Darknet::Box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-05-12");
	// this function is used in several places

	boxabs ba = box_c(a, b);
	const float w = ba.right - ba.left;
	const float h = ba.bot - ba.top;
	const float c = w * h;

	const float iou = box_iou(a, b);
	if (c == 0.0f)
	{
		return iou;
	}

	const float u = box_union(a, b);
	const float giou_term = (c - u) / c;

	return iou - giou_term;
}


float box_diou(const Darknet::Box & a, const Darknet::Box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-05-12");
	// this function is used in several places

	/// https://github.com/Zzh-tju/DIoU-darknet
	/// https://arxiv.org/abs/1911.08287

	const boxabs ba = box_c(a, b);
	const float w = ba.right - ba.left;
	const float h = ba.bot - ba.top;
	const float c = w * w + h * h;
	const float iou = box_iou(a, b);
	if (c == 0.0f)
	{
		return iou;
	}

	const float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
	const float diou_term = pow(d / c, 0.6);

	return iou - diou_term;
}


float box_ciou(const Darknet::Box & a, const Darknet::Box & b)
{
	TAT(TATPARMS);
	// this function is used in several places

	// https://github.com/Zzh-tju/DIoU-darknet
	// https://arxiv.org/abs/1911.08287

	const boxabs ba = box_c(a, b);
	const float w = ba.right - ba.left;
	const float h = ba.bot - ba.top;
	const float c = w * w + h * h;
	const float iou = box_iou(a, b);
	if (c == 0.0f)
	{
		return iou;
	}

	const float u = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
	const float d = u / c;
	const float ar_gt = b.w / b.h;
	const float ar_pred = a.w / a.h;
	const float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
	const float alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
	const float ciou_term = d + alpha * ar_loss;                   //ciou

	return iou - ciou_term;
}


dxrep dx_box_iou(const Darknet::Box & pred, const Darknet::Box & truth, const IOU_LOSS iou_loss)
{
	TAT(TATPARMS);
	// this function is used in several places

	const boxabs pred_tblr = to_tblr(pred);
	const float pred_t = fmin(pred_tblr.top, pred_tblr.bot);
	const float pred_b = fmax(pred_tblr.top, pred_tblr.bot);
	const float pred_l = fmin(pred_tblr.left, pred_tblr.right);
	const float pred_r = fmax(pred_tblr.left, pred_tblr.right);
	const boxabs truth_tblr = to_tblr(truth);

	float X = (pred_b - pred_t) * (pred_r - pred_l);
	float Xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
	float Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
	float Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
	float I = Iw * Ih;
	float U = X + Xhat - I;
	float S = (pred.x-truth.x)*(pred.x-truth.x)+(pred.y-truth.y)*(pred.y-truth.y);
	float giou_Cw = fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
	float giou_Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
	float giou_C = giou_Cw * giou_Ch;

	//Partial Derivatives, derivatives
	float dX_wrt_t = -1 * (pred_r - pred_l);
	float dX_wrt_b = pred_r - pred_l;
	float dX_wrt_l = -1 * (pred_b - pred_t);
	float dX_wrt_r = pred_b - pred_t;

	// gradient of I min/max in IoU calc (prediction)
	float dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
	float dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
	float dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
	float dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
	// derivative of U with regard to x
	float dU_wrt_t = dX_wrt_t - dI_wrt_t;
	float dU_wrt_b = dX_wrt_b - dI_wrt_b;
	float dU_wrt_l = dX_wrt_l - dI_wrt_l;
	float dU_wrt_r = dX_wrt_r - dI_wrt_r;
	// gradient of C min/max in IoU calc (prediction)
	float dC_wrt_t = pred_t < truth_tblr.top ? (-1 * giou_Cw) : 0;
	float dC_wrt_b = pred_b > truth_tblr.bot ? giou_Cw : 0;
	float dC_wrt_l = pred_l < truth_tblr.left ? (-1 * giou_Ch) : 0;
	float dC_wrt_r = pred_r > truth_tblr.right ? giou_Ch : 0;

	float p_dt = 0;
	float p_db = 0;
	float p_dl = 0;
	float p_dr = 0;
	if (U > 0 )
	{
		p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
		p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
		p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
		p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
	}

	// apply grad from prediction min/max for correct corner selection
	p_dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
	p_db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
	p_dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
	p_dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

	if (iou_loss == GIOU)
	{
		if (giou_C > 0)
		{
			// apply "C" term from gIOU
			p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
			p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
			p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
			p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
		}

		if (Iw<=0||Ih<=0)
		{
			p_dt = ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
			p_db = ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
			p_dl = ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
			p_dr = ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
		}
	}

	float Ct = fmin(pred.y - pred.h / 2,truth.y - truth.h / 2);
	float Cb = fmax(pred.y + pred.h / 2,truth.y + truth.h / 2);
	float Cl = fmin(pred.x - pred.w / 2,truth.x - truth.w / 2);
	float Cr = fmax(pred.x + pred.w / 2,truth.x + truth.w / 2);
	float Cw = Cr - Cl;
	float Ch = Cb - Ct;
	float C = Cw * Cw + Ch * Ch;

	float dCt_dx = 0;
	float dCt_dy = pred_t < truth_tblr.top ? 1 : 0;
	float dCt_dw = 0;
	float dCt_dh = pred_t < truth_tblr.top ? -0.5 : 0;

	float dCb_dx = 0;
	float dCb_dy = pred_b > truth_tblr.bot ? 1 : 0;
	float dCb_dw = 0;
	float dCb_dh = pred_b > truth_tblr.bot ? 0.5: 0;

	float dCl_dx = pred_l < truth_tblr.left ? 1 : 0;
	float dCl_dy = 0;
	float dCl_dw = pred_l < truth_tblr.left ? -0.5 : 0;
	float dCl_dh = 0;

	float dCr_dx = pred_r > truth_tblr.right ? 1 : 0;
	float dCr_dy = 0;
	float dCr_dw = pred_r > truth_tblr.right ? 0.5 : 0;
	float dCr_dh = 0;

	float dCw_dx = dCr_dx - dCl_dx;
	float dCw_dy = dCr_dy - dCl_dy;
	float dCw_dw = dCr_dw - dCl_dw;
	float dCw_dh = dCr_dh - dCl_dh;

	float dCh_dx = dCb_dx - dCt_dx;
	float dCh_dy = dCb_dy - dCt_dy;
	float dCh_dw = dCb_dw - dCt_dw;
	float dCh_dh = dCb_dh - dCt_dh;

	// Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
	float p_dx = 0;
	float p_dy = 0;
	float p_dw = 0;
	float p_dh = 0;

	p_dx = p_dl + p_dr;           //p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
	p_dy = p_dt + p_db;
	p_dw = (p_dr - p_dl);         //For dw and dh, we do not divided by 2.
	p_dh = (p_db - p_dt);

	// https://github.com/Zzh-tju/DIoU-darknet
	// https://arxiv.org/abs/1911.08287
	if (iou_loss == DIOU)
	{
		if (C > 0)
		{
			p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
			p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
			p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
			p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
		}

		if (Iw<=0||Ih<=0)
		{
				p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
				p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
				p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
				p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
		}
	}

	// The following codes are calculating the gradient of ciou.

	if (iou_loss == CIOU)
	{
		float ar_gt = truth.w / truth.h;
		float ar_pred = pred.w / pred.h;
		float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
		float alpha = ar_loss / (1 - I/U + ar_loss + 0.000001);
		float ar_dw=8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.h;
		float ar_dh=-8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.w;

		if (C > 0)
		{
			// dar*
			p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
			p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
			p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
			p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
		}

		if (Iw<=0||Ih<=0)
		{
			p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
			p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
			p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
			p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
		}
	}

	dxrep ddx = {0};
	ddx.dt = p_dx;      //We follow the original code released from GDarknet. So in yolo_layer.c, dt, db, dl, dr are already dx, dy, dw, dh.
	ddx.db = p_dy;
	ddx.dl = p_dw;
	ddx.dr = p_dh;

	return ddx;
}


float box_rmse(const Darknet::Box & a, const Darknet::Box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");
	// this function is used in multiple places

	return sqrt(pow(a.x-b.x, 2) +
				pow(a.y-b.y, 2) +
				pow(a.w-b.w, 2) +
				pow(a.h-b.h, 2));
}


void test_dunion()
{
	TAT(TATPARMS);

	Darknet::Box a = {0, 0, 1, 1};
	Darknet::Box dxa= {0+.0001, 0, 1, 1};
	Darknet::Box dya= {0, 0+.0001, 1, 1};
	Darknet::Box dwa= {0, 0, 1+.0001, 1};
	Darknet::Box dha= {0, 0, 1, 1+.0001};

	Darknet::Box b = {.5, .5, .2, .2};
	dbox di = dunion(a,b);

	*cfg_and_state.output
		<< "Union: "
		<< di.dx << " "
		<< di.dy << " "
		<< di.dw << " "
		<< di.dh << std::endl;

	float inter =  box_union(a, b);
	float xinter = box_union(dxa, b);
	float yinter = box_union(dya, b);
	float winter = box_union(dwa, b);
	float hinter = box_union(dha, b);
	xinter = (xinter - inter)/(0.0001f);
	yinter = (yinter - inter)/(0.0001f);
	winter = (winter - inter)/(0.0001f);
	hinter = (hinter - inter)/(0.0001f);

	*cfg_and_state.output
		<< "Union Manual "
		<< xinter << " "
		<< yinter << " "
		<< winter << " "
		<< hinter << std::endl;
}


void test_dintersect()
{
	TAT(TATPARMS);

	Darknet::Box a = {0, 0, 1, 1};
	Darknet::Box dxa= {0+.0001, 0, 1, 1};
	Darknet::Box dya= {0, 0+.0001, 1, 1};
	Darknet::Box dwa= {0, 0, 1+.0001, 1};
	Darknet::Box dha= {0, 0, 1, 1+.0001};

	Darknet::Box b = {.5, .5, .2, .2};
	dbox di = dintersect(a,b);

	*cfg_and_state.output
		<< "Inter: "
		<< di.dx << " "
		<< di.dy << " "
		<< di.dw << " "
		<< di.dh << std::endl;

	float inter =  box_intersection(a, b);
	float xinter = box_intersection(dxa, b);
	float yinter = box_intersection(dya, b);
	float winter = box_intersection(dwa, b);
	float hinter = box_intersection(dha, b);
	xinter = (xinter - inter)/(0.0001f);
	yinter = (yinter - inter)/(0.0001f);
	winter = (winter - inter)/(0.0001f);
	hinter = (hinter - inter)/(0.0001f);

	*cfg_and_state.output
		<< "Inter Manual "
		<< xinter << " "
		<< yinter << " "
		<< winter << " "
		<< hinter << std::endl;
}


void test_box()
{
	TAT(TATPARMS);

	test_dintersect();
	test_dunion();
	Darknet::Box a = {0, 0, 1, 1};
	Darknet::Box dxa= {0+.00001, 0, 1, 1};
	Darknet::Box dya= {0, 0+.00001, 1, 1};
	Darknet::Box dwa= {0, 0, 1+.00001, 1};
	Darknet::Box dha= {0, 0, 1, 1+.00001};

	Darknet::Box b = {.5, 0, .2, .2};

	float iou = box_iou(a,b);
	iou = (1-iou)*(1-iou);

	*cfg_and_state.output << iou << std::endl;

	dbox d = diou(a, b);

	*cfg_and_state.output
		<< "Test: "
		<< d.dx << " "
		<< d.dy << " "
		<< d.dw << " "
		<< d.dh << std::endl;

	float xiou = box_iou(dxa, b);
	float yiou = box_iou(dya, b);
	float wiou = box_iou(dwa, b);
	float hiou = box_iou(dha, b);
	xiou = ((1-xiou)*(1-xiou) - iou)/(0.00001f);
	yiou = ((1-yiou)*(1-yiou) - iou)/(0.00001f);
	wiou = ((1-wiou)*(1-wiou) - iou)/(0.00001f);
	hiou = ((1-hiou)*(1-hiou) - iou)/(0.00001f);

	*cfg_and_state.output
		<< "Manual "
		<< xiou << " "
		<< yiou << " "
		<< wiou << " "
		<< hiou << std::endl;
}


dbox diou(const Darknet::Box & a, const Darknet::Box & b)
{
	TAT(TATPARMS); // not marking it as reviewed since the code below has a serious bug!
	// this function is called from multiple locations

	const float u = box_union(a, b);
	const float i = box_intersection(a, b);

	/** @todo the following IF statement always evaluated to @p true due to the @p "|| 1" comparison.
	 * That line was changed by AlexeyAB on 2019-11-23.  Was that an accident?  Should the code below
	 * never run?  Or was this a debug modification that was accidentally left in the code?
	 */
	if (i <= 0 || 1)
	{
		dbox dd;
		dd.dx = b.x - a.x;
		dd.dy = b.y - a.y;
		dd.dw = b.w - a.w;
		dd.dh = b.h - a.h;

		return dd;
	}

	/// @todo due to the error (?) in the IF statmeent above, we'll never get to the code below this line!

	const dbox di = dintersect(a, b);
	const dbox du = dunion(a, b);

	dbox dd;
	dd.dx = (di.dx*u - du.dx*i) / (u*u);
	dd.dy = (di.dy*u - du.dy*i) / (u*u);
	dd.dw = (di.dw*u - du.dw*i) / (u*u);
	dd.dh = (di.dh*u - du.dh*i) / (u*u);

	return dd;
}


void do_nms_sort_v2(Darknet::Box *boxes, float **probs, int total, int classes, float thresh)
{
	TAT(TATPARMS);
	// 2024-04-25:  I think this one is no longer called.

	int i, j, k;
	sortable_bbox* s = (sortable_bbox*)xcalloc(total, sizeof(sortable_bbox));

	for(i = 0; i < total; ++i){
		s[i].index = i;
		s[i].class_id = 0;
		s[i].probs = probs;
	}

	for(k = 0; k < classes; ++k)
	{
		for(i = 0; i < total; ++i)
		{
			s[i].class_id = k;
		}

		/// @todo replace qsort() low priority
		qsort(s, total, sizeof(sortable_bbox), nms_comparator);

		for(i = 0; i < total; ++i)
		{
			if(probs[s[i].index][k] == 0)
			{
				continue;
			}
			Darknet::Box a = boxes[s[i].index];
			for(j = i+1; j < total; ++j)
			{
				Darknet::Box b = boxes[s[j].index];
				if (box_iou(a, b) > thresh)
				{
					probs[s[j].index][k] = 0;
				}
			}
		}
	}

	free(s);
}


void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);
	// this is a "C" function call
	// 2024-04-25:  this one seems to be called often


	int k = total - 1;
	for (int i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			std::swap(dets[i], dets[k]);
			--k;
			--i;
		}
	}
	total = k + 1;

	for (int i = 0; i < total; ++i)
	{
		dets[i].sort_class = -1;
	}

	sort_box_detections(dets, total);

	for (int i = 0; i < total; ++i)
	{
		if (dets[i].objectness == 0)
		{
			continue;
		}
		Darknet::Box a = dets[i].bbox;
		for (int j = i + 1; j < total; ++j)
		{
			if (dets[j].objectness == 0)
			{
				continue;
			}
			Darknet::Box b = dets[j].bbox;
			if (box_iou(a, b) > thresh)
			{
				dets[j].objectness = 0;
				for (int class_idx = 0; class_idx < classes; ++class_idx)
				{
					dets[j].prob[class_idx] = 0;
				}
			}
		}
	}
}


void do_nms_sort(detection * dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);
	// this is a "C" function call
	// this is called from everywhere

	int k = total - 1;
	for (int i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			std::swap(dets[i], dets[k]);
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k)
	{
		for (int i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}

		sort_box_detections(dets, total);

		for (int i = 0; i < total; ++i)
		{
			if (dets[i].prob[k] == 0.0f)
			{
				continue;
			}
			Darknet::Box a = dets[i].bbox;
			for (int j = i + 1; j < total; ++j)
			{
				Darknet::Box b = dets[j].bbox;
				if (box_iou(a, b) > thresh)
				{
					dets[j].prob[k] = 0.0f;
				}
			}
		}
	}

	return;
}


/// @implement OBB: NMS for oriented bounding boxes using rotated IoU
void do_nms_sort_bdp(detection_obb * dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);

	// Remove detections with zero objectness
	int k = total - 1;
	for (int i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			std::swap(dets[i], dets[k]);
			--k;
			--i;
		}
	}
	total = k + 1;

	// For each class, suppress overlapping detections using rotated IoU
	for (k = 0; k < classes; ++k)
	{
		for (int i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}

		// Sort by probability for this class
		sort_box_detections_bdp(dets, total);

		for (int i = 0; i < total; ++i)
		{
			if (dets[i].prob[k] == 0.0f)
			{
				continue;
			}
			DarknetBoxBDP a = dets[i].bbox;
			for (int j = i + 1; j < total; ++j)
			{
				DarknetBoxBDP b = dets[j].bbox;
				// Use rotated IoU instead of standard IoU
				if (box_riou(a, b) > thresh)
				{
					dets[j].prob[k] = 0.0f;
				}
			}
		}
	}

	return;
}


void do_nms(Darknet::Box *boxes, float **probs, int total, int classes, float thresh)
{
	TAT(TATPARMS);
	// this is called from many locations

	for (int i = 0; i < total; ++i)
	{
		int any = 0;
		for (int k = 0; k < classes; ++k)
		{
			any = any || (probs[i][k] > 0.0f);
		}

		if (!any)
		{
			continue;
		}

		for (int j = i + 1; j < total; ++j)
		{
			if (box_iou(boxes[i], boxes[j]) > thresh)
			{
				for (int k = 0; k < classes; ++k)
				{
					if (probs[i][k] < probs[j][k])
					{
						probs[i][k] = 0.0f;
					}
					else
					{
						probs[j][k] = 0.0f;
					}
				}
			}
		}
	}
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1)
{
	TAT(TATPARMS);
	// this is a "C" call
	// this is called from several locations

	int k = total - 1;
	for (int i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			std::swap(dets[i], dets[k]);
			--k;
			--i;
		}
	}
	total = k + 1;

//	*cfg_and_state.output << "diounms total is " << total << std::endl;

	for (k = 0; k < classes; ++k)
	{
		for (int i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}

		sort_box_detections(dets, total);

		for (int i = 0; i < total; ++i)
		{
			if (dets[i].prob[k] == 0)
			{
				continue;
			}
			Darknet::Box a = dets[i].bbox;
			for (int j = i + 1; j < total; ++j)
			{
				Darknet::Box b = dets[j].bbox;
				if (box_iou(a, b) > thresh && nms_kind == CORNERS_NMS)
				{
					dets[j].prob[k] = 0;
				}
				else if (box_diou(a, b) > thresh && nms_kind == GREEDY_NMS)
				{
					dets[j].prob[k] = 0;
				}
				else
				{
					if (box_diounms(a, b, beta1) > thresh && nms_kind == DIOU_NMS)
					{
						dets[j].prob[k] = 0;
					}
				}
			}
		}
	}
}


Darknet::Box encode_box(const Darknet::Box & b, const Darknet::Box & anchor)
{
	TAT_REVIEWED(TATPARMS, "2024-05-12");
	// not called, but exposed in the API

	Darknet::Box encode;
	encode.x = (b.x - anchor.x) / anchor.w;
	encode.y = (b.y - anchor.y) / anchor.h;
	encode.w = log2(b.w / anchor.w);
	encode.h = log2(b.h / anchor.h);

	return encode;
}


Darknet::Box decode_box(const Darknet::Box & b, const Darknet::Box & anchor)
{
	TAT_REVIEWED(TATPARMS, "2024-05-12");
	// not called, but exposed in the API

	Darknet::Box decode;
	decode.x = b.x * anchor.w + anchor.x;
	decode.y = b.y * anchor.h + anchor.y;
	decode.w = pow(2.0, b.w) * anchor.w;
	decode.h = pow(2.0, b.h) * anchor.h;

	return decode;
}

// ============================================================================
// BDP IoU IMPLEMENTATION (BOX WITH DIRECTIONAL POINT)
// Computing the IoU of two boxes rotated * cos of the angle between their front points
// Reference: https://arxiv.org/abs/2208.05433
// ============================================================================
float box_iou_bdp(const DarknetBoxBDP & a, const DarknetBoxBDP & b)
{
	TAT(TATPARMS);
	// Compute rotated IoU and apply angular correction using full RectParams
	// computeFrontPointCosine needs all 6 parameters (x,y,w,h,fx,fy) to compute orientation vectors
	return box_riou(a, b) * computeFrontPointCosine({a.x, a.y, a.w, a.h, a.fx, a.fy}, {b.x, b.y, b.w, b.h, b.fx, b.fy});
}

// ============================================================================
// ROTATED IoU IMPLEMENTATION (TRUE POLYGON INTERSECTION)
// Following ROTATED_IOU_PLAN.md Phase 1
// ============================================================================

/// Helper structure for rotated box corners (4 vertices)
/// Counter-clockwise ordering: 0=front-right, 1=front-left, 2=rear-left, 3=rear-right
struct RotatedCorners {
	float x[4];  // x-coordinates of 4 corners
	float y[4];  // y-coordinates of 4 corners
};

// Compile-time validation
static_assert(sizeof(RotatedCorners) == 8 * sizeof(float), "RotatedCorners must be 8 floats");
static_assert(std::is_trivially_copyable<RotatedCorners>::value, "RotatedCorners must be trivially copyable");


/** Convert BDP box to 4 corner points in counter-clockwise order
 *
 * WHY: To compute true rotated IoU, we need the actual polygon vertices of the rotated rectangle.
 *      Current box_iou_bdp() ignores fx,fy and treats boxes as axis-aligned, which is wrong.
 *
 * HOW: Use fx,fy to compute rotation angle θ = atan2(fy-y, fx-x).
 *      Then apply 2D rotation to get 4 corners from center + half-dimensions.
 *
 * Corner ordering (viewed from above):
 *   1 --F-- 0  (0: front-right, 1: front-left, 2: rear-left, 3: rear-right)
 *   |      |
 *   |  C   |  C = center (x,y), F = front point (fx,fy)
 *   |      |
 *   2 ---- 3
 *
 * Angle θ = atan2(fy - y, fx - x) defines front direction.
 * Width w extends perpendicular to front direction.
 * Height h extends along front direction (from rear to front).
 *
 * @param box BDP box with 6 parameters (x,y,w,h,fx,fy)
 * @return 4 corners in normalized [0,1] coordinates
 */
static RotatedCorners bdp_to_corners(const DarknetBoxBDP & box)
{
	TAT(TATPARMS);

	// Preconditions: Validate input box parameters
	assert(box.w > 0.0f && "Width must be positive");
	assert(box.h > 0.0f && "Height must be positive");
	assert(std::isfinite(box.x) && std::isfinite(box.y) && "Center must be finite");
	assert(std::isfinite(box.fx) && std::isfinite(box.fy) && "Front point must be finite");

	// Vector from center to front point gives the orientation direction
	float front_dx = box.fx - box.x;
	float front_dy = box.fy - box.y;

	// Normalize the front direction vector
	float front_length = std::sqrt(front_dx * front_dx + front_dy * front_dy);
	if (front_length < 1e-8f) {
		// Degenerate case: front point equals center, default to pointing right
		front_dx = 1.0f;
		front_dy = 0.0f;
		front_length = 1.0f;
	}
	float front_dir_x = front_dx / front_length;
	float front_dir_y = front_dy / front_length;

	// Calculate actual front edge midpoint at distance h/2 from center
	float half_height = box.h * 0.5f;
	float front_midpoint_x = box.x + front_dir_x * half_height;
	float front_midpoint_y = box.y + front_dir_y * half_height;

	// Calculate back edge midpoint (opposite side)
	float back_midpoint_x = box.x - front_dir_x * half_height;
	float back_midpoint_y = box.y - front_dir_y * half_height;

	// Calculate perpendicular vector for width (90° CCW rotation of front direction)
	float perp_dx = -front_dir_y;
	float perp_dy = front_dir_x;
	// Already normalized since front_dir is normalized

	// Calculate half-width offset
	float hw_offset_x = perp_dx * box.w * 0.5f;
	float hw_offset_y = perp_dy * box.w * 0.5f;

	RotatedCorners corners;

	// Compute 4 corners in counter-clockwise order
	// Front edge corners (based on computed front midpoint)
	corners.x[0] = front_midpoint_x + hw_offset_x;  // Front-right
	corners.y[0] = front_midpoint_y + hw_offset_y;

	corners.x[1] = back_midpoint_x + hw_offset_x;  // Back-right (CCW order: go around the box)
	corners.y[1] = back_midpoint_y + hw_offset_y;

	// Back edge corners (based on computed back midpoint)
	corners.x[2] = back_midpoint_x - hw_offset_x;  // Back-left
	corners.y[2] = back_midpoint_y - hw_offset_y;

	corners.x[3] = front_midpoint_x - hw_offset_x;  // Front-left (complete the CCW loop)
	corners.y[3] = front_midpoint_y - hw_offset_y;

	// Postconditions: Verify all corners are finite
	assert(std::isfinite(corners.x[0]) && std::isfinite(corners.y[0]) && "Corner 0 must be finite");
	assert(std::isfinite(corners.x[1]) && std::isfinite(corners.y[1]) && "Corner 1 must be finite");
	assert(std::isfinite(corners.x[2]) && std::isfinite(corners.y[2]) && "Corner 2 must be finite");
	assert(std::isfinite(corners.x[3]) && std::isfinite(corners.y[3]) && "Corner 3 must be finite");

	return corners;
}


/** Compute intersection area of two convex quadrilaterals using Sutherland-Hodgman algorithm
 *
 * WHY: To find true IoU of rotated rectangles, we need their intersection area.
 *      Standard axis-aligned IoU uses min/max on edges, but rotated boxes require polygon clipping.
 *
 * HOW: Sutherland-Hodgman algorithm clips subject polygon against each edge of clip polygon.
 *      For each edge, vertices are added/removed based on which side of the edge they're on.
 *      After clipping against all 4 edges, we have the intersection polygon.
 *      Area is computed using shoelace formula.
 *
 * Algorithm steps:can 
 * 1. Start with subject polygon (4 corners of box A)
 * 2. For each of 4 edges of clip polygon (box B):
 *    a. For each edge of current polygon:
 *       - If both vertices inside: add current vertex
 *       - If entering (outside→inside): add intersection point
 *       - If exiting (inside→outside): add intersection point
 *       - If both outside: add nothing
 * 3. Final polygon = intersection region
 * 4. Compute area using shoelace formula: A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
 *
 * References:
 * - Sutherland & Hodgman, "Reentrant Polygon Clipping", CACM 1974
 * - ROTATED_IOU_PLAN.md Phase 1.3
 *
 * @param subject First quadrilateral corners (to be clipped)
 * @param clip Second quadrilateral corners (clipping window)
 * @return Intersection area in [0, min(area_subject, area_clip)]
 */
static float sutherland_hodgman_intersection(const RotatedCorners & subject, const RotatedCorners & clip)
{
	TAT(TATPARMS);

	// Preconditions: All inputs must be finite
	assert(std::isfinite(subject.x[0]) && std::isfinite(clip.x[0]));

	// Working buffer for polygon clipping
	// Maximum possible vertices after clipping a quadrilateral by quadrilateral = 12
	// (Each of 4 edges can add up to 2 intersection points)
	float poly_x[12], poly_y[12];
	int poly_size = 4;

	// Initialize working polygon with subject corners
	for (int i = 0; i < 4; i++) {
		poly_x[i] = subject.x[i];
		poly_y[i] = subject.y[i];
	}

	// Debug disabled
	bool debug_this_call = false;

	// Clip against each of the 4 edges of clip polygon
	for (int edge_idx = 0; edge_idx < 4; edge_idx++) {
		if (poly_size == 0) break;  // No intersection left, early exit

		// Define clipping edge: from clip[edge_idx] to clip[(edge_idx+1)%4]
		int edge_next = (edge_idx + 1) % 4;
		float edge_x1 = clip.x[edge_idx];
		float edge_y1 = clip.y[edge_idx];
		float edge_x2 = clip.x[edge_next];
		float edge_y2 = clip.y[edge_next];

		// Edge vector and outward-pointing normal
		// For counter-clockwise polygon, outward normal is 90° clockwise rotation of edge
		// 90° CW rotation: (dx, dy) -> (dy, -dx)
		float edge_dx = edge_x2 - edge_x1;
		float edge_dy = edge_y2 - edge_y1;
		float normal_x = edge_dy;    // Perpendicular (90° CW rotation)
		float normal_y = -edge_dx;

		if (debug_this_call) {
			fprintf(stderr, "DEBUG: Edge %d: (%.4f,%.4f) -> (%.4f,%.4f), normal=(%.4f,%.4f)\n",
				edge_idx, edge_x1, edge_y1, edge_x2, edge_y2, normal_x, normal_y);
		}

		// Output buffer for this clipping stage
		float output_x[12], output_y[12];
		int output_size = 0;

		// Clip current polygon against this edge
		for (int i = 0; i < poly_size; i++) {
			int next_i = (i + 1) % poly_size;

			float curr_x = poly_x[i];
			float curr_y = poly_y[i];
			float next_x = poly_x[next_i];
			float next_y = poly_y[next_i];

			// Test if vertices are inside half-plane defined by edge
			// Inside if dot product with outward normal is <= 0
			// (outward normal points away from polygon interior, obtained by 90° CW rotation for CCW polygon)
			// Points inside the polygon have negative or zero distance to the outward normal
			float curr_dist = (curr_x - edge_x1) * normal_x + (curr_y - edge_y1) * normal_y;
			float next_dist = (next_x - edge_x1) * normal_x + (next_y - edge_y1) * normal_y;

			bool curr_inside = (curr_dist <= 0.0f);
			bool next_inside = (next_dist <= 0.0f);

			if (curr_inside) {
				// Current vertex is inside → add it to output
				output_x[output_size] = curr_x;
				output_y[output_size] = curr_y;
				output_size++;
			}

			if (curr_inside != next_inside) {
				// Edge crosses clipping line → compute intersection point
				// Parametric line: P(t) = curr + t*(next - curr)
				// Find t where P(t) lies on clipping edge (dist = 0)
				float t = curr_dist / (curr_dist - next_dist);  // Linear interpolation
				float intersect_x = curr_x + t * (next_x - curr_x);
				float intersect_y = curr_y + t * (next_y - curr_y);

				output_x[output_size] = intersect_x;
				output_y[output_size] = intersect_y;
				output_size++;
			}
		}

		// Copy output to polygon buffer for next iteration
		poly_size = output_size;
		for (int i = 0; i < poly_size; i++) {
			poly_x[i] = output_x[i];
			poly_y[i] = output_y[i];
		}

		if (debug_this_call) {
			fprintf(stderr, "DEBUG: After clipping edge %d: poly_size=%d\n", edge_idx, poly_size);
			if (poly_size == 0) {
				fprintf(stderr, "DEBUG: Polygon completely clipped away!\n");
			}
		}
	}

	// No intersection if resulting polygon has < 3 vertices (degenerate)
	if (poly_size < 3) {
		// Debug: print why we got no intersection
		if (debug_this_call) {
			fprintf(stderr, "DEBUG: Sutherland-Hodgman produced %d vertices (no intersection)\n", poly_size);
		}
		return 0.0f;
	}

	if (debug_this_call) {
		fprintf(stderr, "DEBUG: Final intersection polygon has %d vertices:\n", poly_size);
		for (int i = 0; i < poly_size; i++) {
			fprintf(stderr, "  [%d]: (%.4f, %.4f)\n", i, poly_x[i], poly_y[i]);
		}
	}

	// Compute area using shoelace formula (Gauss's area formula for simple polygons)
	// A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
	float area = 0.0f;
	for (int i = 0; i < poly_size; i++) {
		int next_i = (i + 1) % poly_size;
		area += poly_x[i] * poly_y[next_i] - poly_x[next_i] * poly_y[i];
	}
	area = std::abs(area) / 2.0f;

	if (debug_this_call) {
		fprintf(stderr, "DEBUG: Computed intersection area = %.6f\n", area);
	}

	// Postconditions
	assert(area >= 0.0f && "Area cannot be negative");
	assert(std::isfinite(area) && "Area must be finite");

	return area;
}


/** Compute true rotated IoU for BDP boxes using polygon intersection
 *
 * WHY: Current box_iou_bdp() ignores fx,fy and treats boxes as axis-aligned, then applies
 *      angular correction cos(angle/2). This is an approximation that breaks for highly rotated boxes.
 *      True rotated IoU computes actual polygon intersection.
 *
 * HOW: 1. Convert both BDP boxes to 4 corner points (using fx,fy for rotation)
 *      2. Compute polygon intersection area using Sutherland-Hodgman clipping
 *      3. Compute union area = area_a + area_b - intersection_area
 *      4. Return IoU = intersection / union
 *
 * DIFFERENCES FROM box_iou_bdp():
 * - box_iou_bdp: Uses axis-aligned IoU (ignores fx,fy), then multiplies by cos(angle/2)
 * - box_riou: Uses true rotated rectangle intersection (accounts for fx,fy directly)
 *
 * Algorithm complexity: O(1) with small constant (~100 floating-point ops)
 * Performance target: < 50μs per call (from ROTATED_IOU_PLAN.md)
 *
 * @param a Predicted BDP box
 * @param b Ground truth BDP box
 * @return IoU ∈ [0, 1]
 *
 * @see box_iou_bdp() - axis-aligned approximation (faster but inaccurate for rotation)
 * @see bdp_to_corners() - corner calculation from BDP parameters
 * @see sutherland_hodgman_intersection() - polygon clipping algorithm
 */
float box_riou(const DarknetBoxBDP & a, const DarknetBoxBDP & b)
{
	TAT(TATPARMS);

	// Preconditions
	assert(a.w > 0.0f && a.h > 0.0f && "Box 'a' must have positive dimensions");
	assert(b.w > 0.0f && b.h > 0.0f && "Box 'b' must have positive dimensions");

	// Convert both boxes to corner representations
	RotatedCorners corners_a = bdp_to_corners(a);
	RotatedCorners corners_b = bdp_to_corners(b);

	// Compute intersection area using polygon clipping
	float intersection_area = sutherland_hodgman_intersection(corners_a, corners_b);

	// Box areas (simple: width × height, rotation doesn't change area)
	float area_a = a.w * a.h;
	float area_b = b.w * b.h;

	// Union area: area_a + area_b - intersection
	float union_area = area_a + area_b - intersection_area;


	// Avoid division by zero (shouldn't happen with positive dimensions, but be safe)
	if (union_area <= 1e-8f) return 0.0f;

	// Compute IoU
	float iou = intersection_area / union_area;

	// Postconditions: IoU must be in [0, 1]
	assert(iou >= -1e-5f && iou <= 1.0f + 1e-5f && "IoU must be in [0,1]");
	assert(std::isfinite(iou) && "IoU must be finite");

	// Clamp to [0, 1] to handle floating-point errors
	return std::max(0.0f, std::min(1.0f, iou));
}


/** OLD IMPLEMENTATION - COMMENTED OUT - Using GWD-based version at line 1870 instead
 *
 * Compute gradients of rotated IoU with respect to all 6 BDP parameters
 *
 * WHY: Training requires gradients ∂(RIoU)/∂(x,y,w,h,fx,fy) to update box parameters.
 *      Since box_riou() uses polygon clipping (no closed-form derivative), we use numerical gradients.
 *
 * HOW: Finite differences - perturb each parameter by small ε, compute ΔRIoU/ε.
 *      Forward differences: f'(x) ≈ [f(x+ε) - f(x)] / ε (optimized for speed).
 *      Fixed epsilon ε=1e-4 for all parameters (normalized coordinates).
 *
 * OPTIMIZATION (2025-10-05):
 *   - Compute base_riou = box_riou(a, b) once and reuse for all 6 gradients
 *   - Use forward differences instead of central differences (1 eval vs 2 per param)
 *   - Fixed epsilon = 1e-4 for all parameters (simpler, consistent scaling)
 *   - Reduces RIoU calls from 12 → 7 (41% fewer evaluations)
 *
 * ALGORITHM:
 *   1. base_riou = box_riou(a, b)  // Computed once
 *   For each parameter p ∈ {x, y, w, h, fx, fy}:
 *     2. Create perturbed box: a_plus = a with p → p+εₚ (adaptive ε)
 *     3. Compute: iou_plus = box_riou(a_plus, b)
 *     4. Gradient: ∂RIoU/∂p = (iou_plus - base_riou) / εₚ
 *
 * @param a Predicted BDP box (gradients computed w.r.t. this box)
 * @param b Ground truth BDP box (fixed)
 * @param iou_loss Loss type (currently only IoU supported, GIoU/DIoU/CIoU in Phase 4)
 * @return dxrep_bdp with gradients {dx, dy, dw, dh, dfx, dfy}
 *
 * PERFORMANCE: 7 RIoU evaluations (~350μs total, down from 600μs)
 * ACCURACY: Forward differences give O(ε) error (acceptable with gradient clipping)
 *
 * @see box_riou() - forward pass IoU calculation
 * @see delta_yolo_box_bdp() - usage site in yolo_layer.cpp
 */

/* COMMENTED OUT: Old finite difference implementation - replaced with GWD below
dxrep_bdp dx_box_riou(const DarknetBoxBDP & a, const DarknetBoxBDP & b, const IOU_LOSS iou_loss)
{
	TAT(TATPARMS);

	// Preconditions: Validate input box parameters have positive dimensions
	assert(a.w > 0.0f && a.h > 0.0f && "Box 'a' must have positive dimensions");
	assert(b.w > 0.0f && b.h > 0.0f && "Box 'b' must have positive dimensions");

	// Compute base RIoU once (reused for all gradient calculations)
	// OPTIMIZATION: Reduces RIoU calls from 12 to 7 (1 base + 6 forward differences)
	float base_riou = box_riou(a, b);

	// Early exit: if RIOU is very small (< 1e-6), return zero gradients for all parameters
	// This prevents spurious gradients when boxes don't overlap
	if (base_riou < 1e-6f) {
		return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	}

	// Fixed epsilon for all parameters (normalized coordinates in [0,1])
	// Using same epsilon for all parameters ensures consistent gradient scale
	constexpr float epsilon = 1e-4f;  // For all parameters (x, y, w, h, fx, fy)

	dxrep_bdp gradients = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	// Compute gradient w.r.t. x (center x-coordinate)
	// Use forward difference: grad ≈ [f(x+ε) - f(x)] / ε (saves 1 RIoU call per parameter)
	{
		DarknetBoxBDP a_plus = a;
		a_plus.x = a.x + epsilon;
		float iou_plus = box_riou(a_plus, b);

		// Safety: if box_riou returns NaN or inf, gradient is zero
		if (std::isfinite(iou_plus)) {
			gradients.dx = (iou_plus - base_riou) / epsilon;
		}
	}

	// Compute gradient w.r.t. y (center y-coordinate)
	{
		DarknetBoxBDP a_plus = a;
		a_plus.y = a.y + epsilon;
		float iou_plus = box_riou(a_plus, b);

		if (std::isfinite(iou_plus)) {
			gradients.dy = (iou_plus - base_riou) / epsilon;
		}
	}

	// Compute gradient w.r.t. w (width)
	{
		DarknetBoxBDP a_plus = a;
		a_plus.w = a.w + epsilon;
		float iou_plus = box_riou(a_plus, b);

		if (std::isfinite(iou_plus)) {
			gradients.dw = (iou_plus - base_riou) / epsilon;
		}
	}

	// Compute gradient w.r.t. h (height)
	{
		DarknetBoxBDP a_plus = a;
		a_plus.h = a.h + epsilon;
		float iou_plus = box_riou(a_plus, b);

		if (std::isfinite(iou_plus)) {
			gradients.dh = (iou_plus - base_riou) / epsilon;
		}
	}

	// Compute gradient w.r.t. fx (front point x-coordinate)
	{
		DarknetBoxBDP a_plus = a;
		a_plus.fx = a.fx + epsilon;
		float iou_plus = box_riou(a_plus, b);

		if (std::isfinite(iou_plus)) {
			gradients.dfx = (iou_plus - base_riou) / epsilon;
		}
	}

	// Compute gradient w.r.t. fy (front point y-coordinate)
	{
		DarknetBoxBDP a_plus = a;
		a_plus.fy = a.fy + epsilon;
		float iou_plus = box_riou(a_plus, b);

		if (std::isfinite(iou_plus)) {
			gradients.dfy = (iou_plus - base_riou) / epsilon;
		}
	}

	// Postconditions: Gradients must be finite (guaranteed by safety checks above)
	assert(std::isfinite(gradients.dx) && std::isfinite(gradients.dy) && "x,y gradients must be finite");
	assert(std::isfinite(gradients.dw) && std::isfinite(gradients.dh) && "w,h gradients must be finite");
	assert(std::isfinite(gradients.dfx) && std::isfinite(gradients.dfy) && "fx,fy gradients must be finite");

	// TODO Phase 4: Implement GIoU, DIoU, CIoU variants
	// For now, basic IoU gradients are used for all loss types (IoU/GIoU/DIoU/CIoU)
	// This works because the forward pass computes the correct IoU variant,
	// and the gradients approximate the correct direction for optimization

	return gradients;
}
*/


/* COMMENTED OUT - INCORRECT: detection struct doesn't have bbox_bdp member
 * Use the version with detection_obb* instead (line 1005)
 *
void do_nms_sort_bdp(detection * dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);

	int k = total - 1;
	for (int i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			std::swap(dets[i], dets[k]);
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k)
	{
		for (int i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}

		sort_box_detections(dets, total);

		for (int i = 0; i < total; ++i)
		{
			if (dets[i].prob[k] == 0.0f)
			{
				continue;
			}
			const Darknet::BoxBDP & a = dets[i].bbox_bdp;

			if (!a.isValid())
			{
				dets[i].prob[k] = 0.0f;
				continue;
			}

			for (int j = i + 1; j < total; ++j)
			{
				const Darknet::BoxBDP & b = dets[j].bbox_bdp;

				if (!b.isValid())
				{
					dets[j].prob[k] = 0.0f;
					continue;
				}

				if (box_iou_bdp(a, b) > thresh)
				{
					dets[j].prob[k] = 0.0f;
				}
			}
		}
	}

	return;
}
*/


/**
 * Compute gradients for Rotated IoU using Gaussian Wasserstein Distance
 *
 * This function computes the gradient of the Rotated IoU (RIoU) loss with respect
 * to the predicted box parameters using the Gaussian Wasserstein Distance (GWD)
 * approximation. This provides smooth, differentiable gradients even for
 * non-overlapping or arbitrarily rotated boxes.
 *
 * Box representation (BoxBDP):
 *   - x, y: center coordinates
 *   - w, h: width and height (normalized)
 *   - fx, fy: front point coordinates (defines rotation)
 *
 * The GWD models each rotated rectangle as a 2D Gaussian distribution:
 *   N(μ, Σ) where μ = center, Σ = covariance matrix from rotation
 *
 * Reference: "Rethinking Rotated Object Detection with GWD Loss" (Yang et al., 2021)
 */

// Note: BoxBDP, dxrep_bdp, and IOU_LOSS are already defined in darknet_internal.hpp

/**
 * Helper: Compute matrix square root for 2x2 matrix using analytical formula
 * For a 2x2 positive definite matrix, we can compute sqrt directly
 */
inline void matrix_sqrt_2x2(float a11, float a12, float a21, float a22,
                            float& s11, float& s12, float& s21, float& s22)
{
    // For 2x2 symmetric positive definite matrix A:
    // sqrt(A) can be computed analytically
    
    float trace = a11 + a22;
    float det = a11 * a22 - a12 * a21;
    
    if (det <= 0.0f) {
        // Degenerate case - return identity-like matrix
        s11 = 1.0f; s12 = 0.0f;
        s21 = 0.0f; s22 = 1.0f;
        return;
    }
    
    float s = sqrtf(det);
    float t = sqrtf(trace + 2.0f * s);
    
    if (t < 1e-8f) {
        s11 = 1.0f; s12 = 0.0f;
        s21 = 0.0f; s22 = 1.0f;
        return;
    }
    
    s11 = (a11 + s) / t;
    s12 = a12 / t;
    s21 = a21 / t;
    s22 = (a22 + s) / t;
}

/**
 * Convert rotated box to Gaussian representation (mean and covariance)
 */
inline void box_to_gaussian(const DarknetBoxBDP& box,
                           float& mu_x, float& mu_y,
                           float& sigma_11, float& sigma_12,
                           float& sigma_21, float& sigma_22)
{
    // Mean is the center
    mu_x = box.x;
    mu_y = box.y;
    
    // Compute rotation from center to front point
    float dx = box.fx - box.x;
    float dy = box.fy - box.y;
    float front_dist = sqrtf(dx * dx + dy * dy);
    
    // Handle degenerate case (front point == center)
    if (front_dist < 1e-6f) {
        dx = 0.0f;
        dy = -1.0f;  // Default to pointing up
        front_dist = 1.0f;
    }
    
    // Normalized direction vectors
    float up_x = dx / front_dist;
    float up_y = dy / front_dist;
    float right_x = up_y;    // 90° clockwise rotation
    float right_y = -up_x;
    
    // Build rotation matrix R from box frame to world frame
    // R = [right_x  up_x]
    //     [right_y  up_y]
    
    // Covariance in box frame (axis-aligned)
    // Σ_local = diag(w²/12, h²/12)  -- variance of uniform distribution
    float var_w = (box.w * box.w) / 12.0f;
    float var_h = (box.h * box.h) / 12.0f;
    
    // Transform to world frame: Σ = R * Σ_local * R^T
    sigma_11 = var_w * right_x * right_x + var_h * up_x * up_x;
    sigma_12 = var_w * right_x * right_y + var_h * up_x * up_y;
    sigma_21 = sigma_12;  // Symmetric
    sigma_22 = var_w * right_y * right_y + var_h * up_y * up_y;
}

/**
 * Main gradient computation function for Rotated IoU using GWD
 */
dxrep_bdp dx_box_riou(const DarknetBoxBDP& pred,
                      const DarknetBoxBDP& truth,
                      const IOU_LOSS riou_loss)
{
    // Hyperparameters for GWD
    const float tau = 2.0f;      // Scale parameter for GWD -> IoU conversion
    const float eps = 1e-7f;     // Numerical stability
    const float grad_eps = 1e-4f; // Finite difference step size
    
    // Convert both boxes to Gaussian representation
    float pred_mu_x, pred_mu_y;
    float pred_s11, pred_s12, pred_s21, pred_s22;
    box_to_gaussian(pred, pred_mu_x, pred_mu_y, 
                   pred_s11, pred_s12, pred_s21, pred_s22);
    
    float truth_mu_x, truth_mu_y;
    float truth_s11, truth_s12, truth_s21, truth_s22;
    box_to_gaussian(truth, truth_mu_x, truth_mu_y,
                   truth_s11, truth_s12, truth_s21, truth_s22);
    
    // ========================================================================
    // Compute GWD: W²(N₁, N₂) = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁·Σ₂))
    // ========================================================================
    
    // Mean distance squared
    float mu_diff_x = pred_mu_x - truth_mu_x;
    float mu_diff_y = pred_mu_y - truth_mu_y;
    float mean_dist_sq = mu_diff_x * mu_diff_x + mu_diff_y * mu_diff_y;
    
    // Compute Σ₁ · Σ₂ (matrix product)
    float prod_11 = pred_s11 * truth_s11 + pred_s12 * truth_s21;
    float prod_12 = pred_s11 * truth_s12 + pred_s12 * truth_s22;
    float prod_21 = pred_s21 * truth_s11 + pred_s22 * truth_s21;
    float prod_22 = pred_s21 * truth_s12 + pred_s22 * truth_s22;
    
    // Compute √(Σ₁ · Σ₂)
    float sqrt_prod_11, sqrt_prod_12, sqrt_prod_21, sqrt_prod_22;
    matrix_sqrt_2x2(prod_11, prod_12, prod_21, prod_22,
                   sqrt_prod_11, sqrt_prod_12, sqrt_prod_21, sqrt_prod_22);
    
    // Compute trace term: Tr(Σ₁ + Σ₂ - 2√(Σ₁·Σ₂))
    float trace_term = (pred_s11 + truth_s11 - 2.0f * sqrt_prod_11) +
                       (pred_s22 + truth_s22 - 2.0f * sqrt_prod_22);
    
    // Ensure trace_term is non-negative (numerical stability)
    trace_term = fmaxf(0.0f, trace_term);
    
    // Total GWD²
    float gwd_sq = mean_dist_sq + trace_term;
    float gwd = sqrtf(fmaxf(eps, gwd_sq));
    
    // ========================================================================
    // Analytical gradients of GWD with respect to prediction parameters
    // ========================================================================
    
    // Gradient of mean distance term is straightforward
    float d_mean_dist_dx = 2.0f * mu_diff_x;
    float d_mean_dist_dy = 2.0f * mu_diff_y;
    
    // For the trace term and covariance derivatives, we use finite differences
    // This is more practical than deriving the full analytical gradient of matrix sqrt
    // (which involves eigendecomposition derivatives - extremely complex)
    
    dxrep_bdp grad;
    
    // Helper lambda for finite difference approximation
    auto compute_gwd_perturbed = [&](float dx, float dy, float dw, float dh,
                                     float dfx, float dfy) -> float {
        DarknetBoxBDP perturbed = pred;
        perturbed.x += dx;
        perturbed.y += dy;
        perturbed.w += dw;
        perturbed.h += dh;
        perturbed.fx += dfx;
        perturbed.fy += dfy;
        
        float p_mu_x, p_mu_y, p_s11, p_s12, p_s21, p_s22;
        box_to_gaussian(perturbed, p_mu_x, p_mu_y, p_s11, p_s12, p_s21, p_s22);
        
        float p_mu_diff_x = p_mu_x - truth_mu_x;
        float p_mu_diff_y = p_mu_y - truth_mu_y;
        float p_mean_dist_sq = p_mu_diff_x * p_mu_diff_x + p_mu_diff_y * p_mu_diff_y;
        
        float p_prod_11 = p_s11 * truth_s11 + p_s12 * truth_s21;
        float p_prod_12 = p_s11 * truth_s12 + p_s12 * truth_s22;
        float p_prod_21 = p_s21 * truth_s11 + p_s22 * truth_s21;
        float p_prod_22 = p_s21 * truth_s12 + p_s22 * truth_s22;
        
        float p_sqrt_11, p_sqrt_12, p_sqrt_21, p_sqrt_22;
        matrix_sqrt_2x2(p_prod_11, p_prod_12, p_prod_21, p_prod_22,
                       p_sqrt_11, p_sqrt_12, p_sqrt_21, p_sqrt_22);
        
        float p_trace = (p_s11 + truth_s11 - 2.0f * p_sqrt_11) +
                        (p_s22 + truth_s22 - 2.0f * p_sqrt_22);
        p_trace = fmaxf(0.0f, p_trace);
        
        float p_gwd_sq = p_mean_dist_sq + p_trace;
        return sqrtf(fmaxf(eps, p_gwd_sq));
    };
    
    // Compute gradients using central differences
    // This is numerically stable and handles the complex covariance derivatives
    float gwd_xp = compute_gwd_perturbed(grad_eps, 0, 0, 0, 0, 0);
    float gwd_xm = compute_gwd_perturbed(-grad_eps, 0, 0, 0, 0, 0);
    float d_gwd_dx = (gwd_xp - gwd_xm) / (2.0f * grad_eps);
    
    float gwd_yp = compute_gwd_perturbed(0, grad_eps, 0, 0, 0, 0);
    float gwd_ym = compute_gwd_perturbed(0, -grad_eps, 0, 0, 0, 0);
    float d_gwd_dy = (gwd_yp - gwd_ym) / (2.0f * grad_eps);
    
    float gwd_wp = compute_gwd_perturbed(0, 0, grad_eps, 0, 0, 0);
    float gwd_wm = compute_gwd_perturbed(0, 0, -grad_eps, 0, 0, 0);
    float d_gwd_dw = (gwd_wp - gwd_wm) / (2.0f * grad_eps);
    
    float gwd_hp = compute_gwd_perturbed(0, 0, 0, grad_eps, 0, 0);
    float gwd_hm = compute_gwd_perturbed(0, 0, 0, -grad_eps, 0, 0);
    float d_gwd_dh = (gwd_hp - gwd_hm) / (2.0f * grad_eps);
    
    float gwd_fxp = compute_gwd_perturbed(0, 0, 0, 0, grad_eps, 0);
    float gwd_fxm = compute_gwd_perturbed(0, 0, 0, 0, -grad_eps, 0);
    float d_gwd_dfx = (gwd_fxp - gwd_fxm) / (2.0f * grad_eps);
    
    float gwd_fyp = compute_gwd_perturbed(0, 0, 0, 0, 0, grad_eps);
    float gwd_fym = compute_gwd_perturbed(0, 0, 0, 0, 0, -grad_eps);
    float d_gwd_dfy = (gwd_fyp - gwd_fym) / (2.0f * grad_eps);
    
    // ========================================================================
    // Convert GWD to IoU-like loss and compute final gradients
    // ========================================================================
    
    // Loss = GWD (minimize distance)
    // Or alternatively: Loss = 1 - exp(-τ * GWD) for IoU-like behavior
    // We use simple GWD as loss for stability
    
    float loss_gradient_scale = 1.0f;
    
    // Apply different loss formulations based on riou_loss type
    if (riou_loss == GIOU || riou_loss == DIOU || riou_loss == CIOU) {
        // For GIoU/DIoU/CIoU variants, we scale the GWD gradient
        // These variants add penalty terms for aspect ratio and center distance
        
        // Center distance penalty (similar to DIoU)
        float center_dist = sqrtf(mean_dist_sq + eps);
        float max_dist = sqrtf((pred.w + truth.w) * (pred.w + truth.w) + 
                               (pred.h + truth.h) * (pred.h + truth.h)) / 2.0f;
        
        if (max_dist > eps) {
            float diou_penalty = center_dist / (max_dist + eps);
            loss_gradient_scale *= (1.0f + diou_penalty);
        }
        
        // Aspect ratio penalty (for CIoU)
        if (riou_loss == CIOU) {
            float pred_aspect = pred.w / (pred.h + eps);
            float truth_aspect = truth.w / (truth.h + eps);
            float v = (4.0f / (M_PI * M_PI)) * 
                     (atanf(truth_aspect) - atanf(pred_aspect)) *
                     (atanf(truth_aspect) - atanf(pred_aspect));
            
            // Aspect ratio gradients
            float alpha = v / (1.0f - gwd / (gwd + 1.0f) + v + eps);
            float ar_dw = (8.0f / (M_PI * M_PI)) * 
                         (atanf(truth_aspect) - atanf(pred_aspect)) * 
                         pred.h / (pred.w * pred.w + pred.h * pred.h + eps);
            float ar_dh = -(8.0f / (M_PI * M_PI)) * 
                          (atanf(truth_aspect) - atanf(pred_aspect)) * 
                          pred.w / (pred.w * pred.w + pred.h * pred.h + eps);
            
            d_gwd_dw += alpha * ar_dw;
            d_gwd_dh += alpha * ar_dh;
        }
    }
    
    // Final gradients for IoU maximization
    // GWD is distance (minimize), but IoU is similarity (maximize)
    // NEGATE gradients: grad_IoU = -grad_GWD
    // For gradient ascent (maximize IoU): param_new = param_old + lr * grad
    grad.dx = -loss_gradient_scale * d_gwd_dx;
    grad.dy = -loss_gradient_scale * d_gwd_dy;
    grad.dw = -loss_gradient_scale * d_gwd_dw;
    grad.dh = -loss_gradient_scale * d_gwd_dh;
    grad.dfx = -loss_gradient_scale * d_gwd_dfx;
    grad.dfy = -loss_gradient_scale * d_gwd_dfy;

    // Clamp gradients to prevent exploding gradients
    const float max_grad = 10.0f;
    grad.dx = fmaxf(-max_grad, fminf(max_grad, grad.dx));
    grad.dy = fmaxf(-max_grad, fminf(max_grad, grad.dy));
    grad.dw = fmaxf(-max_grad, fminf(max_grad, grad.dw));
    grad.dh = fmaxf(-max_grad, fminf(max_grad, grad.dh));
    grad.dfx = fmaxf(-max_grad, fminf(max_grad, grad.dfx));
    grad.dfy = fmaxf(-max_grad, fminf(max_grad, grad.dfy));

    return grad;
}

// ============================================================================
// BDP TO PIXEL COORDINATE CONVERSION FOR RIOU LOSS CALCULATION
// ============================================================================

/** Forward transform: BDP normalized params → 8 pixel coordinates
 *
 * Converts BDP box (x,y,w,h,fx,fy) in normalized [0,1] coordinates to
 * 4 corner points in pixel space for RIOU loss calculation during training.
 *
 * INTERACTION WITH OTHER FUNCTIONS:
 * - Called from: delta_yolo_box_bdp() when computing RIOU loss
 * - Uses: BDP front point (fx,fy) to determine rotation angle
 * - Returns: 4 corners in pixel coordinates for rotated IoU computation
 */
RectCorners RotatedRectTransform::forward(
	const RectParams& params,
	int imageWidth,
	int imageHeight)
{
	TAT(TATPARMS);

	// Preconditions: Validate inputs to prevent invalid loss calculations
	assert(params.isValid() && "BDP parameters must be valid for RIOU loss");
	assert(imageWidth > 0 && imageHeight > 0 && "Image dimensions must be positive");

	static_assert(sizeof(RectCorners) == 8 * sizeof(float), "RectCorners must be 8 floats");

	// Direction vector from center to front point
	float dx = params.fx - params.x;
	float dy = params.fy - params.y;
	float len = std::sqrt(dx * dx + dy * dy);

	// Handle degenerate case: front point equals center
	if (len < 1e-6f) {
		dx = 0.0f; dy = -1.0f; len = 1.0f;  // Default to pointing up
	}

	// Normalize to get front direction unit vector
	float front_x = dx / len;
	float front_y = dy / len;

	// Perpendicular vector (90° counter-clockwise for width direction)
	// In standard 2D coords: 90° CCW rotation of (x,y) is (-y, x)
	float perp_x = -front_y;
	float perp_y = front_x;

	// Half extents for corner calculation
	float half_w = params.w * 0.5f;
	float half_h = params.h * 0.5f;

	// Calculate corners in normalized space, then convert to pixels
	// Height extends along front direction, width extends perpendicular
	RectCorners corners;

	// Front edge midpoint (center + front_dir * half_h)
	float front_mid_x = params.x + front_x * half_h;
	float front_mid_y = params.y + front_y * half_h;

	// Back edge midpoint (center - front_dir * half_h)
	float back_mid_x = params.x - front_x * half_h;
	float back_mid_y = params.y - front_y * half_h;

	// Corner 1: Front-left (front_mid - perp * half_w)
	float x1_norm = front_mid_x - perp_x * half_w;
	float y1_norm = front_mid_y - perp_y * half_w;
	corners.p1 = Point2D(x1_norm * imageWidth, y1_norm * imageHeight);

	// Corner 2: Front-right (front_mid + perp * half_w)
	float x2_norm = front_mid_x + perp_x * half_w;
	float y2_norm = front_mid_y + perp_y * half_w;
	corners.p2 = Point2D(x2_norm * imageWidth, y2_norm * imageHeight);

	// Corner 3: Back-right (back_mid + perp * half_w)
	float x3_norm = back_mid_x + perp_x * half_w;
	float y3_norm = back_mid_y + perp_y * half_w;
	corners.p3 = Point2D(x3_norm * imageWidth, y3_norm * imageHeight);

	// Corner 4: Back-left (back_mid - perp * half_w)
	float x4_norm = back_mid_x - perp_x * half_w;
	float y4_norm = back_mid_y - perp_y * half_w;
	corners.p4 = Point2D(x4_norm * imageWidth, y4_norm * imageHeight);

	// Postconditions: Verify all corners are finite for safe loss calculation
	assert(std::isfinite(corners.p1.x) && std::isfinite(corners.p1.y) && "Corner 1 must be finite");
	assert(std::isfinite(corners.p2.x) && std::isfinite(corners.p2.y) && "Corner 2 must be finite");
	assert(std::isfinite(corners.p3.x) && std::isfinite(corners.p3.y) && "Corner 3 must be finite");
	assert(std::isfinite(corners.p4.x) && std::isfinite(corners.p4.y) && "Corner 4 must be finite");

	return corners;
}


/** Inverse transform: 8 pixel coordinates → BDP normalized params
 *
 * Converts 4 corner points in pixel coords back to normalized BDP parameters.
 * This is the inverse of the forward transform.
 */
std::optional<RectParams> RotatedRectTransform::inverse(
	const RectCorners& corners,
	int imageWidth,
	int imageHeight)
{
	// Preconditions: Validate inputs for safe inverse calculation
	assert(imageWidth > 0 && imageHeight > 0 && "Image dimensions must be positive");
	assert(std::isfinite(corners.p1.x) && std::isfinite(corners.p1.y) && "Corner 1 must be finite");
	assert(std::isfinite(corners.p2.x) && std::isfinite(corners.p2.y) && "Corner 2 must be finite");
	assert(std::isfinite(corners.p3.x) && std::isfinite(corners.p3.y) && "Corner 3 must be finite");
	assert(std::isfinite(corners.p4.x) && std::isfinite(corners.p4.y) && "Corner 4 must be finite");

	// Convert pixel coordinates to normalized [0,1] space
	float p1_x = corners.p1.x / imageWidth;
	float p1_y = corners.p1.y / imageHeight;
	float p2_x = corners.p2.x / imageWidth;
	float p2_y = corners.p2.y / imageHeight;
	float p3_x = corners.p3.x / imageWidth;
	float p3_y = corners.p3.y / imageHeight;
	float p4_x = corners.p4.x / imageWidth;
	float p4_y = corners.p4.y / imageHeight;

	// Calculate center as average of all 4 corners
	float center_x = (p1_x + p2_x + p3_x + p4_x) * 0.25f;
	float center_y = (p1_y + p2_y + p3_y + p4_y) * 0.25f;

	// Calculate front edge midpoint (p1 and p2)
	float front_mid_x = (p1_x + p2_x) * 0.5f;
	float front_mid_y = (p1_y + p2_y) * 0.5f;

	// Calculate front direction vector (from center to front midpoint)
	float front_dx = front_mid_x - center_x;
	float front_dy = front_mid_y - center_y;
	float front_len = std::sqrt(front_dx * front_dx + front_dy * front_dy);

	if (front_len < 1e-6f) {
		// Degenerate case: cannot determine orientation
		return std::nullopt;
	}

	// Height is twice the distance from center to front edge
	float height = front_len * 2.0f;

	// Width is distance between p1 and p2 (front edge length)
	float width_dx = p2_x - p1_x;
	float width_dy = p2_y - p1_y;
	float width = std::sqrt(width_dx * width_dx + width_dy * width_dy);

	// Front point is simply the front edge midpoint
	// In forward transform: front_mid = center + front_dir * half_h
	// So in inverse: fx,fy = front_mid_x,y
	float fx = front_mid_x;
	float fy = front_mid_y;

	// Construct result
	RectParams params;
	params.x = center_x;
	params.y = center_y;
	params.w = width;
	params.h = height;
	params.fx = fx;
	params.fy = fy;

	// Postcondition: Verify recovered parameters are valid
	if (!params.isValid()) {
		return std::nullopt;
	}

	return params;
}


/** Compute cos(angle/2) between predicted and target orientation vectors
 *
 * WHY: BDP loss needs to penalize orientation misalignment. Direct angle difference
 *      can cause discontinuities at 0°/360°. Using cos(angle/2) provides smooth gradients
 *      and naturally maps to [0,1] range where 1=perfect alignment, 0=opposite directions.
 *
 * HOW: 1. Extract direction vectors from center to front point for both boxes
 *      2. Compute cos(θ) via normalized dot product
 *      3. Apply half-angle formula: cos(θ/2) = sqrt((1 + cos(θ)) / 2)
 *
 * INTERACTION WITH OTHER FUNCTIONS:
 * - Called from: box_iou_bdp() for angular correction in rotated IoU calculation
 * - Uses: RectParams.fx, RectParams.fy to determine orientation vectors
 * - Returns: Similarity metric for orientation alignment in [0,1]
 *
 * @param pred Predicted BDP box parameters (normalized [0,1])
 * @param target Ground truth BDP box parameters (normalized [0,1])
 * @return cos(angle/2) in [0,1] where 1=aligned, 0=opposite (180° apart)
 */
float computeFrontPointCosine(const RectParams& pred, const RectParams& target)
{
	TAT(TATPARMS);

	// Preconditions: Validate input parameters
	assert(pred.isValid() && "Predicted box parameters must be valid");
	assert(target.isValid() && "Target box parameters must be valid");
	assert(std::isfinite(pred.x) && std::isfinite(pred.y) && "Pred center must be finite");
	assert(std::isfinite(target.x) && std::isfinite(target.y) && "Target center must be finite");
	assert(pred.w > 0 && pred.h > 0 && "Pred width/height must be positive");
	assert(target.w > 0 && target.h > 0 && "Target width/height must be positive");
	assert(pred.fx >= 0.0f && pred.fx <= 1.0f && "Pred fx must be in [0,1]");
	assert(pred.fy >= 0.0f && pred.fy <= 1.0f && "Pred fy must be in [0,1]");
	assert(target.fx >= 0.0f && target.fx <= 1.0f && "Target fx must be in [0,1]");
	assert(target.fy >= 0.0f && target.fy <= 1.0f && "Target fy must be in [0,1]");

	// Compile-time validation
	static_assert(sizeof(RectParams) == 6 * sizeof(float), "RectParams must be 6 floats");

	// Vectors from center to front point define orientation direction
	float pred_vec_x = pred.fx - pred.x;
	float pred_vec_y = pred.fy - pred.y;
	float target_vec_x = target.fx - target.x;
	float target_vec_y = target.fy - target.y;

	// Dot product: measures alignment between vectors
	float dot = pred_vec_x * target_vec_x + pred_vec_y * target_vec_y;

	// Magnitudes for normalization
	float mag_pred = std::sqrt(pred_vec_x * pred_vec_x + pred_vec_y * pred_vec_y);
	float mag_target = std::sqrt(target_vec_x * target_vec_x + target_vec_y * target_vec_y);

	// Handle zero-length vectors (degenerate case: front point equals center)
	// Check the product to prevent underflow: if both are small (e.g., 1e-8 * 1e-8 = 1e-16)
	// the division would create huge values leading to NaN propagation
	float mag_product = mag_pred * mag_target;
	if (mag_product < 1e-8f) {
		return 1.0f;  // Default to perfect alignment for degenerate cases
	}

	// Half-angle formula: cos(θ/2) = sqrt((1 + cos(θ)) / 2)
	// This maps:
	//   θ=0°   → cos(θ)=1  → cos(θ/2)=1     (perfect alignment)
	//   θ=90°  → cos(θ)=0  → cos(θ/2)=√2/2  (perpendicular)
	//   θ=180° → cos(θ)=-1 → cos(θ/2)=0     (opposite directions)
	float cos_half_theta = std::sqrt((1.0f + cos_theta) / 2.0f);

	// Postconditions: Result must be in [0,1] and finite
	assert(cos_half_theta >= 0.0f && cos_half_theta <= 1.0f && "cos(θ/2) must be in [0,1]");
	assert(std::isfinite(cos_half_theta) && cos_half_theta < 1e6f && "Result must be finite");

	return cos_half_theta;
}

/** Compute gradients of cos(θ/2) with respect to BDP parameters
 *
 * WHY: Angular correction term cos(θ/2) in box_iou_bdp needs gradients
 *      for training. This provides analytical derivatives instead of
 *      numerical approximations for better accuracy and stability.
 *
 * HOW: Chain rule through:
 *      1. cos(θ/2) = sqrt((1 + cos(θ)) / 2)
 *      2. cos(θ) = dot / (mag_pred * mag_target)
 *      3. dot = (fx-x)*(fx_t-x_t) + (fy-y)*(fy_t-y_t)
 *      4. Differentiate w.r.t. each parameter (x,y,w,h,fx,fy) of pred box
 *
 * INTERACTS WITH:
 *   - Called by: Future implementation of analytical box_iou_bdp gradients
 *   - Uses: computeFrontPointCosine() for validation (debug builds)
 *
 * @param pred Predicted BDP box
 * @param target Ground truth BDP box
 * @return Gradients {dx, dy, dw, dh, dfx, dfy}
 */
dxrep_bdp dx_computeFrontPointCosine(const DarknetBoxBDP& pred, const DarknetBoxBDP& target) {
	// Preconditions: Valid boxes with finite coordinates
	assert(pred.isValid() && "Pred box parameters must be valid");
	assert(target.isValid() && "Target box parameters must be valid");
	assert(std::isfinite(pred.x) && std::isfinite(pred.y) && "Pred center must be finite");
	assert(std::isfinite(target.x) && std::isfinite(target.y) && "Target center must be finite");

	dxrep_bdp gradients = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	// Orientation vectors: from center to front point
	float pvx = pred.fx - pred.x;
	float pvy = pred.fy - pred.y;
	float tvx = target.fx - target.x;
	float tvy = target.fy - target.y;

	// Dot product and magnitudes
	float dot = pvx * tvx + pvy * tvy;
	float mag_pred_sq = pvx * pvx + pvy * pvy;
	float mag_target_sq = tvx * tvx + tvy * tvy;
	float mag_pred = std::sqrt(mag_pred_sq);
	float mag_target = std::sqrt(mag_target_sq);

	// Early exit for degenerate cases (zero-length vectors)
	// Check the product to prevent underflow: if both are small (e.g., 1e-8 * 1e-8 = 1e-16)
	// the division would create huge values leading to NaN propagation
	float mag_product = mag_pred * mag_target;
	if (mag_product < 1e-8f) {
		// cos(θ/2) = 1 for degenerate cases, so all gradients are zero
		return gradients;
	}

	// Normalized dot product: cos(θ)
	float cos_theta_unclamped = dot / mag_product;
	float cos_theta = std::clamp(cos_theta_unclamped, -1.0f, 1.0f);

	// Chain rule: ∂cos(θ/2)/∂cos(θ) = 1 / (4 * cos(θ/2))
	// From: cos(θ/2) = sqrt((1 + cos(θ)) / 2)
	float cos_half_theta = std::sqrt((1.0f + cos_theta) / 2.0f);

	// Near θ=180° (cos_theta ≈ -1), gradient becomes infinite, return zero
	// Also check if we're at clamping boundary where derivatives are undefined
	if (cos_half_theta < 1e-6f || cos_theta_unclamped < -0.999f) {
		return gradients;
	}
	float d_coshalf_d_costheta = 1.0f / (4.0f * cos_half_theta);

	// Product denominator: mag_pred * mag_target
	float denom = mag_pred * mag_target;

	// Gradients w.r.t. x: affects both pvx (fx-x) and affects dot product
	// ∂dot/∂x = -tvx (from ∂(fx-x)/∂x = -1)
	// ∂mag_pred/∂x = -pvx / mag_pred
	float d_dot_dx = -tvx;
	float d_magpred_dx = -pvx / mag_pred;
	float d_costheta_dx = (d_dot_dx * denom - dot * mag_target * d_magpred_dx) / (denom * denom);
	gradients.dx = d_coshalf_d_costheta * d_costheta_dx;

	// Gradients w.r.t. y: similar to x
	float d_dot_dy = -tvy;
	float d_magpred_dy = -pvy / mag_pred;
	float d_costheta_dy = (d_dot_dy * denom - dot * mag_target * d_magpred_dy) / (denom * denom);
	gradients.dy = d_coshalf_d_costheta * d_costheta_dy;

	// Gradients w.r.t. fx: affects pvx (fx-x) directly
	// ∂dot/∂fx = tvx (from ∂(fx-x)/∂fx = 1)
	// ∂mag_pred/∂fx = pvx / mag_pred
	float d_dot_dfx = tvx;
	float d_magpred_dfx = pvx / mag_pred;
	float d_costheta_dfx = (d_dot_dfx * denom - dot * mag_target * d_magpred_dfx) / (denom * denom);
	gradients.dfx = d_coshalf_d_costheta * d_costheta_dfx;

	// Gradients w.r.t. fy: similar to fx
	float d_dot_dfy = tvy;
	float d_magpred_dfy = pvy / mag_pred;
	float d_costheta_dfy = (d_dot_dfy * denom - dot * mag_target * d_magpred_dfy) / (denom * denom);
	gradients.dfy = d_coshalf_d_costheta * d_costheta_dfy;

	// Gradients w.r.t. w and h are ZERO
	// WHY: Width and height do not affect orientation vectors (fx-x, fy-y)
	//      Angular correction only depends on the direction from center to front point
	gradients.dw = 0.0f;
	gradients.dh = 0.0f;

	// Postconditions: All gradients must be finite
	assert(std::isfinite(gradients.dx) && "dx must be finite");
	assert(std::isfinite(gradients.dy) && "dy must be finite");
	assert(std::isfinite(gradients.dw) && "dw must be finite");
	assert(std::isfinite(gradients.dh) && "dh must be finite");
	assert(std::isfinite(gradients.dfx) && "dfx must be finite");
	assert(std::isfinite(gradients.dfy) && "dfy must be finite");

	return gradients;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}