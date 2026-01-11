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
		TAT_REVIEWED(TATPARMS, "2025-12-02");

		if (total <= 1)
		{
			return;
		}

		// precompute the scores to use during sorting
		for (int i = 0; i < total; i ++)
		{
			dets[i].sort_score = dets[i].prob[dets[i].sort_class];
		}

		// We want to sort from high probability to low probability.  The default sort behaviour would be to
		// sort from low to high.  We reverse the sort order by comparing RHS to LHS instead of LHS to RHS.

		std::sort(/** @todo try this again in 2026? std::execution::par_unseq,*/ dets, dets + total,
				[](const Darknet::Detection & lhs, const Darknet::Detection & rhs) -> bool
				{
					return rhs.sort_score < lhs.sort_score;
				});

		return;
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


void do_nms_obj(DarknetDetection *dets, int total, int classes, float thresh)
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


static void do_nms_sort_cpu(DarknetDetection * dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);

	for (int k = 0; k < classes; ++k)
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

			const Darknet::Box & a = dets[i].bbox;

			for (int j = i + 1; j < total; ++j)
			{
				const Darknet::Box & b = dets[j].bbox;

				if (box_iou(a, b) > thresh)
				{
					dets[j].prob[k] = 0.0f;
				}
			}
		}
	}
}

void do_nms_sort(DarknetDetection * dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);
	// this is a "C" function call
	// this is called from everywhere

	// move all items with zero objectness to the end of the array
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

	// reset the size "total" to exclude the zero objectness
	total = k + 1;

	do_nms_sort_cpu(dets, total, classes, thresh);

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
void diounms_sort(DarknetDetection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1)
{
	TAT_REVIEWED(TATPARMS, "2025-12-02");

	// this is a "C" call
	// this is called from several locations

	// Remove items where the objectness score is exactly zero.  This moves them to the bottom of the array.
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

	// re-calculate the number of non-zero items since all zero items are now moved to the end of the array
	total = k + 1;

	// for each class perform NMS
	for (k = 0; k < classes; ++k)
	{
		for (int i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}

		// sort items in descending order based on probability, meaning the "best" (highest probability) class is now first
		sort_box_detections(dets, total);

		// perform NMS
		for (int i = 0; i < total; ++i)
		{
			if (dets[i].prob[k] == 0)
			{
				// probability is zero so skip
				continue;
			}

			// for each detection, compare it with all the later ones (i + 1)
			// remember that "i" will always have higher probability than "j" since they were sorted above
			const Darknet::Box & a = dets[i].bbox;

			for (int j = i + 1; j < total; ++j)
			{
				const Darknet::Box & b = dets[j].bbox;
				switch (nms_kind)
				{
					case DEFAULT_NMS: // original YOLO NMS
					case CORNERS_NMS:
					{
						if (box_iou(a, b) > thresh)
						{
							// suppress "j" and keep "i"
							dets[j].prob[k] = 0;
						}
						break;
					}
					case GREEDY_NMS: // greedy NMS (distance-based IoU)
					{
						if (box_diou(a, b) > thresh)
						{
							// suppress "j" and keep "i"
							dets[j].prob[k] = 0;
						}
						break;
					}
					case DIOU_NMS: // full DIoU-NMS
					{
						if (box_diounms(a, b, beta1) > thresh)
						{
							// suppress "j" and keep "i"
							dets[j].prob[k] = 0;
						}
						break;
					}
				}
			}
		}
	}

	return;
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
