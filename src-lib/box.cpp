#ifdef __GNUC__
// 2023-06-25:  hide some of the warnings which for now we need to ignore in this file
#pragma GCC diagnostic ignored "-Wswitch"
#endif

#include "darknet_internal.hpp"

box float_to_box(const float * f)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	box b;
	b.x = f[0];
	b.y = f[1];
	b.w = f[2];
	b.h = f[3];

	return b;
}


box float_to_box_stride(const float *f, const int stride)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	box b;
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];

	return b;
}


dbox derivative(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	dbox d;
	d.dx = a.x < b.x ? 1.0 : -1.0;
	d.dy = a.y < b.y ? 1.0 : -1.0;
	d.dw = a.w < b.w ? 1.0 : -1.0;
	d.dh = a.h < b.h ? 1.0 : -1.0;

	return d;
}


/// where c is the smallest box that fully encompases a and b
boxabs box_c(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	boxabs ba;
	ba.top		= fmin(a.y - a.h / 2.0f, b.y - b.h / 2.0f);
	ba.bot		= fmax(a.y + a.h / 2.0f, b.y + b.h / 2.0f);
	ba.left		= fmin(a.x - a.w / 2.0f, b.x - b.w / 2.0f);
	ba.right	= fmax(a.x + a.w / 2.0f, b.x + b.w / 2.0f);

	return ba;
}


/// representation from x, y, w, h to top, left, bottom, right
boxabs to_tblr(const box & a)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	boxabs tblr;
	tblr.top	= a.y - (a.h / 2.0f);
	tblr.bot	= a.y + (a.h / 2.0f);
	tblr.left	= a.x - (a.w / 2.0f);
	tblr.right	= a.x + (a.w / 2.0f);

	return tblr;
}


float overlap(const float x1, const float w1, const float x2, const float w2)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	const float l1 = x1 - w1 / 2.0f;
	const float l2 = x2 - w2 / 2.0f;
	const float left = std::max(l1, l2);

	const float r1 = x1 + w1 / 2.0f;
	const float r2 = x2 + w2 / 2.0f;
	const float right = std::min(r1, r2);

	return right - left;
}

float box_intersection(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	const float w = overlap(a.x, a.w, b.x, b.w);
	const float h = overlap(a.y, a.h, b.y, b.h);
	if (w <= 0.0f || h <= 0.0f)
	{
		return 0.0f;
	}

	const float area = w * h;

	return area;
}

float box_union(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	const float i = box_intersection(a, b);
	const float u = a.w * a.h + b.w * b.h - i;

	return u;
}

float box_iou_kind(const box & a, const box & b, const IOU_LOSS iou_kind)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	//IOU, GIOU, MSE, DIOU, CIOU
	switch(iou_kind)
	{
		/// @todo what about MSE?
		case IOU:	return box_iou(a, b);
		case GIOU:	return box_giou(a, b);
		case DIOU:	return box_diou(a, b);
		case CIOU:	return box_ciou(a, b);
	}

	return box_iou(a, b);
}

float box_iou(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	const float I = box_intersection(a, b);
	if (I == 0.0f)
	{
		return 0;
	}

	// if intersection is non-zero, then union will of course be non-zero, so no need to worry about divide-by-zero
	const float U = box_union(a, b);

	return I / U;
}

float box_giou(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

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

float box_diou(const box & a, const box & b)
{
	/// https://github.com/Zzh-tju/DIoU-darknet
	/// https://arxiv.org/abs/1911.08287

	TAT_REVIEWED(TATPARMS, "2024-03-19");

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

float box_diounms(const box & a, const box & b, const float beta1)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

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

float box_ciou(const box & a, const box & b)
{
	// https://github.com/Zzh-tju/DIoU-darknet
	// https://arxiv.org/abs/1911.08287

	TAT(TATPARMS);

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

dxrep dx_box_iou(const box & pred, const box & truth, const IOU_LOSS iou_loss)
{
	TAT(TATPARMS);

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
	if (U > 0 ) {
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

	if (iou_loss == GIOU) {
	if (giou_C > 0) {
		// apply "C" term from gIOU
		p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
		p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
		p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
		p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
	}
	if (Iw<=0||Ih<=0) {
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
	if (iou_loss == DIOU) {
		if (C > 0) {
			p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
			p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
			p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
			p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
		}
	if (Iw<=0||Ih<=0){
			p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
			p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
			p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
			p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
		}
	}
	//The following codes are calculating the gradient of ciou.

	if (iou_loss == CIOU) {
	float ar_gt = truth.w / truth.h;
		float ar_pred = pred.w / pred.h;
		float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
	float alpha = ar_loss / (1 - I/U + ar_loss + 0.000001);
	float ar_dw=8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.h;
		float ar_dh=-8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.w;
		if (C > 0) {
		// dar*
			p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
			p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
			p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
			p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
		}
	if (Iw<=0||Ih<=0){
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

float box_rmse(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	return sqrt(pow(a.x-b.x, 2) +
				pow(a.y-b.y, 2) +
				pow(a.w-b.w, 2) +
				pow(a.h-b.h, 2));
}

dbox dintersect(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

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

dbox dunion(const box & a, const box & b)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	const dbox di = dintersect(a, b);

	dbox du;
	du.dw = a.h - di.dw;
	du.dh = a.w - di.dh;
	du.dx = -di.dx;
	du.dy = -di.dy;

	return du;
}


void test_dunion()
{
	TAT(TATPARMS);

	box a = {0, 0, 1, 1};
	box dxa= {0+.0001, 0, 1, 1};
	box dya= {0, 0+.0001, 1, 1};
	box dwa= {0, 0, 1+.0001, 1};
	box dha= {0, 0, 1, 1+.0001};

	box b = {.5, .5, .2, .2};
	dbox di = dunion(a,b);
	printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
	float inter =  box_union(a, b);
	float xinter = box_union(dxa, b);
	float yinter = box_union(dya, b);
	float winter = box_union(dwa, b);
	float hinter = box_union(dha, b);
	xinter = (xinter - inter)/(.0001);
	yinter = (yinter - inter)/(.0001);
	winter = (winter - inter)/(.0001);
	hinter = (hinter - inter)/(.0001);
	printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_dintersect()
{
	TAT(TATPARMS);

	box a = {0, 0, 1, 1};
	box dxa= {0+.0001, 0, 1, 1};
	box dya= {0, 0+.0001, 1, 1};
	box dwa= {0, 0, 1+.0001, 1};
	box dha= {0, 0, 1, 1+.0001};

	box b = {.5, .5, .2, .2};
	dbox di = dintersect(a,b);
	printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
	float inter =  box_intersection(a, b);
	float xinter = box_intersection(dxa, b);
	float yinter = box_intersection(dya, b);
	float winter = box_intersection(dwa, b);
	float hinter = box_intersection(dha, b);
	xinter = (xinter - inter)/(.0001);
	yinter = (yinter - inter)/(.0001);
	winter = (winter - inter)/(.0001);
	hinter = (hinter - inter)/(.0001);
	printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
	TAT(TATPARMS);

	test_dintersect();
	test_dunion();
	box a = {0, 0, 1, 1};
	box dxa= {0+.00001, 0, 1, 1};
	box dya= {0, 0+.00001, 1, 1};
	box dwa= {0, 0, 1+.00001, 1};
	box dha= {0, 0, 1, 1+.00001};

	box b = {.5, 0, .2, .2};

	float iou = box_iou(a,b);
	iou = (1-iou)*(1-iou);
	printf("%f\n", iou);
	dbox d = diou(a, b);
	printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

	float xiou = box_iou(dxa, b);
	float yiou = box_iou(dya, b);
	float wiou = box_iou(dwa, b);
	float hiou = box_iou(dha, b);
	xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
	yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
	wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
	hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
	printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(const box & a, const box & b)
{
	TAT(TATPARMS); // not marking it as reviewed since the code below has a serious bug!

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

struct sortable_bbox
{
	int index;
	int class_id;
	float **probs;
};

int nms_comparator(const void *pa, const void *pb)
{
	TAT(TATPARMS);

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

void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh)
{
	TAT(TATPARMS);

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
			box a = boxes[s[i].index];
			for(j = i+1; j < total; ++j)
			{
				box b = boxes[s[j].index];
				if (box_iou(a, b) > thresh)
				{
					probs[s[j].index][k] = 0;
				}
			}
		}
	}

	free(s);
}

int nms_comparator_v3(const void *pa, const void *pb)
{
	TAT(TATPARMS);

	detection a = *(detection *)pa;
	detection b = *(detection *)pb;
	float diff = 0.0f;
	if (b.sort_class >= 0)
	{
		diff = a.prob[b.sort_class] - b.prob[b.sort_class]; // there is already: prob = objectness*prob
	}
	else
	{
		diff = a.objectness - b.objectness;
	}

	if (diff < 0.0f)
	{
		return 1;
	}

	if (diff > 0.0f)
	{
		return -1;
	}

	return 0;
}

void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);

	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (i = 0; i < total; ++i)
	{
		dets[i].sort_class = -1;
	}

	/// @todo replace qsort() higher priority
	qsort(dets, total, sizeof(detection), nms_comparator_v3);

	for (i = 0; i < total; ++i)
	{
		if (dets[i].objectness == 0)
		{
			continue;
		}
		box a = dets[i].bbox;
		for (j = i + 1; j < total; ++j)
		{
			if (dets[j].objectness == 0)
			{
				continue;
			}
			box b = dets[j].bbox;
			if (box_iou(a, b) > thresh)
			{
				dets[j].objectness = 0;
				for (k = 0; k < classes; ++k)
				{
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
	TAT(TATPARMS);

	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k)
	{
		for (i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}

		/// @todo replace qsort() higher priority
		qsort(dets, total, sizeof(detection), nms_comparator_v3);

		for (i = 0; i < total; ++i)
		{
			//printf("  k = %d, \t i = %d \n", k, i);
			if (dets[i].prob[k] == 0)
			{
				continue;
			}
			box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j)
			{
				box b = dets[j].bbox;
				if (box_iou(a, b) > thresh)
				{
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
	TAT(TATPARMS);

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

	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k)
	{
		for (i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}

		/// @todo replace qsort() lower priority
		qsort(dets, total, sizeof(detection), nms_comparator_v3);

		for (i = 0; i < total; ++i)
		{
			if (dets[i].prob[k] == 0)
			{
				continue;
			}
			box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j)
			{
				box b = dets[j].bbox;
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

box encode_box(const box & b, const box & anchor)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	box encode;
	encode.x = (b.x - anchor.x) / anchor.w;
	encode.y = (b.y - anchor.y) / anchor.h;
	encode.w = log2(b.w / anchor.w);
	encode.h = log2(b.h / anchor.h);

	return encode;
}

box decode_box(const box & b, const box & anchor)
{
	TAT_REVIEWED(TATPARMS, "2024-03-19");

	box decode;
	decode.x = b.x * anchor.w + anchor.x;
	decode.y = b.y * anchor.h + anchor.y;
	decode.w = pow(2.0, b.w) * anchor.w;
	decode.h = pow(2.0, b.h) * anchor.h;

	return decode;
}
