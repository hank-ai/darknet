#pragma once

#include "darknet_internal.hpp"


ACTIVATION get_activation(char *s);

const char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void gradient_array_swish(const float *x, const int n, const float * sigmoid, float * delta);
void gradient_array_mish(const int n, const float * activation_input, float * delta);
void gradient_array_hard_mish(const int n, const float * activation_input, float * delta);
void activate_array(float *x, const int n, const ACTIVATION a);
void activate_array_swish(float *x, const int n, float * output_sigmoid, float * output);
void activate_array_mish(float *x, const int n, float * activation_input, float * output);
void activate_array_hard_mish(float *x, const int n, float * activation_input, float * output);
void activate_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *output);
void gradient_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *delta);
void activate_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *output, int use_max_val);
void gradient_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *delta);
#ifdef DARKNET_GPU
void activate_array_ongpu(float *x, int n, ACTIVATION a);
void activate_array_swish_ongpu(float *x, int n, float *output_sigmoid_gpu, float *output_gpu);
void activate_array_mish_ongpu(float *x, int n, float *activation_input_gpu, float *output_gpu);
void activate_array_hard_mish_ongpu(float *x, int n, float *activation_input_gpu, float *output_gpu);
void gradient_array_ongpu(float *x, int n, ACTIVATION a, float *delta);
void gradient_array_swish_ongpu(float *x, int n, float *sigmoid_gpu, float *delta);
void gradient_array_mish_ongpu(int n, float *activation_input_gpu, float *delta);
void gradient_array_hard_mish_ongpu(int n, float *activation_input_gpu, float *delta);
void activate_array_normalize_channels_ongpu(float *x, int n, int batch, int channels, int wh_step, float *output_gpu);
void gradient_array_normalize_channels_ongpu(float *output_gpu, int n, int batch, int channels, int wh_step, float *delta_gpu);
void activate_array_normalize_channels_softmax_ongpu(float *x, int n, int batch, int channels, int wh_step, float *output_gpu, int use_max_val);
void gradient_array_normalize_channels_softmax_ongpu(float *output_gpu, int n, int batch, int channels, int wh_step, float *delta_gpu);

#endif

static inline float stair_activate(float x)
{
	TAT(TATPARMS);

	const int n = std::floor(x);
	if (n % 2 == 0)
	{
		return std::floor(x / 2.0f);
	}

	return (x - n) + std::floor(x / 2.0f);
}

static inline float hardtan_activate(float x)
{
	TAT(TATPARMS);

	if (x < -1)
	{
		return -1;
	}
	if (x > 1)
	{
		return 1;
	}

	return x;
}

static inline float linear_activate(float x)
{
	TAT(TATPARMS);
	return x;
}

static inline float logistic_activate(float x)
{
	TAT(TATPARMS);
	return 1.0f / (1.0f + std::exp(-x));
}

static inline float loggy_activate(float x)
{
	TAT(TATPARMS);
	return 2.0f / (1.0f + std::exp(-x)) - 1.0f;
}

static inline float relu_activate(float x)
{
	TAT(TATPARMS);
	return x * (x > 0);
}

static inline float relu6_activate(float x)
{
	TAT(TATPARMS);
	std::clamp(x, 0.0f, 6.0f);
	return x;
}

static inline float elu_activate(float x)
{
	TAT(TATPARMS);
	return (x >= 0.0f) * x + (x < 0.0f) * (expf(x)-1.0f);
}

static inline float selu_activate(float x)
{
	TAT(TATPARMS);
	return (x >= 0.0f) * 1.0507f * x + (x < 0.0f) * 1.0507f * 1.6732f * (expf(x) - 1.0f);
}

static inline float relie_activate(float x)
{
	TAT(TATPARMS);
	return (x > 0) ? x : 0.01f * x;
}

static inline float ramp_activate(float x)
{
	TAT(TATPARMS);
	return x * (x > 0) + 0.1f * x;
}

static inline float leaky_activate(float x)
{
	TAT(TATPARMS);
	return (x > 0.0f) ? x : 0.1f * x;
}

//static inline float tanh_activate(float x){return (expf(2*x)-1)/(expf(2*x)+1);}
static inline float tanh_activate(float x)
{
	TAT(TATPARMS);
	return (2.0f / (1.0f + std::exp(-2.0f * x)) - 1.0f);
}

static inline float gelu_activate(float x)
{
	TAT(TATPARMS);
	return (0.5f * x * (1.0f + std::tanh(0.797885f * x + 0.035677f * std::pow(x, 3))));
}

static inline float softplus_activate(float x, float threshold)
{
	TAT(TATPARMS);
	if (x > threshold)
	{
		return x; // too large
	}
	else if (x < -threshold)
	{
		return expf(x); // too small
	}
	return std::log(expf(x) + 1.0f);
}

static inline float plse_activate(float x)
{
	TAT(TATPARMS);
	if (x < -4.0f)
	{
		return 0.01f * (x + 4.0f);
	}
	if (x > 4.0f)
	{
		return 0.01f * (x - 4.0f) + 1.0f;
	}
	return 0.125f * x + 0.5f;
}

static inline float lhtan_activate(float x)
{
	TAT(TATPARMS);
	if(x < 0.0f)
	{
		return 0.001f * x;
	}
	if(x > 1.0f)
	{
		return 0.001f * (x - 1.0f) + 1.0f;
	}
	return x;
}

static inline float lhtan_gradient(float x)
{
	TAT(TATPARMS);
	if(x > 0.0f && x < 1.0f)
	{
		return 1.0f;
	}
	return 0.001f;
}

static inline float hardtan_gradient(float x)
{
	TAT(TATPARMS);
	if (x > -1.0f && x < 1.0f)
	{
		return 1.0f;
	}
    return 0.0f;
}

static inline float linear_gradient(float x)
{
	TAT(TATPARMS);
	return 1.0f;
}

static inline float logistic_gradient(float x)
{
	TAT(TATPARMS);
	return (1-x)*x;
}

static inline float loggy_gradient(float x)
{
	TAT(TATPARMS);
	const float y = (x + 1.0f) / 2.0f;
    return 2.0f * (1.0f - y) * y;
}

static inline float stair_gradient(float x)
{
	TAT(TATPARMS);
	if (std::floor(x) == x)
	{
		return 0.0f;
	}
    return 1.0f;
}

static inline float relu_gradient(float x)
{
	TAT(TATPARMS);
	return (x > 0);
}

static inline float relu6_gradient(float x)
{
	TAT(TATPARMS);
	return (x > 0 && x < 6);
}

static inline float elu_gradient(float x)
{
	TAT(TATPARMS);
	return (x >= 0) + (x < 0) * (x + 1);
}

static inline float selu_gradient(float x)
{
	TAT(TATPARMS);
	return (x >= 0.0f) * 1.0507f + (x < 0.0f) * (x + 1.0507f * 1.6732f);
}

static inline float relie_gradient(float x)
{
	TAT(TATPARMS);
	return (x > 0) ? 1.0f : 0.01f;
}

static inline float ramp_gradient(float x)
{
	TAT(TATPARMS);
	return (x > 0) + 0.1f;
}

static inline float leaky_gradient(float x)
{
	TAT(TATPARMS);
	return (x > 0) ? 1.0f : 0.1f;
}

static inline float tanh_gradient(float x)
{
	TAT(TATPARMS);
	return 1.0f - x * x;
}

static inline float sech(float x)
{
	TAT(TATPARMS);
	return 2.0f / (std::exp(x) + std::exp(-x));
}

static inline float gelu_gradient(float x)
{
	TAT(TATPARMS);
	const float x3 = std::pow(x, 3);
    return 0.5 * std::tanh(0.0356774f * x3 + 0.797885f * x) + (0.0535161f * x3 + 0.398942f * x) * std::pow(sech(0.0356774f * x3 + 0.797885f * x), 2) + 0.5f;
}

static inline float plse_gradient(float x)
{
	TAT(TATPARMS);
	return (x < 0.0f || x > 1.0f) ? 0.01f : 0.125f;
}
