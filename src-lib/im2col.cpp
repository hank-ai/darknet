#include "im2col.hpp"
#include <stdio.h>
#include "Timing.hpp"

float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad)
{
	TAT(TATPARMS);

	row -= pad;
	col -= pad;

	if (row < 0 ||
		col < 0 ||
		row >= height ||
		col >= width)
	{
		return 0;
	}

	return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
	int channels,  int height,  int width,
	int ksize,  int stride, int pad, float* data_col)
{
	TAT(TATPARMS);

	int c,h,w;
	int height_col = (height + 2*pad - ksize) / stride + 1;
	int width_col = (width + 2*pad - ksize) / stride + 1;

	int channels_col = channels * ksize * ksize;
	for (c = 0; c < channels_col; ++c) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;
		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
						im_row, im_col, c_im, pad);
			}
		}
	}
}


// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline static bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
	TAT(TATPARMS);
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void im2col_cpu_ext(
	const float* data_im,						// input
	const int channels,							// input channels
	const int height, const int width,			// input size
	const int kernel_h, const int kernel_w,		// kernel size
	const int pad_h, const int pad_w,			// padding size
	const int stride_h, const int stride_w,		// stride
	const int dilation_h, const int dilation_w,	// dilation
	float* data_col)							// output
{
	TAT(TATPARMS);

	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;

#pragma omp parallel for schedule(dynamic, 1)
	for (int channel = 0; channel < channels; ++channel)
	{
		const float* data_im_ptr = data_im + channel * channel_size;

		for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row)
		{
			for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col)
			{
				int input_row = -pad_h + kernel_row * dilation_h;
				float* data_col_ptr = data_col +
					((channel * kernel_h + kernel_row) * kernel_w + kernel_col) *
					output_h * output_w;

				for (int output_rows = 0; output_rows < output_h; ++output_rows)
				{
					if (!is_a_ge_zero_and_a_lt_b(input_row, height))
					{
						// padding column
						memset(data_col_ptr, 0, output_w * sizeof(float));
						data_col_ptr += output_w;
					}
					else {
						// stride=1, dilation=1 optimization
						if (stride_w == 1 && dilation_w == 1)
						{
							int input_col = -pad_w + kernel_col * dilation_w;
							int output_col = 0;

							// left 
							int left_pad = 0;
							if (input_col < 0)
							{
								left_pad = -input_col;
								output_col = left_pad;
								input_col = 0;
							}

							// right 
							int copy_width = output_w - output_col;
							if (input_col + copy_width > width)
							{
								copy_width = width - input_col;
							}

							// left padding
							if (left_pad > 0)
							{
								memset(data_col_ptr, 0, left_pad * sizeof(float));
							}

							// center copy
							if (copy_width > 0)
							{
								memcpy(data_col_ptr + output_col,
									data_im_ptr + input_row * width + input_col,
									copy_width * sizeof(float));
							}

							// right padding
							int right_pad_start = output_col + copy_width;
							if (right_pad_start < output_w)
							{
								memset(data_col_ptr + right_pad_start, 0,
									(output_w - right_pad_start) * sizeof(float));
							}

							data_col_ptr += output_w;
						}
						else
						{
							// usual case（stride > 1 or dilation > 1）
							int input_col = -pad_w + kernel_col * dilation_w;

#pragma omp simd
							for (int output_col = 0; output_col < output_w; ++output_col)
							{
								int col_idx = input_col + output_col * stride_w;
								data_col_ptr[output_col] = is_a_ge_zero_and_a_lt_b(col_idx, width) ?
									data_im_ptr[input_row * width + col_idx] : 0;
							}
							data_col_ptr += output_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}
