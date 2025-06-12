#include "data.hpp"
#include "darknet_internal.hpp"

#define NUMCHARS 37


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	/** The permanent data-loading (image, bboxes) threads started by @ref Darknet::run_image_loading_control_thread().
	 *
	 * @since 2024-04-03
	 */
	static Darknet::VThreads data_loading_threads;


	/** Flag used by the image data loading threads to determine if they need to exit.
	 *
	 * @since 2024-04-02
	 */
	static std::atomic<bool> image_data_loading_threads_must_exit = false;


	/** Flags to indicate to individual data loading threads what they should do.  @p 0 is stop, and @p 1 is go.
	 * These flags are normally @p 0 and then are set to @p 1 by @ref run_image_loading_control_thread().
	 *
	 * (Was @p std::vector<bool> but that individual bit handling, and we only have a few threads.)
	 *
	 * @since 2024-04-10
	 */
	static std::vector<int> data_loading_per_thread_flag;


	/// @{ @todo: delete these once the code is cleaned up
	static const std::chrono::milliseconds thread_wait_ms(5); ///< @todo DELETE THIS! :(
	static load_args * args_swap = NULL; ///< @todo I wish I better understood how/why this exists...
	static std::mutex args_swap_mutex; // used to protect access to args_swap
	/// @}


	static inline data concat_datas(data *d, int n)
	{
		TAT(TATPARMS);

		data out = {0};
		for (int i = 0; i < n; ++i)
		{
			data newdata = concat_data(d[i], out);
			Darknet::free_data(out);
			out = newdata;
		}

		return out;
	}
}


list *get_paths(const char *filename)
{
	TAT(TATPARMS);

	char *path;
	FILE *file = fopen(filename, "r");
	if (!file)
	{
		file_error(filename, DARKNET_LOC);
	}

	list *lines = make_list();
	while((path=fgetl(file)))
	{
		list_insert(lines, path);
	}
	fclose(file);

	if (lines->size == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to read any lines from %s", filename);
	}

	return lines;
}

char **get_sequential_paths(char **paths, int n, int m, int mini_batch, int augment_speed, int contrastive)
{
	TAT(TATPARMS);

	int speed = rand_int(1, augment_speed);
	if (speed < 1)
	{
		speed = 1;
	}

	char** sequentia_paths = (char**)xcalloc(n, sizeof(char*));

	unsigned int *start_time_indexes = (unsigned int *)xcalloc(mini_batch, sizeof(unsigned int));
	for (int i = 0; i < mini_batch; ++i)
	{
		if (contrastive && (i % 2) == 1)
		{
			start_time_indexes[i] = start_time_indexes[i - 1];
		}
		else
		{
			start_time_indexes[i] = random_gen(0, m - 1);
		}
	}

	for (int i = 0; i < n; ++i)
	{
		int time_line_index = i % mini_batch;
		unsigned int index = start_time_indexes[time_line_index] % m;
		start_time_indexes[time_line_index] += speed;

		sequentia_paths[i] = paths[index];
	}
	free(start_time_indexes);

	return sequentia_paths;
}


char **get_random_paths_custom(char **paths, int n, int m, int contrastive)
{
	TAT(TATPARMS);

	char** random_paths = (char**)xcalloc(n, sizeof(char*));

	int old_index = 0;

	// "n" is the total number of filenames to be returned at once
	for (int i = 0; i < n; ++i)
	{
		int index = random_gen(0, m - 1);
		if (contrastive && (i % 2 == 1))
		{
			index = old_index;
		}
		else
		{
			old_index = index;
		}
		random_paths[i] = paths[index];
	}

	return random_paths;
}


char **get_random_paths(char **paths, int n, int m)
{
	TAT(TATPARMS);

	return get_random_paths_custom(paths, n, m, 0);
}


box_label *read_boxes(char *filename, int *n)
{
	TAT(TATPARMS);

	box_label* boxes = (box_label*)xcalloc(1, sizeof(box_label));
	FILE *file = fopen(filename, "r");
	if (!file)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to open annotation file \"%s\"", filename);
	}

	const int max_obj_img = 4000;// 30000;
	const int img_hash = (custom_hash(filename) % max_obj_img)*max_obj_img;
	float x, y, h, w;
	int id;
	int count = 0;
	while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
	{
//		*cfg_and_state.output << "x=" << x << " y=" << y << " w=" << w << " h=" << h << std::endl;

		boxes = (box_label*)xrealloc(boxes, (count + 1) * sizeof(box_label));
		boxes[count].track_id = count + img_hash;
		boxes[count].id = id;
		boxes[count].x = x;
		boxes[count].y = y;
		boxes[count].h = h;
		boxes[count].w = w;
		boxes[count].left   = x - w / 2.0f;
		boxes[count].right  = x + w / 2.0f;
		boxes[count].top    = y - h / 2.0f;
		boxes[count].bottom = y + h / 2.0f;
		++count;
	}

	fclose(file);
	*n = count;

	return boxes;
}


void randomize_boxes(box_label *b, int n)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < n; ++i)
	{
		const auto index = random_gen(0, n - 1);
		std::swap(b[i], b[index]);
	}
}


void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < n; ++i)
	{
		if(boxes[i].x == 0 && boxes[i].y == 0)
		{
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		if ((boxes[i].x + boxes[i].w / 2) < 0 || (boxes[i].y + boxes[i].h / 2) < 0 ||
			(boxes[i].x - boxes[i].w / 2) > 1 || (boxes[i].y - boxes[i].h / 2) > 1)
		{
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		boxes[i].left   = boxes[i].left  * sx - dx;
		boxes[i].right  = boxes[i].right * sx - dx;
		boxes[i].top    = boxes[i].top   * sy - dy;
		boxes[i].bottom = boxes[i].bottom* sy - dy;

		if(flip)
		{
			float swap = boxes[i].left;
			boxes[i].left = 1.0f - boxes[i].right;
			boxes[i].right = 1.0f - swap;
		}

		boxes[i].left =  constrain(0, 1, boxes[i].left);
		boxes[i].right = constrain(0, 1, boxes[i].right);
		boxes[i].top =   constrain(0, 1, boxes[i].top);
		boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

		boxes[i].x = (boxes[i].left+boxes[i].right)/2;
		boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
		boxes[i].w = (boxes[i].right - boxes[i].left);
		boxes[i].h = (boxes[i].bottom - boxes[i].top);

		boxes[i].w = constrain(0, 1, boxes[i].w);
		boxes[i].h = constrain(0, 1, boxes[i].h);
	}
}


int fill_truth_detection(const char *path, int num_boxes, int truth_size, float *truth, int classes, int flip, float dx, float dy, float sx, float sy, int net_w, int net_h)
{
	TAT(TATPARMS);

	// This method is used during the training process to load the boxes for the given image.

	char labelpath[4096];
	replace_image_to_label(path, labelpath);

	int count = 0;
	int i;
	box_label *boxes = read_boxes(labelpath, &count);
	int min_w_h = 0;
	float lowest_w = 1.F / net_w;
	float lowest_h = 1.F / net_h;
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	if (count > num_boxes)
	{
		count = num_boxes;
	}
	float x, y, w, h;
	int id;
	int sub = 0;

	for (i = 0; i < count; ++i)
	{
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;
		int track_id = boxes[i].track_id;

		// not detect small objects
		//if ((w < 0.001F || h < 0.001F)) continue;
		// if truth (box for object) is smaller than 1x1 pix
		//char buff[256];
		if (id >= classes)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d in %s", id, labelpath);
		}
		if ((w < lowest_w || h < lowest_h))
		{
			++sub;
			continue;
		}

		if (x == 999999 || y == 999999)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid annotation for class ID #%d in %s", id, labelpath);
		}
		/// @todo shouldn't this be x - w/2 < 0.0f?  And same for other variables?
		if (x <= 0.0f || x > 1.0f || y <= 0.0f || y > 1.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid coordinates for class ID #%d in %s", id, labelpath);
		}
		/// @todo again, instead of checking for > 1, shouldn't we check x + w / 2 ?
		if (w > 1.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid width for class ID #%d in %s", id, labelpath);
		}
		/// @todo check for y - h/2 and y + h/2?
		if (h > 1.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid height for class ID #%d in %s", id, labelpath);
		}

		if (x == 0) x += lowest_w;
		if (y == 0) y += lowest_h;

		truth[(i-sub)*truth_size +0] = x;
		truth[(i-sub)*truth_size +1] = y;
		truth[(i-sub)*truth_size +2] = w;
		truth[(i-sub)*truth_size +3] = h;
		truth[(i-sub)*truth_size +4] = id;
		truth[(i-sub)*truth_size +5] = track_id;

		if (min_w_h == 0) min_w_h = w*net_w;
		if (min_w_h > w*net_w) min_w_h = w*net_w;
		if (min_w_h > h*net_h) min_w_h = h*net_h;
	}
	free(boxes);
	return min_w_h;
}


void Darknet::free_data(data & d)
{
	TAT_REVIEWED(TATPARMS, "2024-04-04");

	// this is the only place in the entire codebase where the "shallow" flag is checked
	if (d.shallow == 0)
	{
		free_matrix(d.X);
		free_matrix(d.y);
	}
	else
	{
		free(d.X.vals);
		free(d.y.vals);
		d.X.vals = nullptr;
		d.y.vals = nullptr;
	}
	return;
}


void blend_truth(float *new_truth, int boxes, int truth_size, float *old_truth)
{
	TAT(TATPARMS);

	int count_new_truth = 0;
	int t;
	for (t = 0; t < boxes; ++t)
	{
		float x = new_truth[t*truth_size];
		if (!x)
		{
			break;
		}
		count_new_truth++;

	}
	for (t = count_new_truth; t < boxes; ++t)
	{
		float *new_truth_ptr = new_truth + t*truth_size;
		float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
		float x = old_truth_ptr[0];
		if (!x)
		{
			break;
		}

		new_truth_ptr[0] = old_truth_ptr[0];
		new_truth_ptr[1] = old_truth_ptr[1];
		new_truth_ptr[2] = old_truth_ptr[2];
		new_truth_ptr[3] = old_truth_ptr[3];
		new_truth_ptr[4] = old_truth_ptr[4];
	}
}


void blend_truth_mosaic(float *new_truth, int boxes, int truth_size, float *old_truth, int w, int h, float cut_x, float cut_y, int i_mixup, int left_shift, int right_shift, int top_shift, int bot_shift, int net_w, int net_h, int mosaic_bound)
{
	TAT(TATPARMS);

	const float lowest_w = 1.F / net_w;
	const float lowest_h = 1.F / net_h;

	int count_new_truth = 0;
	int t;
	for (t = 0; t < boxes; ++t)
	{
		float x = new_truth[t*truth_size];
		if (!x)
		{
			break;
		}
		count_new_truth++;

	}
	int new_t = count_new_truth;
	for (t = count_new_truth; t < boxes; ++t)
	{
		float *new_truth_ptr = new_truth + new_t*truth_size;
		new_truth_ptr[0] = 0;
		float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
		float x = old_truth_ptr[0];
		if (!x)
		{
			break;
		}

		float xb = old_truth_ptr[0];
		float yb = old_truth_ptr[1];
		float wb = old_truth_ptr[2];
		float hb = old_truth_ptr[3];



		// shift 4 images
		if (i_mixup == 0)
		{
			xb = xb - (float)(w - cut_x - right_shift) / w;
			yb = yb - (float)(h - cut_y - bot_shift) / h;
		}
		if (i_mixup == 1)
		{
			xb = xb + (float)(cut_x - left_shift) / w;
			yb = yb - (float)(h - cut_y - bot_shift) / h;
		}
		if (i_mixup == 2)
		{
			xb = xb - (float)(w - cut_x - right_shift) / w;
			yb = yb + (float)(cut_y - top_shift) / h;
		}
		if (i_mixup == 3)
		{
			xb = xb + (float)(cut_x - left_shift) / w;
			yb = yb + (float)(cut_y - top_shift) / h;
		}

		int left = (xb - wb / 2)*w;
		int right = (xb + wb / 2)*w;
		int top = (yb - hb / 2)*h;
		int bot = (yb + hb / 2)*h;

		if(mosaic_bound)
		{
			// fix out of Mosaic-bound
			float left_bound = 0, right_bound = 0, top_bound = 0, bot_bound = 0;
			if (i_mixup == 0)
			{
				left_bound = 0;
				right_bound = cut_x;
				top_bound = 0;
				bot_bound = cut_y;
			}
			if (i_mixup == 1)
			{
				left_bound = cut_x;
				right_bound = w;
				top_bound = 0;
				bot_bound = cut_y;
			}
			if (i_mixup == 2)
			{
				left_bound = 0;
				right_bound = cut_x;
				top_bound = cut_y;
				bot_bound = h;
			}
			if (i_mixup == 3)
			{
				left_bound = cut_x;
				right_bound = w;
				top_bound = cut_y;
				bot_bound = h;
			}


			if (left < left_bound)
			{
				left = left_bound;
			}
			if (right > right_bound)
			{
				right = right_bound;
			}
			if (top < top_bound) top = top_bound;
			if (bot > bot_bound) bot = bot_bound;


			xb = ((float)(right + left) / 2) / w;
			wb = ((float)(right - left)) / w;
			yb = ((float)(bot + top) / 2) / h;
			hb = ((float)(bot - top)) / h;
		}
		else
		{
			// fix out of bound
			if (left < 0)
			{
				float diff = (float)left / w;
				xb = xb - diff / 2;
				wb = wb + diff;
			}

			if (right > w)
			{
				float diff = (float)(right - w) / w;
				xb = xb - diff / 2;
				wb = wb - diff;
			}

			if (top < 0)
			{
				float diff = (float)top / h;
				yb = yb - diff / 2;
				hb = hb + diff;
			}

			if (bot > h)
			{
				float diff = (float)(bot - h) / h;
				yb = yb - diff / 2;
				hb = hb - diff;
			}

			left = (xb - wb / 2)*w;
			right = (xb + wb / 2)*w;
			top = (yb - hb / 2)*h;
			bot = (yb + hb / 2)*h;
		}


		// leave only within the image
		if(left >= 0 && right <= w && top >= 0 && bot <= h &&
			wb > 0 && wb < 1 && hb > 0 && hb < 1 &&
			xb > 0 && xb < 1 && yb > 0 && yb < 1 &&
			wb > lowest_w && hb > lowest_h)
		{
			new_truth_ptr[0] = xb;
			new_truth_ptr[1] = yb;
			new_truth_ptr[2] = wb;
			new_truth_ptr[3] = hb;
			new_truth_ptr[4] = old_truth_ptr[4];
			new_t++;
		}
	}
}


data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int use_gaussian_noise, int use_blur, int use_mixup,
	float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs)
{
	TAT(TATPARMS);

	// This is the method that gets called to load the "n" images for each loading thread while training a network.

	c = c ? c : 3;

	if (use_mixup == 2 || use_mixup == 4)
	{
		darknet_fatal_error(DARKNET_LOC, "cutmix=1 isn't supported for detector");
	}

	if (use_mixup == 3 && letter_box)
	{
		darknet_fatal_error(DARKNET_LOC, "letterbox and mosaic cannot be combined");
	}

	if (random_gen() % 2 == 0)
	{
		use_mixup = 0;
	}

	int *cut_x = nullptr;
	int *cut_y = nullptr;

	if (use_mixup == 3)
	{
		cut_x = (int*)calloc(n, sizeof(int));
		cut_y = (int*)calloc(n, sizeof(int));
		const float min_offset = 0.2; // 20%

		for (int i = 0; i < n; ++i)
		{
			cut_x[i] = rand_int(w*min_offset, w*(1 - min_offset));
			cut_y[i] = rand_int(h*min_offset, h*(1 - min_offset));
		}
	}

	data d = {0};
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
	d.X.cols = h*w*c;

	float r1 = 0.0f;
	float r2 = 0.0f;
	float r3 = 0.0f;
	float r4 = 0.0f;
	float resize_r1 = 0.0f;
	float resize_r2 = 0.0f;
	float dhue = 0.0f;
	float dsat = 0.0f;
	float dexp = 0.0f;
	float flip = 0.0f;
	float blur = 0.0f;
	int augmentation_calculated = 0;
	int gaussian_noise = 0;

	d.y = make_matrix(n, truth_size * boxes);

	for (int i_mixup = 0; i_mixup <= use_mixup; i_mixup++)
	{
		if (i_mixup)
		{
			augmentation_calculated = 0;   // recalculate augmentation for the 2nd sequence if(track==1)
		}

		char **random_paths;
		if (track)
		{
			random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
		}
		else
		{
			random_paths = get_random_paths_custom(paths, n, m, contrastive);
		}

		// about to load multiple images ("n"), usually batch size divided by the number of loading threads
		for (int i = 0; i < n; ++i)
		{
			float *truth = (float*)xcalloc(truth_size * boxes, sizeof(float));
			const char *filename = random_paths[i];

			cv::Mat src = load_rgb_mat_image(filename, c);

			const int oh = src.rows;	// original height
			const int ow = src.cols;	// original width

			int dw = (ow*jitter);
			int dh = (oh*jitter);

			float resize_down = resize;
			float resize_up = resize;

			if (resize_down > 1.0f)
			{
				resize_down = 1.0f / resize_down;
			}
			const int min_rdw = ow *(1.0f - (1.0f / resize_down)) / 2.0f;   // < 0
			const int min_rdh = oh *(1.0f - (1.0f / resize_down)) / 2.0f;   // < 0

			if (resize_up < 1.0f)
			{
				resize_up = 1.0f / resize_up;
			}
			const int max_rdw = ow * (1.0f - (1.0f / resize_up)) / 2.0f;     // > 0
			const int max_rdh = oh * (1.0f - (1.0f / resize_up)) / 2.0f;     // > 0

			if (!augmentation_calculated || !track)
			{
				augmentation_calculated = 1;
				resize_r1 = random_float();
				resize_r2 = random_float();

				if (!contrastive || contrastive_jit_flip || i % 2 == 0)
				{
					r1 = random_float();
					r2 = random_float();
					r3 = random_float();
					r4 = random_float();

					flip = use_flip ? random_gen() % 2 : 0;
				}

				if (!contrastive || contrastive_color || i % 2 == 0)
				{
					dhue = rand_uniform_strong(-hue, hue);
					dsat = rand_scale(saturation);
					dexp = rand_scale(exposure);
				}

				if (use_blur)
				{
					int tmp_blur = rand_int(0, 2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
					if (tmp_blur == 0)
					{
						blur = 0;
					}
					else if (tmp_blur == 1)
					{
						blur = 1;
					}
					else
					{
						blur = use_blur;
					}
				}

				if (use_gaussian_noise && rand_int(0, 1) == 1)
				{
					gaussian_noise = use_gaussian_noise;
				}
				else
				{
					gaussian_noise = 0;
				}
			}

			int pleft = rand_precalc_random(-dw, dw, r1);
			int pright = rand_precalc_random(-dw, dw, r2);
			int ptop = rand_precalc_random(-dh, dh, r3);
			int pbot = rand_precalc_random(-dh, dh, r4);

			if (resize < 1.0f)
			{
				// downsize only
				pleft += rand_precalc_random(min_rdw, 0, resize_r1);
				pright += rand_precalc_random(min_rdw, 0, resize_r2);
				ptop += rand_precalc_random(min_rdh, 0, resize_r1);
				pbot += rand_precalc_random(min_rdh, 0, resize_r2);
			}
			else
			{
				pleft += rand_precalc_random(min_rdw, max_rdw, resize_r1);
				pright += rand_precalc_random(min_rdw, max_rdw, resize_r2);
				ptop += rand_precalc_random(min_rdh, max_rdh, resize_r1);
				pbot += rand_precalc_random(min_rdh, max_rdh, resize_r2);
			}

			if (letter_box)
			{
				float img_ar = (float)ow / (float)oh;
				float net_ar = (float)w / (float)h;
				float result_ar = img_ar / net_ar;
				if (result_ar > 1)  // sheight - should be increased
				{
					float oh_tmp = ow / net_ar;
					float delta_h = (oh_tmp - oh)/2;
					ptop = ptop - delta_h;
					pbot = pbot - delta_h;
				}
				else  // swidth - should be increased
				{
					float ow_tmp = oh * net_ar;
					float delta_w = (ow_tmp - ow)/2;
					pleft = pleft - delta_w;
					pright = pright - delta_w;
				}
			}

			// move each 2nd image to the corner - so that most of it was visible
			if (use_mixup == 3 && random_gen() % 2 == 0)
			{
				if (flip)
				{
					if (i_mixup == 0) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 1) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 2) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
					if (i_mixup == 3) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
				}
				else
				{
					if (i_mixup == 0) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 1) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 2) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
					if (i_mixup == 3) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
				}
			}

			const int swidth = ow - pleft - pright;
			const int sheight = oh - ptop - pbot;

			const float sx = (float)swidth / ow;
			const float sy = (float)sheight / oh;

			const float dx = ((float)pleft / ow) / sx;
			const float dy = ((float)ptop / oh) / sy;

			// This is where we get the annotations for this image.
			const int min_w_h = fill_truth_detection(filename, boxes, truth_size, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);

			if ((min_w_h / 8) < blur && blur > 1)
			{
				blur = min_w_h / 8;   // disable blur if one of the objects is too small
			}

			Darknet::Image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur, boxes, truth_size, truth);

			if (use_mixup == 0)
			{
				d.X.vals[i] = ai.data;
				memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
			}
			else if (use_mixup == 1)
			{
				if (i_mixup == 0)
				{
					d.X.vals[i] = ai.data;
					memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
				}
				else if (i_mixup == 1)
				{
					Darknet::Image old_img = make_empty_image(w, h, c);
					old_img.data = d.X.vals[i];
					blend_images_cv(ai, 0.5, old_img, 0.5);
					blend_truth(d.y.vals[i], boxes, truth_size, truth);
					Darknet::free_image(old_img);
					d.X.vals[i] = ai.data;
				}
			}
			else if (use_mixup == 3)
			{
				if (i_mixup == 0)
				{
					Darknet::Image tmp_img = make_image(w, h, c);
					d.X.vals[i] = tmp_img.data;
				}

				if (flip)
				{
					int tmp = pleft;
					pleft = pright;
					pright = tmp;
				}

				const int left_shift = min_val_cmp(cut_x[i], max_val_cmp(0, (-pleft*w / ow)));
				const int top_shift = min_val_cmp(cut_y[i], max_val_cmp(0, (-ptop*h / oh)));

				const int right_shift = min_val_cmp((w - cut_x[i]), max_val_cmp(0, (-pright*w / ow)));
				const int bot_shift = min_val_cmp(h - cut_y[i], max_val_cmp(0, (-pbot*h / oh)));


				//int k, x, y;
				for (int k = 0; k < c; ++k)
				{
					for (int y = 0; y < h; ++y)
					{
						int j = y*w + k*w*h;
						if (i_mixup == 0 && y < cut_y[i])
						{
							int j_src = (w - cut_x[i] - right_shift) + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
							memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
						}
						if (i_mixup == 1 && y < cut_y[i])
						{
							int j_src = left_shift + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
							memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w-cut_x[i]) * sizeof(float));
						}
						if (i_mixup == 2 && y >= cut_y[i])
						{
							int j_src = (w - cut_x[i] - right_shift) + (top_shift + y - cut_y[i])*w + k*w*h;
							memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
						}
						if (i_mixup == 3 && y >= cut_y[i])
						{
							int j_src = left_shift + (top_shift + y - cut_y[i])*w + k*w*h;
							memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w - cut_x[i]) * sizeof(float));
						}
					}
				}

				blend_truth_mosaic(d.y.vals[i], boxes, truth_size, truth, w, h, cut_x[i], cut_y[i], i_mixup, left_shift, right_shift, top_shift, bot_shift, w, h, mosaic_bound);

				Darknet::free_image(ai);
				ai.data = d.X.vals[i];
			}

			if (show_imgs && i_mixup == use_mixup)   // delete i_mixup
			{
				const int random_index = random_gen();

				Darknet::Image tmp_ai = Darknet::copy_image(ai);
				char buff[1000];
				sprintf(buff, "aug_%d_%d_%d", random_index, i, random_gen());
				int t;
				for (t = 0; t < boxes; ++t)
				{
					Darknet::Box b = float_to_box_stride(d.y.vals[i] + t*truth_size, 1);
					if (!b.x) break;
					int left = (b.x - b.w / 2.)*ai.w;
					int right = (b.x + b.w / 2.)*ai.w;
					int top = (b.y - b.h / 2.)*ai.h;
					int bot = (b.y + b.h / 2.)*ai.h;
					Darknet::draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
				}

				Darknet::save_image(tmp_ai, buff);
				if (show_imgs == 1)
				{
					Darknet::show_image(tmp_ai, buff);
					cv::waitKey(0);
				}
				Darknet::free_image(tmp_ai);
			}

			free(truth);
		}

		if (random_paths)
		{
			free(random_paths);
		}
	}

	return d;
}


void Darknet::load_single_image_data(load_args args)
{
	TAT(TATPARMS);

	// Note:  even though the name is load_single_image_data() note that this will likely result in more than 1 image
	// loaded due to the args.n parameter.

	if (args.aspect		== 0.0f)	args.aspect		= 1.0f;
	if (args.exposure	== 0.0f)	args.exposure	= 1.0f;
	if (args.saturation	== 0.0f)	args.saturation	= 1.0f;

	switch (args.type)
	{
		case IMAGE_DATA:
		{
			// 2024:  used in coco.cpp, detector.cpp, yolo.cpp
			*(args.im) = Darknet::load_image(args.path, 0, 0, args.c);
			*(args.resized) = Darknet::resize_image(*(args.im), args.w, args.h);
			break;
		}
		case LETTERBOX_DATA:
		{
			// 2024:  used in detector.cpp
			*(args.im) = Darknet::load_image(args.path, 0, 0, args.c);
			*(args.resized) = Darknet::letterbox_image(*(args.im), args.w, args.h);
			break;
		}
		case DETECTION_DATA:
		{
			// 2024:  used in detector.cpp (when training a neural network)
			*args.d = load_data_detection(args.n, args.paths, args.m, args.w, args.h, args.c, args.num_boxes, args.truth_size, args.classes, args.flip, args.gaussian_noise, args.blur, args.mixup, args.jitter, args.resize,
					args.hue, args.saturation, args.exposure, args.mini_batch, args.track, args.augment_speed, args.letter_box, args.mosaic_bound, args.contrastive, args.contrastive_jit_flip, args.contrastive_color, args.show_imgs);
			break;
		}
	}

	return;
}


void Darknet::image_loading_loop(const int idx, load_args args)
{
	TAT_REVIEWED(TATPARMS, "2024-04-11");

	/* This loop runs on a secondary thread.  There are several of these threads started when the training starts,
	 * and they stay active throughout the training process, until stop_image_loading_threads() is eventually called.
	 */

	const std::string name = "image loading loop #" + std::to_string(idx);
	cfg_and_state.set_thread_name(name);

	const int number_of_threads	= args.threads;	// typically will be 6
	const int number_of_images	= args.n;		// typically will be 64 (batch size)

	// calculate the number of images this thread needs to load at once
	// e.g., 64 batch size / 6 threads = 10 or 11 images per thread
	args.n = (idx + 1) * number_of_images / number_of_threads - idx * number_of_images / number_of_threads;

	while (image_data_loading_threads_must_exit == false and
			cfg_and_state.must_immediately_exit == false)
	{
		/// @todo get rid of this busy-loop

		// wait until the control thread tells us we can load the next set of images
		if (data_loading_per_thread_flag[idx] == 0)
		{
			Darknet::TimingAndTracking tat2(name, false, "SLEEPING!");
			std::this_thread::sleep_for(thread_wait_ms);
			continue;
		}

		// if we get here, then the control thread has told us to load the next images

		args_swap_mutex.lock();
		load_args args_local = args_swap[idx];
		args_swap_mutex.unlock();

		Darknet::load_single_image_data(args_local);

		data_loading_per_thread_flag[idx] = 0;
	}

	cfg_and_state.del_thread_name();

	return;
}


void Darknet::run_image_loading_control_thread(load_args args)
{
	TAT(TATPARMS);

	/* NOTE:  This is normally started on a new thread!  For example, you might see this:
	 *
	 *		std::thread t(Darknet::run_image_loading_control_thread, args);
	 *
	 * There is a new one of these threads started to run this function at *every* iteration!
	 */

	const std::string name = "image loading control thread";
	cfg_and_state.set_thread_name(name);

	const auto timestamp1 = std::chrono::high_resolution_clock::now();

	if (args.threads == 0)
	{
		args.threads = 1;
	}
	const int number_of_threads	= args.threads;	// typically will be 6
	const int number_of_images	= args.n;		// typically will be 64 (batch size)

	data * out = args.d;
	data * buffers = (data*)xcalloc(number_of_threads, sizeof(data));

	// create the secondary threads (this should only happen once)
	if (data_loading_threads.empty())
	{
		*cfg_and_state.output << "Creating " << number_of_threads << " permanent CPU threads to load images and bounding boxes." << std::endl;

		data_loading_threads			.reserve(number_of_threads);
		data_loading_per_thread_flag	.reserve(number_of_threads);

		args_swap = (load_args *)xcalloc(number_of_threads, sizeof(load_args));

		for (int idx = 0; idx < number_of_threads; ++idx)
		{
			data_loading_per_thread_flag.push_back(0);
			data_loading_threads.emplace_back(image_loading_loop, idx, args);
		}
	}

	// tell each thread that we want more images, and where they can be stored
	for (int idx = 0; idx < number_of_threads; ++idx)
	{
		args.d = buffers + idx;
		args.n = (idx + 1) * number_of_images / number_of_threads - idx * number_of_images / number_of_threads;

		args_swap_mutex.lock();
		args_swap[idx] = args;
		args_swap_mutex.unlock();

		data_loading_per_thread_flag[idx] = 1;
	}

	// wait for the loading threads to be done
	for (int idx = 0; idx < number_of_threads; ++idx)
	{
		while (image_data_loading_threads_must_exit == false and
				cfg_and_state.must_immediately_exit == false and
				data_loading_per_thread_flag[idx] != 0) // the loading thread will reset this flag to zero once it is ready
		{
			Darknet::TimingAndTracking tat2(name, false, "SLEEPING!");
			std::this_thread::sleep_for(thread_wait_ms);
		}
	}

	// process the results
	*out = concat_datas(buffers, number_of_threads);
	out->shallow = 0;

	for (int idx = 0; idx < number_of_threads; ++idx)
	{
		buffers[idx].shallow = 1;
		Darknet::free_data(buffers[idx]);
	}
	free(buffers);

	const auto timestamp2 = std::chrono::high_resolution_clock::now();
	out->nanoseconds_to_load = std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp2 - timestamp1).count();

	cfg_and_state.del_thread_name();

	return;
}


void Darknet::stop_image_loading_threads()
{
	TAT(TATPARMS);

	if (not data_loading_threads.empty())
	{
		image_data_loading_threads_must_exit = true;

		for (auto & t : data_loading_threads)
		{
			if (t.joinable())
			{
				t.join();
			}
		}
		free(args_swap);
		data_loading_threads.clear();

		image_data_loading_threads_must_exit = false;
	}

	return;
}


matrix concat_matrix(matrix m1, matrix m2)
{
	TAT(TATPARMS);

	int i, count = 0;
	matrix m;
	m.cols = m1.cols;
	m.rows = m1.rows+m2.rows;
	m.vals = (float**)xcalloc(m1.rows + m2.rows, sizeof(float*));
	for(i = 0; i < m1.rows; ++i)
	{
		m.vals[count++] = m1.vals[i];
	}
	for(i = 0; i < m2.rows; ++i)
	{
		m.vals[count++] = m2.vals[i];
	}
	return m;
}


data concat_data(data d1, data d2)
{
	TAT(TATPARMS);

	data d = {0};
	d.shallow = 1;
	d.X = concat_matrix(d1.X, d2.X);
	d.y = concat_matrix(d1.y, d2.y);

	return d;
}


void get_next_batch(data d, int n, int offset, float *X, float *y)
{
	TAT(TATPARMS);

	int j;
	for(j = 0; j < n; ++j)
	{
		int index = offset + j;
		memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
		memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
	}
}


data get_data_part(data d, int part, int total)
{
	TAT(TATPARMS);

	data p = {0};
	p.shallow = 1;
	p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
	p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
	p.X.cols = d.X.cols;
	p.y.cols = d.y.cols;
	p.X.vals = d.X.vals + d.X.rows * part / total;
	p.y.vals = d.y.vals + d.y.rows * part / total;

	return p;
}
