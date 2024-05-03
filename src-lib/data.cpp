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
}


list *get_paths(char *filename)
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
			start_time_indexes[i] = random_gen() % m;
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
		int index = random_gen() % m;
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
	return get_random_paths_custom(paths, n, m, 0);
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
	TAT(TATPARMS);

	char** replace_paths = (char**)xcalloc(n, sizeof(char*));
	int i;
	for(i = 0; i < n; ++i)
	{
		char replaced[4096];
		find_replace(paths[i], find, replace, replaced);
		replace_paths[i] = copy_string(replaced);
	}

	return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
	TAT(TATPARMS);

	int i;
	matrix X;
	X.rows = n;
	X.vals = (float**)xcalloc(X.rows, sizeof(float*));
	X.cols = 0;

	for(i = 0; i < n; ++i)
	{
		image im = load_image(paths[i], w, h, 3);	///< @todo #COLOR

		image gray = grayscale_image(im);
		free_image(im);
		im = gray;

		X.vals[i] = im.data;
		X.cols = im.h*im.w*im.c;
	}

	return X;
}

matrix load_image_paths(char **paths, int n, int w, int h, int c)
{
	TAT(TATPARMS);

	int i;
	matrix X;
	X.rows = n;
	X.vals = (float**)xcalloc(X.rows, sizeof(float*));
	X.cols = 0;

	for (i = 0; i < n; ++i)
	{
		image im = load_image(paths[i], w, h, c);  ///< @todo #COLOR
		X.vals[i] = im.data;
		X.cols = im.h*im.w*im.c;
	}

	return X;
}

matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int w, int h, int c, float angle, float aspect, float hue, float saturation, float exposure, int contrastive)
{
	TAT(TATPARMS);

	int i;
	matrix X;
	X.rows = n;
	X.vals = (float**)xcalloc(X.rows, sizeof(float*));
	X.cols = 0;

	for(i = 0; i < n; ++i)
	{
		int size = w > h ? w : h;
		const int img_index = (contrastive) ? (i / 2) : i;
		image im = load_image(paths[img_index], 0, 0, c);

		image crop = random_augment_image(im, angle, aspect, min, max, size);
		int flip = use_flip ? random_gen() % 2 : 0;
		if (flip)
		{
			flip_image(crop);
		}
		random_distort_image(crop, hue, saturation, exposure);

		image sized = resize_image(crop, w, h);

		//show_image(im, "orig");
		//show_image(sized, "sized");
		//show_image(sized, paths[img_index]);
		//wait_until_press_key_cv();
		//printf("w = %d, h = %d \n", sized.w, sized.h);

		free_image(im);
		free_image(crop);
		X.vals[i] = sized.data;
		X.cols = sized.h*sized.w*sized.c;
	}
	return X;
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
	//printf(" img_hash = %d, filename = %s; ", img_hash, filename);
	float x, y, h, w;
	int id;
	int count = 0;
	while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
	{
		boxes = (box_label*)xrealloc(boxes, (count + 1) * sizeof(box_label));
		boxes[count].track_id = count + img_hash;
		//printf(" boxes[count].track_id = %d, count = %d \n", boxes[count].track_id, count);
		boxes[count].id = id;
		boxes[count].x = x;
		boxes[count].y = y;
		boxes[count].h = h;
		boxes[count].w = w;
		boxes[count].left   = x - w/2;
		boxes[count].right  = x + w/2;
		boxes[count].top    = y - h/2;
		boxes[count].bottom = y + h/2;
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
		const auto index = random_gen()%n;
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

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
	TAT(TATPARMS);

	char labelpath[4096];
	replace_image_to_label(path, labelpath);

	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	float x,y,w,h;
	int id;
	int i;

	for (i = 0; i < count && i < 30; ++i)
	{
		x =  boxes[i].x;
		y =  boxes[i].y;
		w =  boxes[i].w;
		h =  boxes[i].h;
		id = boxes[i].id;

		if (w < 0.0f || h < 0.0f)
		{
			continue;
		}

		int index = (4+classes) * i;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;

		if (id < classes)
		{
			truth[index+id] = 1;
		}
	}
	free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
	TAT(TATPARMS);

	char labelpath[4096];
	replace_image_to_label(path, labelpath);

	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	float x,y,w,h;
	int id;
	int i;

	for (i = 0; i < count; ++i)
	{
		x =  boxes[i].x;
		y =  boxes[i].y;
		w =  boxes[i].w;
		h =  boxes[i].h;
		id = boxes[i].id;

		if (w < 0.001f || h < 0.001f)
		{
			continue;
		}

		int col = (int)(x*num_boxes);
		int row = (int)(y*num_boxes);

		x = x*num_boxes - col;
		y = y*num_boxes - row;

		int index = (col+row*num_boxes)*(5+classes);
		if (truth[index])
		{
			continue;
		}
		truth[index++] = 1;

		if (id < classes)
		{
			truth[index+id] = 1;
		}
		index += classes;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;
	}

	free(boxes);
}

int fill_truth_detection(const char *path, int num_boxes, int truth_size, float *truth, int classes, int flip, float dx, float dy, float sx, float sy, int net_w, int net_h)
{
	// This method is used during the training process to load the boxes for the given image.

	TAT(TATPARMS);

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

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-overflow"
#endif

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
			//sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
			//system(buff);
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

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

		if (x == 0) x += lowest_w;
		if (y == 0) y += lowest_h;

		truth[(i-sub)*truth_size +0] = x;
		truth[(i-sub)*truth_size +1] = y;
		truth[(i-sub)*truth_size +2] = w;
		truth[(i-sub)*truth_size +3] = h;
		truth[(i-sub)*truth_size +4] = id;
		truth[(i-sub)*truth_size +5] = track_id;
		//float val = track_id;
		//printf(" i = %d, sub = %d, truth_size = %d, track_id = %d, %f, %f\n", i, sub, truth_size, track_id, truth[(i - sub)*truth_size + 5], val);

		if (min_w_h == 0) min_w_h = w*net_w;
		if (min_w_h > w*net_w) min_w_h = w*net_w;
		if (min_w_h > h*net_h) min_w_h = h*net_h;
	}
	free(boxes);
	return min_w_h;
}


void print_letters(float *pred, int n)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < n; ++i)
	{
		int index = max_index(pred+i*NUMCHARS, NUMCHARS);
		printf("%c", int_to_alphanum(index));
	}
	printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
	TAT(TATPARMS);

	char *begin = strrchr(path, '/');
	++begin;
	int i;
	for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i)
	{
		int index = alphanum_to_int(begin[i]);
		if(index > 35)
		{
			printf("Bad %c\n", begin[i]);
		}
		truth[i*NUMCHARS+index] = 1;
	}
	for(;i < n; ++i)
	{
		truth[i*NUMCHARS + NUMCHARS-1] = 1;
	}
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
	TAT(TATPARMS);

	int i;
	memset(truth, 0, k*sizeof(float));
	int count = 0;
	for(i = 0; i < k; ++i)
	{
		if(strstr(path, labels[i]))
		{
			truth[i] = 1;
			++count;
		}
	}
	if (count != 1)
	{
		printf("Too many or too few labels: %d, %s\n", count, path);
		count = 0;
		for (i = 0; i < k; ++i)
		{
			if (strstr(path, labels[i]))
			{
				printf("\t label %d: %s  \n", count, labels[i]);
				count++;
			}
		}
	}
}

void fill_truth_smooth(char *path, char **labels, int k, float *truth, float label_smooth_eps)
{
	TAT(TATPARMS);

	int i;
	memset(truth, 0, k * sizeof(float));
	int count = 0;
	for (i = 0; i < k; ++i)
	{
		if (strstr(path, labels[i]))
		{
			truth[i] = (1 - label_smooth_eps);
			++count;
		}
		else
		{
			truth[i] = label_smooth_eps / (k - 1);
		}
	}
	if (count != 1)
	{
		printf("Too many or too few labels: %d, %s\n", count, path);
		count = 0;
		for (i = 0; i < k; ++i)
		{
			if (strstr(path, labels[i]))
			{
				printf("\t label %d: %s  \n", count, labels[i]);
				count++;
			}
		}
	}
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
	TAT(TATPARMS);

	int j;
	for (j = 0; j < k; ++j)
	{
		if(truth[j])
		{
			int parent = hierarchy->parent[j];
			while(parent >= 0)
			{
				truth[parent] = 1;
				parent = hierarchy->parent[parent];
			}
		}
	}
	int i;
	int count = 0;
	for (j = 0; j < hierarchy->groups; ++j)
	{
		//printf("%d\n", count);
		int mask = 1;
		for (i = 0; i < hierarchy->group_size[j]; ++i)
		{
			if (truth[count + i])
			{
				mask = 0;
				break;
			}
		}
		if (mask)
		{
			for (i = 0; i < hierarchy->group_size[j]; ++i)
			{
				truth[count + i] = SECRET_NUM;
			}
		}
		count += hierarchy->group_size[j];
	}
}

int find_max(float *arr, int size)
{
	TAT(TATPARMS);

	int i;
	float max = 0;
	int n = 0;
	for (i = 0; i < size; ++i)
	{
		if (arr[i] > max)
		{
			max = arr[i];
			n = i;
		}
	}
	return n;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy, float label_smooth_eps, int contrastive)
{
	TAT(TATPARMS);

	matrix y = make_matrix(n, k);
	int i;
	if (labels)
	{
		// supervised learning
		for (i = 0; i < n; ++i)
		{
			const int img_index = (contrastive) ? (i / 2) : i;
			fill_truth_smooth(paths[img_index], labels, k, y.vals[i], label_smooth_eps);
			//printf(" n = %d, i = %d, img_index = %d, class_id = %d \n", n, i, img_index, find_max(y.vals[i], k));
			if (hierarchy)
			{
				fill_hierarchy(y.vals[i], k, hierarchy);
			}
		}
	}
	else
	{
		// unsupervised learning
		for (i = 0; i < n; ++i)
		{
			const int img_index = (contrastive) ? (i / 2) : i;
			const uintptr_t path_p = (uintptr_t)paths[img_index];// abs(random_gen());
			const int class_id = path_p % k;
			int l;
			for (l = 0; l < k; ++l)
			{
				y.vals[i][l] = 0;
			}
			y.vals[i][class_id] = 1;
		}
	}
	return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
	TAT(TATPARMS);

	matrix y = make_matrix(n, k);
	int i;
	int count = 0;
	for(i = 0; i < n; ++i)
	{
		char label[4096];
		find_replace(paths[i], "imgs", "labels", label);
		find_replace(label, "_iconl.jpeg", ".txt", label);
		FILE *file = fopen(label, "r");
		if(!file)
		{
			find_replace(label, "labels", "labels2", label);
			file = fopen(label, "r");
			if(!file)
			{
				continue;
			}
		}
		++count;
		int tag;
		while(fscanf(file, "%d", &tag) == 1)
		{
			if(tag < k)
			{
				y.vals[i][tag] = 1;
			}
		}
		fclose(file);
	}
	printf("%d/%d\n", count, n);
	return y;
}

char **get_labels_custom(char *filename, int *size)
{
	TAT(TATPARMS);

	list *plist = get_paths(filename);
	if (size)
	{
		*size = plist->size;
	}
	char **labels = (char **)list_to_array(plist);
	free_list(plist);
	return labels;
}

char **get_labels(char *filename)
{
	return get_labels_custom(filename, NULL);
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

data load_data_region(int n, char **paths, int m, int w, int h, int c, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
	TAT(TATPARMS);

	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = {0};
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
	d.X.cols = h*w*3;


	int k = size*size*(5+classes);
	d.y = make_matrix(n, k);
	for (i = 0; i < n; ++i)
	{
		image orig = load_image(random_paths[i], 0, 0, c);

		int oh = orig.h;
		int ow = orig.w;

		int dw = (ow*jitter);
		int dh = (oh*jitter);

		int pleft  = rand_uniform(-dw, dw);
		int pright = rand_uniform(-dw, dw);
		int ptop   = rand_uniform(-dh, dh);
		int pbot   = rand_uniform(-dh, dh);

		int swidth =  ow - pleft - pright;
		int sheight = oh - ptop - pbot;

		float sx = (float)swidth  / ow;
		float sy = (float)sheight / oh;

		int flip = random_gen()%2;
		image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

		float dx = ((float)pleft/ow)/sx;
		float dy = ((float)ptop /oh)/sy;

		image sized = resize_image(cropped, w, h);
		if(flip) flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;

		fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

		free_image(orig);
		free_image(cropped);
	}

	free(random_paths);
	return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h, int c)
{
	TAT(TATPARMS);

	if(m) paths = get_random_paths(paths, 2*n, m);
	int i,j;
	data d = {0};
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
	d.X.cols = h*w*6;

	int k = 2*(classes);
	d.y = make_matrix(n, k);
	for(i = 0; i < n; ++i)
	{
		image im1 = load_image(paths[i*2],   w, h, c);
		image im2 = load_image(paths[i*2+1], w, h, c);

		d.X.vals[i] = (float*)xcalloc(d.X.cols, sizeof(float));
		memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
		memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

		int id;
		float iou;

		char imlabel1[4096];
		char imlabel2[4096];
		find_replace(paths[i*2],   "imgs", "labels", imlabel1);
		find_replace(imlabel1, "jpg", "txt", imlabel1);
		FILE *fp1 = fopen(imlabel1, "r");

		while(fscanf(fp1, "%d %f", &id, &iou) == 2)
		{
			if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
		}

		find_replace(paths[i*2+1], "imgs", "labels", imlabel2);
		find_replace(imlabel2, "jpg", "txt", imlabel2);
		FILE *fp2 = fopen(imlabel2, "r");

		while(fscanf(fp2, "%d %f", &id, &iou) == 2){
			if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
		}

		for (j = 0; j < classes; ++j)
		{
			if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < 0.5f)
			{
				d.y.vals[i][2*j] = 1;
				d.y.vals[i][2*j+1] = 0;
			}
			else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > 0.5f)
			{
				d.y.vals[i][2*j] = 0;
				d.y.vals[i][2*j+1] = 1;
			}
			else
			{
				d.y.vals[i][2*j]   = SECRET_NUM;
				d.y.vals[i][2*j+1] = SECRET_NUM;
			}
		}
		fclose(fp1);
		fclose(fp2);

		free_image(im1);
		free_image(im2);
	}

	if(m)
	{
		free(paths);
	}

	return d;
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
	//printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}


void blend_truth_mosaic(float *new_truth, int boxes, int truth_size, float *old_truth, int w, int h, float cut_x, float cut_y, int i_mixup,
	int left_shift, int right_shift, int top_shift, int bot_shift,
	int net_w, int net_h, int mosaic_bound)
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
				//printf(" i_mixup = %d, left = %d, left_bound = %f \n", i_mixup, left, left_bound);
				left = left_bound;
			}
			if (right > right_bound)
			{
				//printf(" i_mixup = %d, right = %d, right_bound = %f \n", i_mixup, right, right_bound);
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
	//printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}


data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int use_gaussian_noise, int use_blur, int use_mixup,
	float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs)
{
	// This is the method that gets called to load the "n" images for each loading thread while training a network.

	TAT(TATPARMS);

	c = c ? c : 3;

	if (use_mixup == 2 || use_mixup == 4)
	{
		darknet_fatal_error(DARKNET_LOC, "cutmix=1 isn't supported for detector (only classifier can use cutmix=1)");
	}

	if (use_mixup == 3 && letter_box)
	{
		//printf("\n Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters \n");
		//if (check_mistakes) getzzzchar();
		//exit(0);
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

			mat_cv *src;
			src = load_image_mat_cv(filename, c);
			if (src == NULL)
			{
				darknet_fatal_error(DARKNET_LOC, "failed to read image \"%s\"", filename);
			}

			const int oh = get_height_mat(src);	// original height
			const int ow = get_width_mat(src);	// original width

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

			//printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh = %d \n", pleft, pright, ptop, pbot, ow, oh);

			//float scale = rand_precalc_random(.25, 2, r_scale); // unused currently
			//printf(" letter_box = %d \n", letter_box);

			if (letter_box)
			{
				float img_ar = (float)ow / (float)oh;
				float net_ar = (float)w / (float)h;
				float result_ar = img_ar / net_ar;
				//printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
				if (result_ar > 1)  // sheight - should be increased
				{
					float oh_tmp = ow / net_ar;
					float delta_h = (oh_tmp - oh)/2;
					ptop = ptop - delta_h;
					pbot = pbot - delta_h;
					//printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
				}
				else  // swidth - should be increased
				{
					float ow_tmp = oh * net_ar;
					float delta_w = (ow_tmp - ow)/2;
					pleft = pleft - delta_w;
					pright = pright - delta_w;
					//printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
				}

				//printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh = %d \n", pleft, pright, ptop, pbot, ow, oh);
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
			//for (int z = 0; z < boxes; ++z) if(truth[z*truth_size] > 0) printf(" track_id = %f \n", truth[z*truth_size + 5]);
			//printf(" truth_size = %d \n", truth_size);

			if ((min_w_h / 8) < blur && blur > 1)
			{
				blur = min_w_h / 8;   // disable blur if one of the objects is too small
			}

			image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur, boxes, truth_size, truth);

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
					image old_img = make_empty_image(w, h, c);
					old_img.data = d.X.vals[i];
					//show_image(ai, "new");
					//show_image(old_img, "old");
					//wait_until_press_key_cv();
					blend_images_cv(ai, 0.5, old_img, 0.5);
					blend_truth(d.y.vals[i], boxes, truth_size, truth);
					free_image(old_img);
					d.X.vals[i] = ai.data;
				}
			}
			else if (use_mixup == 3)
			{
				if (i_mixup == 0)
				{
					image tmp_img = make_image(w, h, c);
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

				free_image(ai);
				ai.data = d.X.vals[i];
			}

			if (show_imgs && i_mixup == use_mixup)   // delete i_mixup
			{
				const int random_index = random_gen();

				image tmp_ai = copy_image(ai);
				char buff[1000];
				//sprintf(buff, "aug_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
				sprintf(buff, "aug_%d_%d_%d", random_index, i, random_gen());
				int t;
				for (t = 0; t < boxes; ++t)
				{
					box b = float_to_box_stride(d.y.vals[i] + t*truth_size, 1);
					if (!b.x) break;
					int left = (b.x - b.w / 2.)*ai.w;
					int right = (b.x + b.w / 2.)*ai.w;
					int top = (b.y - b.h / 2.)*ai.h;
					int bot = (b.y + b.h / 2.)*ai.h;
					draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
				}

				save_image(tmp_ai, buff);
				if (show_imgs == 1)
				{
					//char buff_src[1000];
					//sprintf(buff_src, "src_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
					//show_image_mat(src, buff_src);
					show_image(tmp_ai, buff);
					wait_until_press_key_cv();
				}
				printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
				free_image(tmp_ai);
			}

			release_mat(&src);
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
	// Note:  even though the name is load_single_image_data() note that this will likely result in more than 1 image
	// loaded due to the args.n parameter.

	TAT(TATPARMS);

	if (args.aspect		== 0.0f)	args.aspect		= 1.0f;
	if (args.exposure	== 0.0f)	args.exposure	= 1.0f;
	if (args.saturation	== 0.0f)	args.saturation	= 1.0f;

	switch (args.type)
	{
		case IMAGE_DATA:
		{
			// 2024:  used in coco.cpp, detector.cpp, yolo.cpp
			*(args.im) = load_image(args.path, 0, 0, args.c);
			*(args.resized) = resize_image(*(args.im), args.w, args.h);
			break;
		}
		case LETTERBOX_DATA:
		{
			// 2024:  used in detector.cpp
			*(args.im) = load_image(args.path, 0, 0, args.c);
			*(args.resized) = letterbox_image(*(args.im), args.w, args.h);
			break;
		}
		case DETECTION_DATA:
		{
			// 2024:  used in detector.cpp (when training a neural network)
			*args.d = load_data_detection(args.n, args.paths, args.m, args.w, args.h, args.c, args.num_boxes, args.truth_size, args.classes, args.flip, args.gaussian_noise, args.blur, args.mixup, args.jitter, args.resize,
					args.hue, args.saturation, args.exposure, args.mini_batch, args.track, args.augment_speed, args.letter_box, args.mosaic_bound, args.contrastive, args.contrastive_jit_flip, args.contrastive_color, args.show_imgs);
			break;
		}
		case OLD_CLASSIFICATION_DATA:
		{
			// 2024:  used in classifier.cpp
			*args.d = load_data_old(args.paths, args.n, args.m, args.labels, args.classes, args.w, args.h, args.c);
			break;
		}
		case CLASSIFICATION_DATA:
		{
			// 2024:  used in captcha.cpp and classifier.cpp
			*args.d = load_data_augment(args.paths, args.n, args.m, args.labels, args.classes, args.hierarchy, args.flip, args.min, args.max, args.w, args.h, args.c, args.angle, args.aspect, args.hue, args.saturation, args.exposure, args.mixup, args.blur, args.show_imgs, args.label_smooth_eps, args.contrastive);
			break;
		}
		case SUPER_DATA:
		{
			// 2024:  used in super.cpp and voxel.cpp
			*args.d = load_data_super(args.paths, args.n, args.m, args.w, args.h, args.c, args.scale);
			break;
		}
		case WRITING_DATA:
		{
			// 2024:  used in writing.cpp
			*args.d = load_data_writing(args.paths, args.n, args.m, args.w, args.h,args.c, args.out_w, args.out_h);
			break;
		}
		case REGION_DATA:
		{
			// 2024:  used in coco.cpp and yolo.cpp
			*args.d = load_data_region(args.n, args.paths, args.m, args.w, args.h, args.c, args.num_boxes, args.classes, args.jitter, args.hue, args.saturation, args.exposure);
			break;
		}
		case COMPARE_DATA:
		{
			// 2024:  used in compare.cpp
			*args.d = load_data_compare(args.n, args.paths, args.m, args.classes, args.w, args.h, args.c);
			break;
		}
		case TAG_DATA:
		{
			// 2024:  used in tag.cpp
			*args.d = load_data_tag(args.paths, args.n, args.m, args.classes, args.flip, args.min, args.max, args.w, args.h, args.c, args.angle, args.aspect, args.hue, args.saturation, args.exposure);
			break;
		}
	}

	return;
}


void Darknet::image_loading_loop(const int idx, load_args args)
{
	// This loop runs on a secondary thread.

	TAT_REVIEWED(TATPARMS, "2024-04-11");

	cfg_and_state.set_thread_name("image loading loop #" + std::to_string(idx));

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
			std::this_thread::sleep_for(thread_wait_ms);
			continue;
		}

		// if we get here, then the control thread has told us to load the next image

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
	/* NOTE:  This is normally started on a new thread!  For example, you might see this:
	 *
	 *		std::thread t(Darknet::run_image_loading_control_thread, args);
	 */

	TAT(TATPARMS);

	cfg_and_state.set_thread_name("image loading control thread");

	if (args.threads == 0)
	{
		args.threads = 1;
	}
	const int number_of_threads	= args.threads;	// typically will be 6
	const int number_of_images	= args.n;		// typically will be 64 (batch size)

	data * out = args.d;
	data * buffers = (data*)xcalloc(number_of_threads, sizeof(data));

	// create the secondary threads
	if (data_loading_threads.empty())
	{
		std::cout << "Creating " << number_of_threads << " permanent CPU threads to load images and bounding boxes." << std::endl;

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


data load_data_writing(char **paths, int n, int m, int w, int h, int c, int out_w, int out_h)
{
	TAT(TATPARMS);

	if(m) paths = get_random_paths(paths, n, m);
	char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
	data d = {0};
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h, c);
	d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
	if(m) free(paths);
	int i;
	for(i = 0; i < n; ++i) free(replace_paths[i]);
	free(replace_paths);
	return d;
}


data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h, int c)
{
	TAT(TATPARMS);

	if(m) paths = get_random_paths(paths, n, m);
	data d = {0};
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h, c);
	d.y = load_labels_paths(paths, n, labels, k, 0, 0, 0);
	if(m) free(paths);
	return d;
}


data load_data_super(char **paths, int n, int m, int w, int h, int c, int scale)
{
	TAT(TATPARMS);

	if(m) paths = get_random_paths(paths, n, m);
	data d = {0};
	d.shallow = 0;

	int i;
	d.X.rows = n;
	d.X.vals = (float**)xcalloc(n, sizeof(float*));
	d.X.cols = w*h*c;

	d.y.rows = n;
	d.y.vals = (float**)xcalloc(n, sizeof(float*));
	d.y.cols = w*scale * h*scale * c;

	for(i = 0; i < n; ++i)
	{
		image im = load_image(paths[i], 0, 0, c);
		image crop = random_crop_image(im, w*scale, h*scale);
		int flip = random_gen()%2;
		if (flip)
		{
			flip_image(crop);
		}
		image resize = resize_image(crop, w, h);
		d.X.vals[i] = resize.data;
		d.y.vals[i] = crop.data;
		free_image(im);
	}

	if(m) free(paths);
	return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min, int max, int w, int h, int c, float angle,
	float aspect, float hue, float saturation, float exposure, int use_mixup, int use_blur, int show_imgs, float label_smooth_eps, int contrastive)
{
	TAT(TATPARMS);

	char **paths_stored = paths;
	if(m) paths = get_random_paths(paths, n, m);
	data d = {0};
	d.shallow = 0;
	d.X = load_image_augment_paths(paths, n, use_flip, min, max, w, h, c, angle, aspect, hue, saturation, exposure, contrastive);
	d.y = load_labels_paths(paths, n, labels, k, hierarchy, label_smooth_eps, contrastive);

	if (use_mixup && rand_int(0, 1))
	{
		char **paths_mix = get_random_paths(paths_stored, n, m);
		data d2 = { 0 };
		d2.shallow = 0;
		d2.X = load_image_augment_paths(paths_mix, n, use_flip, min, max, w, h, c, angle, aspect, hue, saturation, exposure, contrastive);
		d2.y = load_labels_paths(paths_mix, n, labels, k, hierarchy, label_smooth_eps, contrastive);
		free(paths_mix);

		data d3 = { 0 };
		d3.shallow = 0;
		data d4 = { 0 };
		d4.shallow = 0;
		if (use_mixup >= 3)
		{
			char **paths_mix3 = get_random_paths(paths_stored, n, m);
			d3.X = load_image_augment_paths(paths_mix3, n, use_flip, min, max, w, h, c, angle, aspect, hue, saturation, exposure, contrastive);
			d3.y = load_labels_paths(paths_mix3, n, labels, k, hierarchy, label_smooth_eps, contrastive);
			free(paths_mix3);

			char **paths_mix4 = get_random_paths(paths_stored, n, m);
			d4.X = load_image_augment_paths(paths_mix4, n, use_flip, min, max, w, h, c, angle, aspect, hue, saturation, exposure, contrastive);
			d4.y = load_labels_paths(paths_mix4, n, labels, k, hierarchy, label_smooth_eps, contrastive);
			free(paths_mix4);
		}


		// mix
		int i, j;
		for (i = 0; i < d2.X.rows; ++i)
		{
			int mixup = use_mixup;
			if (use_mixup == 4)
			{
				mixup = rand_int(2, 3); // alternate CutMix and Mosaic
			}

			// MixUp -----------------------------------
			if (mixup == 1)
			{
				// mix images
				for (j = 0; j < d2.X.cols; ++j)
				{
					d.X.vals[i][j] = (d.X.vals[i][j] + d2.X.vals[i][j]) / 2.0f;
				}

				// mix labels
				for (j = 0; j < d2.y.cols; ++j)
				{
					d.y.vals[i][j] = (d.y.vals[i][j] + d2.y.vals[i][j]) / 2.0f;
				}
			}
			// CutMix -----------------------------------
			else if (mixup == 2)
			{
				const float min = 0.3;  // 0.3*0.3 = 9%
				const float max = 0.8;  // 0.8*0.8 = 64%
				const int cut_w = rand_int(w*min, w*max);
				const int cut_h = rand_int(h*min, h*max);
				const int cut_x = rand_int(0, w - cut_w - 1);
				const int cut_y = rand_int(0, h - cut_h - 1);
				const int left = cut_x;
				const int right = cut_x + cut_w;
				const int top = cut_y;
				const int bot = cut_y + cut_h;

				assert(cut_x >= 0 && cut_x <= w);
				assert(cut_y >= 0 && cut_y <= h);
				assert(cut_w >= 0 && cut_w <= w);
				assert(cut_h >= 0 && cut_h <= h);

				assert(right >= 0 && right <= w);
				assert(bot >= 0 && bot <= h);

				assert(top <= bot);
				assert(left <= right);

				const float alpha = (float)(cut_w*cut_h) / (float)(w*h);
				const float beta = 1 - alpha;

				int channel, x, y;
				for (channel = 0; channel < 3; ++channel) ///< @todo #COLOR
				{
					for (y = top; y < bot; ++y)
					{
						for (x = left; x < right; ++x)
						{
							int j = x + y*w + channel *w*h;
							d.X.vals[i][j] = d2.X.vals[i][j];
						}
					}
				}

				//printf("\n alpha = %f, beta = %f \n", alpha, beta);
				// mix labels
				for (j = 0; j < d.y.cols; ++j)
				{
					d.y.vals[i][j] = d.y.vals[i][j] * beta + d2.y.vals[i][j] * alpha;
				}
			}
			// Mosaic -----------------------------------
			else if (mixup == 3)
			{
				const float min_offset = 0.2; // 20%
				const int cut_x = rand_int(w*min_offset, w*(1 - min_offset));
				const int cut_y = rand_int(h*min_offset, h*(1 - min_offset));

				float s1 = (float)(cut_x * cut_y) / (w*h);
				float s2 = (float)((w - cut_x) * cut_y) / (w*h);
				float s3 = (float)(cut_x * (h - cut_y)) / (w*h);
				float s4 = (float)((w - cut_x) * (h - cut_y)) / (w*h);

				int channel, x, y;
				for (channel = 0; channel < 3; ++channel) ///< @todo #COLOR
				{
					for (y = 0; y < h; ++y)
					{
						for (x = 0; x < w; ++x)
						{
							int j = x + y*w + channel *w*h;
							if (x < cut_x && y < cut_y) d.X.vals[i][j] = d.X.vals[i][j];
							if (x >= cut_x && y < cut_y) d.X.vals[i][j] = d2.X.vals[i][j];
							if (x < cut_x && y >= cut_y) d.X.vals[i][j] = d3.X.vals[i][j];
							if (x >= cut_x && y >= cut_y) d.X.vals[i][j] = d4.X.vals[i][j];
						}
					}
				}

				for (j = 0; j < d.y.cols; ++j)
				{
					const float max_s = 1;// max_val_cmp(s1, max_val_cmp(s2, max_val_cmp(s3, s4)));

					d.y.vals[i][j] = d.y.vals[i][j] * s1 / max_s + d2.y.vals[i][j] * s2 / max_s + d3.y.vals[i][j] * s3 / max_s + d4.y.vals[i][j] * s4 / max_s;
				}
			}
		}

		Darknet::free_data(d2);

		if (use_mixup >= 3)
		{
			Darknet::free_data(d3);
			Darknet::free_data(d4);
		}
	}

	if (use_blur)
	{
		int i;
		for (i = 0; i < d.X.rows; ++i)
		{
			if (random_gen() % 4 == 0)
			{
				image im = make_empty_image(w, h, c);
				im.data = d.X.vals[i];
				int ksize = use_blur;
				if (use_blur == 1)
				{
					ksize = 15;
				}
				image blurred = blur_image(im, ksize);
				free_image(im);
				d.X.vals[i] = blurred.data;
			}
		}
	}

	if (show_imgs)
	{
		int i, j;
		for (i = 0; i < d.X.rows; ++i)
		{
			image im = make_empty_image(w, h, c);
			im.data = d.X.vals[i];
			char buff[1000];
			sprintf(buff, "aug_%d_%s_%d", i, basecfg((char*)paths[i]), random_gen());
			save_image(im, buff);

			char buff_string[1000];
			sprintf(buff_string, "\n Classes: ");
			for (j = 0; j < d.y.cols; ++j)
			{
				if (d.y.vals[i][j] > 0)
				{
					char buff_tmp[100];
					sprintf(buff_tmp, " %d (%f), ", j, d.y.vals[i][j]);
					strcat(buff_string, buff_tmp);
				}
			}
			printf("%s \n", buff_string);

			if (show_imgs == 1)
			{
				show_image(im, buff);
				wait_until_press_key_cv();
			}
		}
		printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
	}

	if (m) free(paths);

	return d;
}

data load_data_tag(char **paths, int n, int m, int k, int use_flip, int min, int max, int w, int h, int c, float angle, float aspect, float hue, float saturation, float exposure)
{
	TAT(TATPARMS);

	if(m)
	{
		paths = get_random_paths(paths, n, m);
	}
	data d = {0};
	d.w = w;
	d.h = h;
	d.shallow = 0;
	d.X = load_image_augment_paths(paths, n, use_flip, min, max, w, h, c, angle, aspect, hue, saturation, exposure, 0);
	d.y = load_tags_paths(paths, n, k);
	if(m)
	{
		free(paths);
	}

	return d;
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

data concat_datas(data *d, int n)
{
	TAT(TATPARMS);

	int i;
	data out = {0};
	for (i = 0; i < n; ++i)
	{
		data newdata = concat_data(d[i], out);
		Darknet::free_data(out);
		out = newdata;
	}

	return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
	TAT(TATPARMS);

	data d = {0};
	d.shallow = 0;
	matrix X = csv_to_matrix(filename);
	float *truth_1d = pop_column(&X, target);
	float **truth = one_hot_encode(truth_1d, X.rows, k);
	matrix y;
	y.rows = X.rows;
	y.cols = k;
	y.vals = truth;
	d.X = X;
	d.y = y;
	free(truth_1d);
	return d;
}

data load_cifar10_data(char *filename)
{
	TAT(TATPARMS);

	data d = {0};
	d.shallow = 0;
	long i,j;
	matrix X = make_matrix(10000, 3072);
	matrix y = make_matrix(10000, 10);
	d.X = X;
	d.y = y;

	FILE *fp = fopen(filename, "rb");
	if(!fp) file_error(filename, DARKNET_LOC);
	for(i = 0; i < 10000; ++i)
	{
		unsigned char bytes[3073];
		fread(bytes, 1, 3073, fp);
		int class_id = bytes[0];
		y.vals[i][class_id] = 1;
		for(j = 0; j < X.cols; ++j)
		{
			X.vals[i][j] = (double)bytes[j+1];
		}
	}
	//translate_data_rows(d, -128);
	scale_data_rows(d, 1./255);
	//normalize_data_rows(d);
	fclose(fp);
	return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
	TAT(TATPARMS);

	int j;
	for(j = 0; j < n; ++j)
	{
		int index = random_gen()%d.X.rows;
		memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
		memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
	}
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

void smooth_data(data d)
{
	TAT(TATPARMS);

	int i, j;
	float scale = 1. / d.y.cols;
	float eps = .1;
	for(i = 0; i < d.y.rows; ++i)
	{
		for(j = 0; j < d.y.cols; ++j)
		{
			d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
		}
	}
}

data load_all_cifar10()
{
	TAT(TATPARMS);

	data d = {0};
	d.shallow = 0;
	int i,j,b;
	matrix X = make_matrix(50000, 3072);
	matrix y = make_matrix(50000, 10);
	d.X = X;
	d.y = y;


	for(b = 0; b < 5; ++b)
	{
		char buff[256];
		sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
		FILE *fp = fopen(buff, "rb");
		if(!fp)
		{
			file_error(buff, DARKNET_LOC);
		}
		for (i = 0; i < 10000; ++i)
		{
			unsigned char bytes[3073];
			fread(bytes, 1, 3073, fp);
			int class_id = bytes[0];
			y.vals[i+b*10000][class_id] = 1;
			for (j = 0; j < X.cols; ++j)
			{
				X.vals[i+b*10000][j] = (double)bytes[j+1];
			}
		}
		fclose(fp);
	}
	//normalize_data_rows(d);
	//translate_data_rows(d, -128);
	scale_data_rows(d, 1./255);
	smooth_data(d);
	return d;
}

data load_go(char *filename)
{
	TAT(TATPARMS);

	FILE *fp = fopen(filename, "rb");
	matrix X = make_matrix(3363059, 361);
	matrix y = make_matrix(3363059, 361);
	int row, col;

	if(!fp) file_error(filename, DARKNET_LOC);
	char *label;
	int count = 0;
	while((label = fgetl(fp)))
	{
		int i;
		if(count == X.rows)
		{
			X = resize_matrix(X, count*2);
			y = resize_matrix(y, count*2);
		}
		sscanf(label, "%d %d", &row, &col);
		char *board = fgetl(fp);

		int index = row*19 + col;
		y.vals[count][index] = 1;

		for(i = 0; i < 19*19; ++i)
		{
			float val = 0;
			if (board[i] == '1')
			{
				val = 1;
			}
			else if(board[i] == '2')
			{
				val = -1;
			}
			X.vals[count][i] = val;
		}
		++count;
		free(label);
		free(board);
	}
	X = resize_matrix(X, count);
	y = resize_matrix(y, count);

	data d = {0};
	d.shallow = 0;
	d.X = X;
	d.y = y;


	fclose(fp);

	return d;
}


void randomize_data(data d)
{
	TAT(TATPARMS);

	int i;
	for (i = d.X.rows - 1; i > 0; --i)
	{
		const int index = random_gen() % i;

		std::swap(d.X.vals[index], d.X.vals[i]);
		std::swap(d.y.vals[index], d.y.vals[i]);
	}
}

void scale_data_rows(data d, float s)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < d.X.rows; ++i)
	{
		scale_array(d.X.vals[i], d.X.cols, s);
	}
}

void translate_data_rows(data d, float s)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < d.X.rows; ++i)
	{
		translate_array(d.X.vals[i], d.X.cols, s);
	}
}

void normalize_data_rows(data d)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < d.X.rows; ++i)
	{
		normalize_array(d.X.vals[i], d.X.cols);
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

data get_random_data(data d, int num)
{
	TAT(TATPARMS);

	data r = {0};
	r.shallow = 1;

	r.X.rows = num;
	r.y.rows = num;

	r.X.cols = d.X.cols;
	r.y.cols = d.y.cols;

	r.X.vals = (float**)xcalloc(num, sizeof(float*));
	r.y.vals = (float**)xcalloc(num, sizeof(float*));

	int i;
	for(i = 0; i < num; ++i)
	{
		int index = random_gen()%d.X.rows;
		r.X.vals[i] = d.X.vals[index];
		r.y.vals[i] = d.y.vals[index];
	}
	return r;
}

data *split_data(data d, int part, int total)
{
	TAT(TATPARMS);

	data* split = (data*)xcalloc(2, sizeof(data));
	int i;
	int start = part*d.X.rows/total;
	int end = (part+1)*d.X.rows/total;
	data train ={0};
	data test ={0};
	train.shallow = test.shallow = 1;

	test.X.rows = test.y.rows = end-start;
	train.X.rows = train.y.rows = d.X.rows - (end-start);
	train.X.cols = test.X.cols = d.X.cols;
	train.y.cols = test.y.cols = d.y.cols;

	train.X.vals = (float**)xcalloc(train.X.rows, sizeof(float*));
	test.X.vals = (float**)xcalloc(test.X.rows, sizeof(float*));
	train.y.vals = (float**)xcalloc(train.y.rows, sizeof(float*));
	test.y.vals = (float**)xcalloc(test.y.rows, sizeof(float*));

	for (i = 0; i < start; ++i)
	{
		train.X.vals[i] = d.X.vals[i];
		train.y.vals[i] = d.y.vals[i];
	}
	for (i = start; i < end; ++i)
	{
		test.X.vals[i-start] = d.X.vals[i];
		test.y.vals[i-start] = d.y.vals[i];
	}
	for (i = end; i < d.X.rows; ++i)
	{
		train.X.vals[i-(end-start)] = d.X.vals[i];
		train.y.vals[i-(end-start)] = d.y.vals[i];
	}
	split[0] = train;
	split[1] = test;
	return split;
}
