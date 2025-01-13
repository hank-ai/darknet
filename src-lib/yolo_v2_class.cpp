#ifdef __GNUC__
// 2023-06-25:  hide some of the warnings which for now we need to ignore in this file
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif


/* PLEASE NOTE:
 *
 * This file is no longer maintained, and actually had been removed from the more recent versions of Darknet in the
 * Hank.ai repo.  However, it seems some projects and language bindings still use it.  If you are starting a *NEW*
 * project or you're looking for the C++ API which is maintained, please use DarkHelp instead of this source file and
 * library.  You can find more information on the DarkHelp library here:
 *
 * - https://www.ccoderun.ca/darkhelp/api/API.html
 * - https://github.com/stephanecharette/DarkHelp#what-is-the-darkhelp-c-api
 */


#include "darknet_internal.hpp"
#include "yolo_v2_class.hpp"

#define NFRAMES 3

//static Detector* detector = NULL;
static std::unique_ptr<Detector> detector;

int init(const char *configurationFilename, const char *weightsFilename, int gpu, int batch_size)
{
	TAT(TATPARMS);

	detector.reset(new Detector(configurationFilename, weightsFilename, gpu, batch_size));
	return 1;
}

int detect_image(const char *filename, bbox_t_container &container)
{
	TAT(TATPARMS);

	std::vector<bbox_t> detection = detector->detect(filename);
	for (size_t i = 0; i < detection.size() && i < C_SHARP_MAX_OBJECTS; ++i)
		container.candidates[i] = detection[i];
	return detection.size();
}

int detect_mat(const uint8_t* data, const size_t data_length, bbox_t_container &container)
{
	TAT(TATPARMS);

	std::vector<char> vdata(data, data + data_length);
	cv::Mat image = imdecode(cv::Mat(vdata), 1);

	std::vector<bbox_t> detection = detector->detect(image);
	for (size_t i = 0; i < detection.size() && i < C_SHARP_MAX_OBJECTS; ++i)
		container.candidates[i] = detection[i];
	return detection.size();
}

int dispose()
{
	TAT(TATPARMS);

	//if (detector != NULL) delete detector;
	//detector = NULL;
	detector.reset();
	return 1;
}

int get_device_count()
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	int count = 0;
	CHECK_CUDA(cudaGetDeviceCount(&count));
	return count;
#else
	return -1;
#endif	// DARKNET_GPU
}

bool built_with_cuda()
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	return true;
#else
	return false;
#endif
}

bool built_with_cudnn()
{
	TAT(TATPARMS);

#ifdef CUDNN
	return true;
#else
	return false;
#endif
}

bool built_with_opencv()
{
	TAT(TATPARMS);

	return true;
}


int get_device_name(int gpu, char* deviceName)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu));
	std::string result = prop.name;
	std::copy(result.begin(), result.end(), deviceName);
	return 1;
#else
	return -1;
#endif	// DARKNET_GPU
}

#ifdef DARKNET_GPU
void check_cuda(cudaError_t status)
{
	TAT(TATPARMS);

	if (status != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		printf("CUDA Error Prev: %s\n", s);
	}
}
#endif

struct detector_gpu_t
{
	Darknet::Network net;
	Darknet::Image images[NFRAMES];
	float *avg;
	float* predictions[NFRAMES];
	int demo_index;
	unsigned int *track_id;
};

Detector::Detector(std::string cfg_filename, std::string weight_filename, int gpu_id, int batch_size)
	: cur_gpu_id(gpu_id)
{
	TAT(TATPARMS);

	wait_stream = 0;
#ifdef DARKNET_GPU
	int old_gpu_index;
	check_cuda( cudaGetDevice(&old_gpu_index) );
#endif

	detector_gpu_ptr = std::make_shared<detector_gpu_t>();
	detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());

#ifdef DARKNET_GPU
	//check_cuda( cudaSetDevice(cur_gpu_id) );
	cuda_set_device(cur_gpu_id);
	printf(" Used GPU %d \n", cur_gpu_id);
#endif
	Darknet::Network & net = detector_gpu.net;
	net.gpu_index = cur_gpu_id;
	//gpu_index = i;

	_cfg_filename = cfg_filename;
	_weight_filename = weight_filename;

	char *cfgfile = const_cast<char *>(_cfg_filename.c_str());
	char *weightfile = const_cast<char *>(_weight_filename.c_str());

	net = parse_network_cfg_custom(cfgfile, batch_size, batch_size);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, batch_size);
	net.gpu_index = cur_gpu_id;
	fuse_conv_batchnorm(net);

	Darknet::Layer & l = net.layers[net.n - 1];
	int j;

	detector_gpu.avg = (float *)calloc(l.outputs, sizeof(float));
	for (j = 0; j < NFRAMES; ++j) detector_gpu.predictions[j] = (float*)calloc(l.outputs, sizeof(float));
	for (j = 0; j < NFRAMES; ++j) detector_gpu.images[j] = make_image(1, 1, 3);

	detector_gpu.track_id = (unsigned int *)calloc(l.classes, sizeof(unsigned int));
	for (j = 0; j < l.classes; ++j) detector_gpu.track_id[j] = 1;

#ifdef DARKNET_GPU
	check_cuda( cudaSetDevice(old_gpu_index) );
#endif
}


Detector::~Detector()
{
	TAT(TATPARMS);

	detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	//layer l = detector_gpu.net.layers[detector_gpu.net.n - 1];

	free(detector_gpu.track_id);

	free(detector_gpu.avg);
	for (int j = 0; j < NFRAMES; ++j) free(detector_gpu.predictions[j]);
	for (int j = 0; j < NFRAMES; ++j) if (detector_gpu.images[j].data) free(detector_gpu.images[j].data);

#ifdef DARKNET_GPU
	int old_gpu_index;
	CHECK_CUDA(cudaGetDevice(&old_gpu_index));
	cuda_set_device(detector_gpu.net.gpu_index);
#endif

	free_network(detector_gpu.net);

#ifdef DARKNET_GPU
	CHECK_CUDA(cudaSetDevice(old_gpu_index));
#endif
}

int Detector::get_net_width() const
{
	TAT(TATPARMS);

	detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	return detector_gpu.net.w;
}

int Detector::get_net_height() const
{
	TAT(TATPARMS);

	detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	return detector_gpu.net.h;
}

int Detector::get_net_color_depth() const
{
	TAT(TATPARMS);

	detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	return detector_gpu.net.c;
}

std::vector<bbox_t> Detector::detect(std::string image_filename, float thresh, bool use_mean)
{
	TAT(TATPARMS);

	std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { if (img->data) free(img->data); delete img; });
	*image_ptr = load_image(image_filename);
	return detect(*image_ptr, thresh, use_mean);
}

image_t Detector::load_image(std::string image_filename)
{
	TAT(TATPARMS);

	Darknet::Image im = Darknet::load_image(const_cast<char*>(image_filename.c_str()), 0, 0, 3);

	image_t img;
	img.c = im.c;
	img.data = im.data;
	img.h = im.h;
	img.w = im.w;

	return img;
}


void Detector::free_image(image_t m)
{
	TAT(TATPARMS);

	if (m.data)
	{
		free(m.data);
	}
}

std::vector<bbox_t> Detector::detect(image_t img, float thresh, bool use_mean)
{
	TAT(TATPARMS);

	detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	Darknet::Network & net = detector_gpu.net;
#ifdef DARKNET_GPU
	int old_gpu_index;
	CHECK_CUDA(cudaGetDevice(&old_gpu_index));
	if (cur_gpu_id != old_gpu_index)
	{
		CHECK_CUDA(cudaSetDevice(net.gpu_index));
	}

	net.wait_stream = wait_stream;    // 1 - wait CUDA-stream, 0 - not to wait
#endif
	//std::cout << "net.gpu_index = " << net.gpu_index << std::endl;

	Darknet::Image im;
	im.c = img.c;
	im.data = img.data;
	im.h = img.h;
	im.w = img.w;

	Darknet::Image sized;

	if (net.w == im.w && net.h == im.h)
	{
		sized = make_image(im.w, im.h, im.c);
		memcpy(sized.data, im.data, im.w*im.h*im.c * sizeof(float));
	}
	else
	{
		sized = Darknet::resize_image(im, net.w, net.h);
	}

	Darknet::Layer & l = net.layers[net.n - 1];

	float *X = sized.data;

	float *prediction = network_predict(net, X);

	if (use_mean) {
		memcpy(detector_gpu.predictions[detector_gpu.demo_index], prediction, l.outputs * sizeof(float));
		mean_arrays(detector_gpu.predictions, NFRAMES, l.outputs, detector_gpu.avg);
		l.output = detector_gpu.avg;
		detector_gpu.demo_index = (detector_gpu.demo_index + 1) % NFRAMES;
	}
	//get_region_boxes(l, 1, 1, thresh, detector_gpu.probs, detector_gpu.boxes, 0, 0);
	//if (nms) do_nms_sort(detector_gpu.boxes, detector_gpu.probs, l.w*l.h*l.n, l.classes, nms);

	int nboxes = 0;
	int letterbox = 0;
	float hier_thresh = 0.5;
	Darknet::Detection * dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

	std::vector<bbox_t> bbox_vec;

	for (int i = 0; i < nboxes; ++i) {
		Darknet::Box b = dets[i].bbox;
		int const obj_id = max_index(dets[i].prob, l.classes);
		float const prob = dets[i].prob[obj_id];

		if (prob > thresh)
		{
			bbox_t bbox;
			bbox.x = std::max((double)0, (b.x - b.w / 2.)*im.w);
			bbox.y = std::max((double)0, (b.y - b.h / 2.)*im.h);
			bbox.w = b.w*im.w;
			bbox.h = b.h*im.h;
			bbox.obj_id = obj_id;
			bbox.prob = prob;
			bbox.track_id = 0;
			bbox.frames_counter = 0;
			bbox.x_3d = NAN;
			bbox.y_3d = NAN;
			bbox.z_3d = NAN;

			bbox_vec.push_back(bbox);
		}
	}

	free_detections(dets, nboxes);
	if(sized.data)
		free(sized.data);

#ifdef DARKNET_GPU
	if (cur_gpu_id != old_gpu_index)
	{
		CHECK_CUDA(cudaSetDevice(old_gpu_index));
	}
#endif

	return bbox_vec;
}

std::vector<std::vector<bbox_t>> Detector::detectBatch(image_t img, int batch_size, int width, int height, float thresh, bool make_nms)
{
	TAT(TATPARMS);

	detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	Darknet::Network net = detector_gpu.net;
#ifdef DARKNET_GPU
	int old_gpu_index;
	CHECK_CUDA(cudaGetDevice(&old_gpu_index));
	if (cur_gpu_id != old_gpu_index)
	{
		CHECK_CUDA(cudaSetDevice(net.gpu_index));
	}

	net.wait_stream = wait_stream;    // 1 - wait CUDA-stream, 0 - not to wait
#endif
	//std::cout << "net.gpu_index = " << net.gpu_index << std::endl;

	Darknet::Layer & l = net.layers[net.n - 1];

	float hier_thresh = 0.5;
	Darknet::Image in_img;
	in_img.c = img.c;
	in_img.w = img.w;
	in_img.h = img.h;
	in_img.data = img.data;
	det_num_pair* prediction = network_predict_batch(&net, in_img, batch_size, width, height, thresh, hier_thresh, 0, 0, 0);

	std::vector<std::vector<bbox_t>> bbox_vec(batch_size);

	for (int bi = 0; bi < batch_size; ++bi)
	{
		auto dets = prediction[bi].dets;

		if (make_nms && nms)
			do_nms_sort(dets, prediction[bi].num, l.classes, nms);

		for (int i = 0; i < prediction[bi].num; ++i)
		{
			Darknet::Box b = dets[i].bbox;
			int const obj_id = max_index(dets[i].prob, l.classes);
			float const prob = dets[i].prob[obj_id];

			if (prob > thresh)
			{
				bbox_t bbox;
				bbox.x = std::max((double)0, (b.x - b.w / 2.));
				bbox.y = std::max((double)0, (b.y - b.h / 2.));
				bbox.w = b.w;
				bbox.h = b.h;
				bbox.obj_id = obj_id;
				bbox.prob = prob;
				bbox.track_id = 0;
				bbox.frames_counter = 0;
				bbox.x_3d = NAN;
				bbox.y_3d = NAN;
				bbox.z_3d = NAN;

				bbox_vec[bi].push_back(bbox);
			}
		}
	}
	free_batch_detections(prediction, batch_size);

#ifdef DARKNET_GPU
	if (cur_gpu_id != old_gpu_index)
	{
		CHECK_CUDA(cudaSetDevice(old_gpu_index));
	}
#endif

	return bbox_vec;
}

std::vector<bbox_t> Detector::tracking_id(std::vector<bbox_t> cur_bbox_vec, bool const change_history,
	int const frames_story, int const max_dist)
{
	TAT(TATPARMS);

	detector_gpu_t &det_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());

	bool prev_track_id_present = false;
	for (auto &i : prev_bbox_vec_deque)
		if (i.size() > 0) prev_track_id_present = true;

	if (!prev_track_id_present) {
		for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
			cur_bbox_vec[i].track_id = det_gpu.track_id[cur_bbox_vec[i].obj_id]++;
		prev_bbox_vec_deque.push_front(cur_bbox_vec);
		if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();
		return cur_bbox_vec;
	}

	std::vector<unsigned int> dist_vec(cur_bbox_vec.size(), std::numeric_limits<unsigned int>::max());

	for (auto &prev_bbox_vec : prev_bbox_vec_deque) {
		for (auto &i : prev_bbox_vec) {
			int cur_index = -1;
			for (size_t m = 0; m < cur_bbox_vec.size(); ++m) {
				bbox_t const& k = cur_bbox_vec[m];
				if (i.obj_id == k.obj_id) {
					float center_x_diff = (float)(i.x + i.w/2) - (float)(k.x + k.w/2);
					float center_y_diff = (float)(i.y + i.h/2) - (float)(k.y + k.h/2);
					unsigned int cur_dist = sqrt(center_x_diff*center_x_diff + center_y_diff*center_y_diff);
					if (cur_dist < max_dist && (k.track_id == 0 || dist_vec[m] > cur_dist)) {
						dist_vec[m] = cur_dist;
						cur_index = m;
					}
				}
			}

			bool track_id_absent = !std::any_of(cur_bbox_vec.begin(), cur_bbox_vec.end(),
				[&i](bbox_t const& b) { return b.track_id == i.track_id && b.obj_id == i.obj_id; });

			if (cur_index >= 0 && track_id_absent){
				cur_bbox_vec[cur_index].track_id = i.track_id;
				cur_bbox_vec[cur_index].w = (cur_bbox_vec[cur_index].w + i.w) / 2;
				cur_bbox_vec[cur_index].h = (cur_bbox_vec[cur_index].h + i.h) / 2;
			}
		}
	}

	for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
		if (cur_bbox_vec[i].track_id == 0)
			cur_bbox_vec[i].track_id = det_gpu.track_id[cur_bbox_vec[i].obj_id]++;

	if (change_history) {
		prev_bbox_vec_deque.push_front(cur_bbox_vec);
		if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();
	}

	return cur_bbox_vec;
}


void *Detector::get_cuda_context()
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	int old_gpu_index;
	CHECK_CUDA(cudaGetDevice(&old_gpu_index));
	if (cur_gpu_id != old_gpu_index)
	{
		CHECK_CUDA(cudaSetDevice(cur_gpu_id));
	}

	void *cuda_context = cuda_get_context();

	if (cur_gpu_id != old_gpu_index)
	{
		CHECK_CUDA(cudaSetDevice(old_gpu_index));
	}

	return cuda_context;
#else   // DARKNET_GPU
	return NULL;
#endif  // DARKNET_GPU
}
