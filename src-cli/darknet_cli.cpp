#include <csignal>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstring>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


extern void run_detector(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);


void average(int argc, char *argv[])
{
	TAT(TATPARMS);

	char *cfgfile = argv[2];
	char *outfile = argv[3];
	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);
	Darknet::Network sum = parse_network_cfg(cfgfile);

	char *weightfile = argv[4];
	load_weights(&sum, weightfile);

	int i, j;
	int n = argc - 5;
	for (i = 0; i < n; ++i)
	{
		weightfile = argv[i+5];
		load_weights(&net, weightfile);
		for (j = 0; j < net.n; ++j)
		{
			Darknet::Layer /*&*/ l = net.layers[j];
			Darknet::Layer /*&*/ out = sum.layers[j];
			if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
			{
				int num = l.n*l.c*l.size*l.size;
				axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
				if (l.batch_normalize)
				{
					axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
					axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
					axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
				}
			}
			if (l.type == Darknet::ELayerType::CONNECTED)
			{
				axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
			}
		}
	}

	n = n+1;
	for (j = 0; j < net.n; ++j)
	{
		Darknet::Layer /*&*/ l = sum.layers[j];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			int num = l.n*l.c*l.size*l.size;
			scal_cpu(l.n, 1./n, l.biases, 1);
			scal_cpu(num, 1./n, l.weights, 1);
			if (l.batch_normalize){
				scal_cpu(l.n, 1./n, l.scales, 1);
				scal_cpu(l.n, 1./n, l.rolling_mean, 1);
				scal_cpu(l.n, 1./n, l.rolling_variance, 1);
			}
		}
		if (l.type == Darknet::ELayerType::CONNECTED)
		{
			scal_cpu(l.outputs, 1./n, l.biases, 1);
			scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
		}
	}

	save_weights(sum, outfile);
}


void speed(const char * cfgfile, int tics)
{
	TAT(TATPARMS);

	if (tics <= 0)
	{
		tics = 1000;
	}

	Darknet::Network net = parse_network_cfg(cfgfile);
	set_batch_network(&net, 1);
	int i;
	Darknet::Image im = make_image(net.w, net.h, net.c);
	
	// BDP detection counting variables
	int total_bdp_detections = 0;
	float detection_threshold = 0.35f; // Threshold for counting meaningful detections
	
	time_t start = time(0);
	for (i = 0; i < tics; ++i)
	{
		network_predict(net, im.data);
		
		// Count BDP detections for this evaluation
		int eval_bdp_detections = 0;
		for (int layer_idx = 0; layer_idx < net.n; ++layer_idx)
		{
			Darknet::Layer& layer = net.layers[layer_idx];
			if (layer.type == Darknet::ELayerType::YOLO_BDP)
			{
				// Count detections in this BDP layer using the 6-parameter format
				eval_bdp_detections += yolo_num_detections_bdp(layer, detection_threshold);
			}
		}
		total_bdp_detections += eval_bdp_detections;
	}
	double t = difftime(time(0), start);

	// Calculate average BDP detections per evaluation
	double avg_bdp_detections = static_cast<double>(total_bdp_detections) / tics;

	*cfg_and_state.output							<< std::endl
		<< tics << " evals, " << t << " seconds"	<< std::endl
		<< "avg BDP detections per eval: " << avg_bdp_detections << std::endl
		<< "Speed: " << t/tics << " sec/eval"		<< std::endl
		<< "Speed: " << tics/t << " Hz"				<< std::endl;
}

void draw_line(Darknet::Image& im, int x0, int y0, int x1, int y1, int r, int g, int b)
{
	// Simple Bresenham's line algorithm
	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);
	int sx = (x0 < x1) ? 1 : -1;
	int sy = (y0 < y1) ? 1 : -1;
	int err = dx - dy;
	
	int x = x0, y = y0;
	
	while (true) {
		// Set pixel if within image bounds
		if (x >= 0 && x < im.w && y >= 0 && y < im.h) {
			// Darknet image format: [R][G][B] non-interlaced
			int idx = y * im.w + x;
			im.data[idx] = r / 255.0f;                    // Red channel
			im.data[im.w * im.h + idx] = g / 255.0f;      // Green channel  
			im.data[2 * im.w * im.h + idx] = b / 255.0f;  // Blue channel
		}
		
		if (x == x1 && y == y1) break;
		
		int e2 = 2 * err;
		if (e2 > -dy) {
			err -= dy;
			x += sx;
		}
		if (e2 < dx) {
			err += dx;
			y += sy;
		}
	}
}

void draw_oriented_box(Darknet::Image& im, DarknetBoxBDP box, int r, int g, int b)
{
	// Convert coordinates to pixel space
	float cx = box.x * im.w;   // Center X
	float cy = box.y * im.h;   // Center Y
	float fx = box.fx * im.w;  // Front point X
	float fy = box.fy * im.h;  // Front point Y
	float w = box.w * im.w;    // Width
	float h = box.h * im.h;    // Height
	
	// The front point (fx, fy) should be one corner of the rectangle
	// Calculate the 4 corners where (fx, fy) is one of them
	
	// Vector from center to front point
	float front_dx = fx - cx;
	float front_dy = fy - cy;
	
	// Calculate the opposite corner (back point)
	float bx = cx - front_dx;  // Back point X (opposite to front)
	float by = cy - front_dy;  // Back point Y (opposite to front)
	
	// Calculate perpendicular vector for width
	float perp_dx = -front_dy;  // Perpendicular to front vector
	float perp_dy = front_dx;
	
	// Normalize perpendicular vector
	float perp_length = sqrt(perp_dx*perp_dx + perp_dy*perp_dy);
	if (perp_length > 0) {
		perp_dx /= perp_length;
		perp_dy /= perp_length;
	}
	
	// Calculate half-width offset
	float hw_offset_x = perp_dx * w * 0.5f;
	float hw_offset_y = perp_dy * w * 0.5f;
	
	// Calculate the 4 corners of the rectangle
	// Front edge corners
	int x1 = (int)(fx + hw_offset_x);  // Front-right corner
	int y1 = (int)(fy + hw_offset_y);
	int x2 = (int)(fx - hw_offset_x);  // Front-left corner  
	int y2 = (int)(fy - hw_offset_y);
	
	// Back edge corners
	int x3 = (int)(bx - hw_offset_x);  // Back-left corner
	int y3 = (int)(by - hw_offset_y);
	int x4 = (int)(bx + hw_offset_x);  // Back-right corner
	int y4 = (int)(by + hw_offset_y);
	
	// Draw the 4 sides of the rectangle
	draw_line(im, x1, y1, x2, y2, r, g, b);  // Front edge
	draw_line(im, x2, y2, x3, y3, r, g, b);  // Left edge
	draw_line(im, x3, y3, x4, y4, r, g, b);  // Back edge
	draw_line(im, x4, y4, x1, y1, r, g, b);  // Right edge
	
	// Draw center point as a + (plus sign)
	int cross_size = 4;
	draw_line(im, (int)cx - cross_size, (int)cy, (int)cx + cross_size, (int)cy, r, g, b);
	draw_line(im, (int)cx, (int)cy - cross_size, (int)cx, (int)cy + cross_size, r, g, b);
	
	// Draw front point as a * (asterisk) - this should be ON the rectangle edge
	int star_size = 3;
	draw_line(im, (int)fx - star_size, (int)fy, (int)fx + star_size, (int)fy, r, g, b);
	draw_line(im, (int)fx, (int)fy - star_size, (int)fx, (int)fy + star_size, r, g, b);
	draw_line(im, (int)fx - star_size, (int)fy - star_size, (int)fx + star_size, (int)fy + star_size, r, g, b);
	draw_line(im, (int)fx - star_size, (int)fy + star_size, (int)fx + star_size, (int)fy - star_size, r, g, b);
}

void experiment(int argc, char** argv, const char* cfgfile, const char* imagepath)
{
	TAT(TATPARMS);

	// Check for --draw parameter
	bool draw_detections = find_arg(argc, argv, "--draw");

	*cfg_and_state.output << "Loading image: " << imagepath << std::endl;

	// Load network from config file
	Darknet::Network net = parse_network_cfg(cfgfile);
	set_batch_network(&net, 1);

	// Load image from file path
	Darknet::Image orig = Darknet::load_image(imagepath, 0, 0, net.c);
	*cfg_and_state.output << "Image size: " << orig.w << "x" << orig.h 
						  << ", Network size: " << net.w << "x" << net.h << std::endl;

	// Resize image to network dimensions
	Darknet::Image sized = Darknet::resize_image(orig, net.w, net.h);

	// BDP detection counting variables
	float detection_threshold = 0.85f; // Threshold for counting meaningful detections

	// Perform single inference with timing
	time_t start = time(0);
	network_predict(net, sized.data);

	// Count BDP detections and extract them if drawing is requested
	int total_bdp_detections = 0;
	std::vector<DarknetDetectionOBB> all_detections;
	
	for (int layer_idx = 0; layer_idx < net.n; ++layer_idx)
	{
		Darknet::Layer& layer = net.layers[layer_idx];
		if (layer.type == Darknet::ELayerType::YOLO_BDP)
		{
			// Count detections in this BDP layer using the 6-parameter format
			int layer_detections = yolo_num_detections_bdp(layer, detection_threshold);
			total_bdp_detections += layer_detections;
			
			// If drawing is requested, extract actual detections
			if (draw_detections && layer_detections > 0)
			{
				*cfg_and_state.output << "Layer " << layer_idx << ": Found " << layer_detections << " detections above threshold " << detection_threshold << std::endl;
				
				// Create detection array with smart pointers for safe memory management
				std::vector<DarknetDetectionOBB> layer_dets(layer_detections);
				std::vector<std::unique_ptr<float[]>> prob_arrays;
				prob_arrays.reserve(layer_detections);
				
				// Initialize each detection with proper memory allocation using smart pointers
				for (int i = 0; i < layer_detections; ++i)
				{
					std::memset(&layer_dets[i], 0, sizeof(DarknetDetectionOBB));
					layer_dets[i].classes = layer.classes;
					
					// Use smart pointer for automatic cleanup
					prob_arrays.emplace_back(std::make_unique<float[]>(layer.classes));
					layer_dets[i].prob = prob_arrays[i].get();
					
					// Initialize other pointers to null
					layer_dets[i].mask = nullptr;
					layer_dets[i].uc = nullptr;
					layer_dets[i].embeddings = nullptr;
					layer_dets[i].embedding_size = 0;
					layer_dets[i].points = 0;
				}
				
				int actual_count = get_yolo_detections_bdp(layer, orig.w, orig.h, net.w, net.h, 
														 detection_threshold, nullptr, 0, layer_dets.data(), 0);
				
				*cfg_and_state.output << "get_yolo_detections_bdp returned " << actual_count << " detections" << std::endl;
				
				// Add valid detections to all detections (the smart pointers will clean up automatically)
				for (int i = 0; i < actual_count && i < layer_detections; ++i)
				{
					// Debug: show detection info
					*cfg_and_state.output << "Detection " << i << ": objectness=" << layer_dets[i].objectness 
										  << " bbox=(" << layer_dets[i].bbox.x << "," << layer_dets[i].bbox.y 
										  << "," << layer_dets[i].bbox.w << "," << layer_dets[i].bbox.h 
										  << "," << layer_dets[i].bbox.fx << "," << layer_dets[i].bbox.fy << ")" << std::endl;
					
					// Copy the detection data (but not the pointer addresses)
					DarknetDetectionOBB det_copy = layer_dets[i];
					
					// Allocate new memory for the prob array in the copy
					std::unique_ptr<float[]> new_prob = std::make_unique<float[]>(layer.classes);
					std::memcpy(new_prob.get(), layer_dets[i].prob, layer.classes * sizeof(float));
					det_copy.prob = new_prob.release(); // Transfer ownership (will need manual cleanup)
					
					all_detections.push_back(det_copy);
				}
				
				// Smart pointers automatically clean up when going out of scope
			}
		}
	}
	double t = difftime(time(0), start);

	// Display results
	*cfg_and_state.output << "BDP detections found: " << total_bdp_detections << std::endl
						  << "Inference time: " << t << " seconds" << std::endl;

	// Debug: Show what we collected for drawing
	if (draw_detections)
	{
		*cfg_and_state.output << "Drawing requested. Collected " << all_detections.size() << " detections for visualization." << std::endl;
		if (all_detections.empty() && total_bdp_detections > 0)
		{
			*cfg_and_state.output << "Warning: " << total_bdp_detections << " detections counted but none extracted for drawing." << std::endl;
		}
	}

	// Add hardcoded BDP examples for testing when drawing is requested
	if (draw_detections)
	{
		*cfg_and_state.output << "Adding 5 hardcoded BDP examples for testing..." << std::endl;
		
		// Clear any existing detections to use only hardcoded ones
		for (auto& det : all_detections)
		{
			if (det.prob) delete[] det.prob;
		}
		all_detections.clear();
		
		// Create 5 example BDP detections with different orientations
		for (int i = 0; i < 5; ++i)
		{
			DarknetDetectionOBB example_det;
			std::memset(&example_det, 0, sizeof(DarknetDetectionOBB));
			
			// Allocate prob array
			example_det.classes = 80; // COCO classes
			example_det.prob = new float[80]();
			
			// Set a high objectness score
			example_det.objectness = 0.9f - (i * 0.1f); // 0.9, 0.8, 0.7, 0.6, 0.5
			
			// Create different oriented bounding boxes across the image
			switch(i)
			{
				case 0: // Top-left, horizontal orientation
					example_det.bbox.x = 0.25f;   // Center X (25% from left)
					example_det.bbox.y = 0.25f;   // Center Y (25% from top)  
					example_det.bbox.w = 0.2f;    // Width (20% of image)
					example_det.bbox.h = 0.1f;    // Height (10% of image)
					example_det.bbox.fx = 0.25f;  // Front point X (right of center)
					example_det.bbox.fy = 0.15f;  // Front point Y (same as center - horizontal)
					break;
					
				case 1: // Top-right, vertical orientation  
					example_det.bbox.x = 0.75f;   // Center X (75% from left)
					example_det.bbox.y = 0.25f;   // Center Y (25% from top)
					example_det.bbox.w = 0.1f;    // Width (10% of image)
					example_det.bbox.h = 0.2f;    // Height (20% of image)
					example_det.bbox.fx = 0.85f;  // Front point X (same as center)
					example_det.bbox.fy = 0.35f;  // Front point Y (above center - vertical up)
					break;
					
				case 2: // Center, diagonal orientation
					example_det.bbox.x = 0.5f;    // Center X (50% from left)
					example_det.bbox.y = 0.5f;    // Center Y (50% from top)
					example_det.bbox.w = 0.15f;   // Width (15% of image)
					example_det.bbox.h = 0.15f;   // Height (15% of image)
					example_det.bbox.fx = 0.6f;   // Front point X (diagonal right)
					example_det.bbox.fy = 0.4f;   // Front point Y (diagonal up)
					break;
					
				case 3: // Bottom-left, angled orientation
					example_det.bbox.x = 0.25f;   // Center X (25% from left)
					example_det.bbox.y = 0.75f;   // Center Y (75% from top)
					example_det.bbox.w = 0.18f;   // Width (18% of image)
					example_det.bbox.h = 0.12f;   // Height (12% of image)
					example_det.bbox.fx = 0.30f;  // Front point X (slight right)
					example_det.bbox.fy = 0.60f;  // Front point Y (angled up-right)
					break;
					
				case 4: // Bottom-right, steep angle
					example_det.bbox.x = 0.75f;   // Center X (75% from left)
					example_det.bbox.y = 0.75f;   // Center Y (75% from top)
					example_det.bbox.w = 0.12f;   // Width (12% of image)
					example_det.bbox.h = 0.18f;   // Height (18% of image)
					example_det.bbox.fx = 0.60f;  // Front point X (left of center)
					example_det.bbox.fy = 0.60f;  // Front point Y (below center - steep angle)
					break;
			}
			
			// Set best class (just use class 0 for simplicity)
			example_det.best_class_idx = 0;
			example_det.prob[0] = example_det.objectness; // Set class probability
			
			all_detections.push_back(example_det);
			
			*cfg_and_state.output << "Added example detection " << (i+1) << ": objectness=" << example_det.objectness 
								  << " bbox=(" << example_det.bbox.x << "," << example_det.bbox.y 
								  << "," << example_det.bbox.w << "," << example_det.bbox.h 
								  << "," << example_det.bbox.fx << "," << example_det.bbox.fy << ")" << std::endl;
		}
	}

	// If drawing is requested and we have detections (real or hardcoded)
	if (draw_detections && !all_detections.empty())
	{
		// Sort detections by objectness score (highest first)
		std::sort(all_detections.begin(), all_detections.end(), 
				  [](const DarknetDetectionOBB& a, const DarknetDetectionOBB& b) {
					  return a.objectness > b.objectness;
				  });

		// Take top 5 detections
		int num_to_draw = std::min(5, (int)all_detections.size());
		*cfg_and_state.output << "Drawing top " << num_to_draw << " BDP detections on image..." << std::endl;

		// Create copy of original image for annotation
		Darknet::Image annotated = Darknet::copy_image(orig);

		// Colors for the boxes (RGB)
		int colors[][3] = {{255,0,0}, {0,255,0}, {0,0,255}, {255,255,0}, {255,0,255}};

		// Draw top detections
		for (int i = 0; i < num_to_draw; ++i)
		{
			DarknetDetectionOBB& det = all_detections[i];
			*cfg_and_state.output << "Detection " << (i+1) << ": objectness=" << det.objectness << std::endl;
			
			draw_oriented_box(annotated, det.bbox, colors[i][0], colors[i][1], colors[i][2]);
		}

		// Save annotated image
		const char* output_filename = "image_bdp_annot.png";
		Darknet::save_image_png(annotated, output_filename);
		*cfg_and_state.output << "Annotated image saved as: " << output_filename << std::endl;

		// Clean up annotated image
		Darknet::free_image(annotated);
	}

	// Clean up manually allocated prob arrays in all_detections
	for (auto& det : all_detections)
	{
		if (det.prob)
		{
			delete[] det.prob;
		}
	}

	// Clean up memory
	Darknet::free_image(orig);
	Darknet::free_image(sized);
}


void operations(char *cfgfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);
	long ops = 0;
	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			ops += 2l * l.n * l.size*l.size*l.c * l.out_h*l.out_w;
		}
		else if (l.type == Darknet::ELayerType::CONNECTED)
		{
			ops += 2l * l.inputs * l.outputs;
		}
		else if (l.type == Darknet::ELayerType::RNN)
		{
			ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
			ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
			ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
		}
		else if (l.type == Darknet::ELayerType::LSTM)
		{
			ops += 2l * l.uf->inputs * l.uf->outputs;
			ops += 2l * l.ui->inputs * l.ui->outputs;
			ops += 2l * l.ug->inputs * l.ug->outputs;
			ops += 2l * l.uo->inputs * l.uo->outputs;
			ops += 2l * l.wf->inputs * l.wf->outputs;
			ops += 2l * l.wi->inputs * l.wi->outputs;
			ops += 2l * l.wg->inputs * l.wg->outputs;
			ops += 2l * l.wo->inputs * l.wo->outputs;
		}
	}

	*cfg_and_state.output
		<< "Floating point operations: " << ops << std::endl
		<< "Floating point operations: " << ops/1000000000.0f << " Bn" << std::endl;

	free_network(net);
}


void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);
	int oldn = net.layers[net.n - 2].n;
	int c = net.layers[net.n - 2].c;
	net.layers[net.n - 2].n = 9372;
	net.layers[net.n - 2].biases += 5;
	net.layers[net.n - 2].weights += 5*c;

	if(weightfile)
	{
		load_weights(&net, weightfile);
	}

	net.layers[net.n - 2].biases -= 5;
	net.layers[net.n - 2].weights -= 5*c;
	net.layers[net.n - 2].n = oldn;
	*cfg_and_state.output << oldn << std::endl;

	Darknet::Layer /*&*/ l = net.layers[net.n - 2];
	copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
	copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
	copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
	copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
	*net.seen = 0;
	*net.cur_iteration = 0;
	save_weights(net, outfile);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg_custom(cfgfile, 1, 1);

	if(weightfile)
	{
		load_weights_upto(&net, weightfile, max);
	}

	*net.seen = 0;
	*net.cur_iteration = 0;

	save_weights_upto(net, outfile, max, 0);
}


void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);

	if(weightfile)
	{
		load_weights(&net, weightfile);
	}

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer /*&*/ l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			rescale_weights(l, 2, -.5);
			break;
		}
	}

	save_weights(net, outfile);
}


void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);

	if (weightfile)
	{
		load_weights(&net, weightfile);
	}

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer /*&*/ l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			// swap red and blue channels?
			rgbgr_weights(l);
			break;
		}
	}

	save_weights(net, outfile);
}


void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);

	if (weightfile)
	{
		load_weights(&net, weightfile);
	}

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer /*&*/ l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL && l.batch_normalize)
		{
			denormalize_convolutional_layer(l);
		}
		if (l.type == Darknet::ELayerType::CONNECTED && l.batch_normalize)
		{
			denormalize_connected_layer(l);
		}
		if (l.type == Darknet::ELayerType::LSTM && l.batch_normalize)
		{
			denormalize_connected_layer(*l.wf);
			denormalize_connected_layer(*l.wi);
			denormalize_connected_layer(*l.wg);
			denormalize_connected_layer(*l.wo);
			denormalize_connected_layer(*l.uf);
			denormalize_connected_layer(*l.ui);
			denormalize_connected_layer(*l.ug);
			denormalize_connected_layer(*l.uo);
		}
	}
	save_weights(net, outfile);
}

Darknet::Layer /*&*/ normalize_layer(Darknet::Layer /*&*/ l, int n)
{
	TAT(TATPARMS);

	int j;
	l.batch_normalize=1;
	l.scales = (float*)xcalloc(n, sizeof(float));
	for(j = 0; j < n; ++j)
	{
		l.scales[j] = 1;
	}
	l.rolling_mean = (float*)xcalloc(n, sizeof(float));
	l.rolling_variance = (float*)xcalloc(n, sizeof(float));
	return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);

	if(weightfile)
	{
		load_weights(&net, weightfile);
	}

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer /*&*/ l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL && !l.batch_normalize)
		{
			net.layers[i] = normalize_layer(l, l.n);
		}
		if (l.type == Darknet::ELayerType::CONNECTED && !l.batch_normalize)
		{
			net.layers[i] = normalize_layer(l, l.outputs);
		}
		if (l.type == Darknet::ELayerType::LSTM && l.batch_normalize)
		{
			*l.wf = normalize_layer(*l.wf, l.wf->outputs);
			*l.wi = normalize_layer(*l.wi, l.wi->outputs);
			*l.wg = normalize_layer(*l.wg, l.wg->outputs);
			*l.wo = normalize_layer(*l.wo, l.wo->outputs);
			*l.uf = normalize_layer(*l.uf, l.uf->outputs);
			*l.ui = normalize_layer(*l.ui, l.ui->outputs);
			*l.ug = normalize_layer(*l.ug, l.ug->outputs);
			*l.uo = normalize_layer(*l.uo, l.uo->outputs);
			net.layers[i].batch_normalize=1;
		}
	}

	save_weights(net, outfile);
}

void statistics_net(const char * cfgfile, const char * weightfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);

	if (weightfile)
	{
		load_weights(&net, weightfile);
	}

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer /*&*/ l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONNECTED && l.batch_normalize)
		{
			*cfg_and_state.output << "Connected Layer " << i << std::endl;
			statistics_connected_layer(l);
		}
		if (l.type == Darknet::ELayerType::LSTM && l.batch_normalize)
		{
			*cfg_and_state.output << "LSTM Layer " << i << std::endl;

			*cfg_and_state.output << "wf" << std::endl;
			statistics_connected_layer(*l.wf);

			*cfg_and_state.output << "wi" << std::endl;
			statistics_connected_layer(*l.wi);

			*cfg_and_state.output << "wg" << std::endl;
			statistics_connected_layer(*l.wg);

			*cfg_and_state.output << "wo" << std::endl;
			statistics_connected_layer(*l.wo);

			*cfg_and_state.output << "uf" << std::endl;
			statistics_connected_layer(*l.uf);

			*cfg_and_state.output << "ui" << std::endl;
			statistics_connected_layer(*l.ui);

			*cfg_and_state.output << "ug" << std::endl;
			statistics_connected_layer(*l.ug);

			*cfg_and_state.output << "uo" << std::endl;
			statistics_connected_layer(*l.uo);
		}
		*cfg_and_state.output << std::endl;
	}
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	TAT(TATPARMS);

	Darknet::CfgAndState::get().gpu_index = -1;
	Darknet::Network net = parse_network_cfg(cfgfile);
	if (weightfile)
	{
		load_weights(&net, weightfile);
	}

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer /*&*/ l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL && l.batch_normalize)
		{
			denormalize_convolutional_layer(l);
			net.layers[i].batch_normalize=0;
		}
		if (l.type == Darknet::ELayerType::CONNECTED && l.batch_normalize)
		{
			denormalize_connected_layer(l);
			net.layers[i].batch_normalize=0;
		}

		/// @todo V3: I'm willing to bet this is supposed to be LSTM, not GRU...?
		if (l.type == Darknet::ELayerType::LSTM && l.batch_normalize)
		{
			denormalize_connected_layer(*l.wf);
			denormalize_connected_layer(*l.wi);
			denormalize_connected_layer(*l.wg);
			denormalize_connected_layer(*l.wo);
			denormalize_connected_layer(*l.uf);
			denormalize_connected_layer(*l.ui);
			denormalize_connected_layer(*l.ug);
			denormalize_connected_layer(*l.uo);
			l.wf->batch_normalize = 0;
			l.wi->batch_normalize = 0;
			l.wg->batch_normalize = 0;
			l.wo->batch_normalize = 0;
			l.uf->batch_normalize = 0;
			l.ui->batch_normalize = 0;
			l.ug->batch_normalize = 0;
			l.uo->batch_normalize = 0;
			net.layers[i].batch_normalize=0;
		}
	}
	save_weights(net, outfile);
}

void visualize(const char * cfgfile, const char * weightfile)
{
	TAT(TATPARMS);

	Darknet::Network net = parse_network_cfg(cfgfile);
	load_weights(&net, weightfile);

	visualize_network(net);
	cv::waitKey(0);
}


void darknet_signal_handler(int sig)
{
//	TAT(TATPARMS); ... don't bother, we're about to abort

	// prevent recursion if this signal happens again (set the default signal action)
	std::signal(sig, SIG_DFL);

	*cfg_and_state.output << "calling Darknet's fatal error handler due to signal #" << sig << std::endl;

	#ifdef WIN32
	darknet_fatal_error(DARKNET_LOC, "signal handler invoked for signal #%d", sig);
	#else
	darknet_fatal_error(DARKNET_LOC, "signal handler invoked for signal #%d (%s)", sig, strsignal(sig));
	#endif
}


int main(int argc, char **argv)
{
	// on purpose move TAT into the try/catch block so it records the value once the try block ends

	try
	{
		TAT(TATPARMS);

		signal(SIGINT   , darknet_signal_handler);  // 2: CTRL+C
		signal(SIGILL   , darknet_signal_handler);  // 4: illegal instruction
		signal(SIGABRT  , darknet_signal_handler);  // 6: abort()
		signal(SIGFPE   , darknet_signal_handler);  // 8: floating point exception
		signal(SIGSEGV  , darknet_signal_handler);  // 11: segfault
		signal(SIGTERM  , darknet_signal_handler);  // 15: terminate
#ifdef WIN32
		signal(SIGBREAK , darknet_signal_handler);  // Break is different than CTRL+C on Windows
#else
		signal(SIGHUP   , darknet_signal_handler);  // 1: hangup
		signal(SIGQUIT  , darknet_signal_handler);  // 3: quit
		signal(SIGUSR1  , darknet_signal_handler);  // 10: user-defined
		signal(SIGUSR2  , darknet_signal_handler);  // 12: user-defined
#endif

		// process the args before printing anything so we can handle "-colour" and "-nocolour" correctly
		auto & cfg_and_state = Darknet::CfgAndState::get();
		cfg_and_state.set_thread_name("main darknet thread");
		cfg_and_state.process_arguments(argc, argv);

		#ifdef _DEBUG
		_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
		Darknet::display_warning_msg("DEBUG is used\n");
		#endif

		#ifdef DEBUG
		Darknet::display_warning_msg("DEBUG=1 is enabled\n");
		#endif

		errno = 0;

		cfg_and_state.gpu_index = find_int_arg(argc, argv, "-i", 0);

#ifndef DARKNET_GPU
		cfg_and_state.gpu_index = -1;
		init_cpu();
#else
		if (cfg_and_state.gpu_index >= 0)
		{
			cuda_set_device(cfg_and_state.gpu_index);
			CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
		}
		cuda_debug_sync = find_arg(argc, argv, "-cuda_debug_sync");
#endif

		errno = 0;

		/// @todo V3 look through these and see what we no longer need
		if		(cfg_and_state.command.empty())				{ Darknet::display_usage();			}

		/// @todo V3 "3d" seems to combine 2 images into a single alpha-blended composite.  It works...but does it belong in Darknet?  What is this for?
		else if (cfg_and_state.command == "3d")				{ Darknet::composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0); }
		else if (cfg_and_state.command == "average")		{ average			(argc, argv);	}
		else if (cfg_and_state.command == "cfglayers")		{ Darknet::cfg_layers();			}
		else if (cfg_and_state.command == "denormalize")	{ denormalize_net	(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "detector")		{ run_detector		(argc, argv);	}
		else if (cfg_and_state.command == "experiment")	{ experiment		(argc, argv, cfg_and_state.cfg_filename.string().c_str(), argv[3]); }
		else if (cfg_and_state.command == "help")			{ Darknet::display_usage();			}
		else if (cfg_and_state.command == "nightmare")		{ run_nightmare		(argc, argv);	}
		else if (cfg_and_state.command == "normalize")		{ normalize_net		(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "oneoff")			{ oneoff			(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "ops")			{ operations		(argv[2]); }
		else if (cfg_and_state.command == "partial")		{ partial			(argv[2], argv[3], argv[4], atoi(argv[5])); }
		else if (cfg_and_state.command == "rescale")		{ rescale_net		(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "reset")			{ reset_normalize_net(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "rgbgr")			{ rgbgr_net			(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "speed")			{ speed				(cfg_and_state.cfg_filename.string().c_str(), 0); }
		else if (cfg_and_state.command == "statistics")		{ statistics_net	(cfg_and_state.cfg_filename.string().c_str(), cfg_and_state.weights_filename.string().c_str()); }
		else if (cfg_and_state.command == "test")			{ Darknet::test_resize(argv[2]);	} ///< @todo V3 what is this?
		else if (cfg_and_state.command == "imtest")			{ Darknet::test_resize(argv[2]);	} ///< @see "test"
		else if (cfg_and_state.command == "version")		{ /* nothing else to do, we've already displayed the version information */ }
		else if (cfg_and_state.command == "visualize")
		{
			if (cfg_and_state.cfg_filename.empty())
			{
				darknet_fatal_error(DARKNET_LOC, "must specify a .cfg file to load");
			}
			if (cfg_and_state.weights_filename.empty())
			{
				darknet_fatal_error(DARKNET_LOC, "must specify a .weights file to load");
			}
			visualize(
				cfg_and_state.cfg_filename		.string().c_str(),
				cfg_and_state.weights_filename	.string().c_str());
		}
		else if (cfg_and_state.command == "detect")
		{
			float thresh = find_float_arg(argc, argv, "-thresh", .24);
			int ext_output = find_arg(argc, argv, "-ext_output");
			char *filename = (argc > 4) ? argv[4]: 0;
			test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, 0.5, 0, ext_output, 0, NULL, 0, 0);
		}
		else
		{
			throw std::invalid_argument("invalid command (run \"" + cfg_and_state.argv[0] + " help\" for a list of possible commands)");
		}
	}
	catch (const std::exception & e)
	{
		*cfg_and_state.output << std::endl << "Exception: " << Darknet::in_colour(Darknet::EColour::kBrightRed, e.what()) << std::endl;
		darknet_fatal_error(DARKNET_LOC, e.what());
	}

	return 0;
}
