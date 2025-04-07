/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2025 Stephane Charette
 */

#include <fstream>
#include "darknet.hpp"

/** @file
 * This application will call predict() on images and create YOLO-format annotations.  This can be used if you already
 * have weights for your neural network, and you need to quickly annotate images which you plan on then fixing up to
 * further train the neural network.
 *
 *     darknet_11_images_to_yolo LegoGears *.jpg
 */


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		const std::filesystem::path json_path = "output.json";
		std::filesystem::remove(json_path);

		size_t images_not_found			= 0;
		size_t yolo_already_exists		= 0;
		size_t images_processed			= 0;
		size_t negative_samples			= 0;
		size_t total_objects_detected	= 0;

		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

			const std::filesystem::path image_filename(parm.string);
			const std::filesystem::path yolo_filename = std::filesystem::path(image_filename).replace_extension(".txt");

			if (std::filesystem::exists(image_filename) == false)
			{
				images_not_found ++;
				std::cout << "ERROR:  image file does not exist: " << image_filename << std::endl;
				continue;
			}
			if (std::filesystem::exists(yolo_filename) == true)
			{
				yolo_already_exists ++;
				std::cout << "WARNING:  YOLO annotation file already exists: " << yolo_filename << std::endl;
				continue;
			}

			std::cout << "processing " << image_filename.string() << std::endl;
			const auto results = Darknet::predict(net, image_filename);
			total_objects_detected += results.size();
			images_processed ++;

			// create a new .txt YOLO annotation file and save the results to disk
			std::ofstream ofs(yolo_filename.string());
			if (results.empty())
			{
				negative_samples ++;
			}
			else
			{
				ofs << std::fixed << std::setprecision(10);
				for (const auto & pred : results)
				{
					ofs << pred.best_class << " " << pred.normalized_point.x << " " << pred.normalized_point.y << " " << pred.normalized_size.width << " " << pred.normalized_size.height << std::endl;
				}
			}
		}

		std::cout
			<< "-> missing images or bad paths .... " << images_not_found		<< std::endl
			<< "-> old YOLO annotations exist ..... " << yolo_already_exists	<< std::endl
			<< "-> new YOLO files saved to disk ... " << images_processed		<< std::endl
			<< "-> negative samples ............... " << negative_samples		<< std::endl
			<< "-> total objects detected ......... " << total_objects_detected	<< std::endl;

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
