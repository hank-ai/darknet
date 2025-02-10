/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include <fstream>
#include "darknet.hpp"
#include "json.hpp"
using JSON = nlohmann::json;

/** @file
 * This application will call predict() on an image or images and store the results in a JSON file.
 *
 *     darknet_06_images_to_json LegoGears DSCN1580_frame_000034.jpg
 *
 * The output should be similar to this:
 *
 */


std::string trim(std::string str)
{
	auto pos = str.find_last_not_of(" \t\r\n");
	if (pos != std::string::npos)
	{
		str.erase(pos + 1);
	}
	pos = str.find_first_not_of(" \t\r\n");
	if (pos != std::string::npos)
	{
		str.erase(0, pos);
	}

	return str;
}


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::show_version_info();
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		const std::filesystem::path json_path = "output.json";
		std::filesystem::remove(json_path);

		JSON json;
		json["file"] = JSON::array();
		size_t total_objects_detected = 0;

		const Darknet::VStr & names = Darknet::get_class_names(net);

		const auto start_time = std::chrono::high_resolution_clock::now();

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
				std::cout << "processing " << parm.string << ": ";

				// so we end up timing 1) loading from disk, 2) resize to network dimensions, and 3) predicting
				const auto t1 = std::chrono::high_resolution_clock::now();
				const auto results = Darknet::predict(net, parm.string);
				const auto t2 = std::chrono::high_resolution_clock::now();

				std::cout << results.size() << " object" << (results.size() == 1 ? "" : "s") << std::endl;
				// store the results in JSON format

				const size_t file_counter = json["file"].size();
				json["file"][file_counter]["filename"	] = parm.string;
				json["file"][file_counter]["count"		] = results.size();
				json["file"][file_counter]["duration"	] = trim(Darknet::format_duration_string(t2 - t1));

				total_objects_detected += results.size();

				for (const auto & pred : results)
				{
					const size_t idx = json["file"][file_counter]["predictions"].size();
					auto & j = json["file"][file_counter]["predictions"][idx];

					for (const auto & [k, v] : pred.prob)
					{
						const size_t count = j["all_probabilities"].size();
						j["all_probabilities"][count]["class"]			= k;
						j["all_probabilities"][count]["probability"]	= v;
						j["all_probabilities"][count]["name"]			= names.at(k);
					}

					j["best_class"]				= pred.best_class;
					j["best_probability"]		= pred.prob.at(pred.best_class);
					j["name"]					= names.at(pred.best_class) + " " + std::to_string(static_cast<int>(std::round(100.0f * pred.prob.at(pred.best_class)))) + "%";
					j["rect"]["x"]				= pred.rect.x;
					j["rect"]["y"]				= pred.rect.y;
					j["rect"]["width"]			= pred.rect.width;
					j["rect"]["height"]			= pred.rect.height;

					j["original_point"]["x"]	= pred.normalized_point.x;
					j["original_point"]["y"]	= pred.normalized_point.y;

					j["original_size"]["width"]	= pred.normalized_size.width;
					j["original_size"]["height"]= pred.normalized_size.height;
				}
			}
		}

		const auto end_time = std::chrono::high_resolution_clock::now();

		if (not json.empty())
		{
			std::ofstream ofs(json_path);
			ofs << std::setw(4) << json << std::endl;
			ofs.close();

			std::cout
				<< "-> JSON results ....... " << std::filesystem::canonical(json_path).string()	<< std::endl
				<< "-> images processed ... " << json["file"].size()							<< std::endl
				<< "-> objects detected ... " << total_objects_detected							<< std::endl
				<< "-> time elapsed ....... " << trim(Darknet::format_duration_string(end_time - start_time)) << std::endl;
		}

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
