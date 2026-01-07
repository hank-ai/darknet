/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#include "darknet.hpp"

/** @file
 * This application will print to @p STDOUT details on each object detected in an image.
 * Output images are saved to disk.  Call it like this:
 *
 *     darknet_01_inference_images LegoGears DSCN1580_frame_000034.jpg
 *
 * The output should be similar to this:
 *
 *     predicting DSCN1580_frame_000034.jpg:
 *     prediction results: 5
 *     -> 1/5: #4 prob=0.999925 x=420 y=47 w=216 h=210 entries=1
 *     -> 2/5: #3 prob=0.999904 x=286 y=92 w=147 h=124 entries=1
 *     -> 3/5: #2 prob=0.991701 x=529 y=133 w=28 h=28 entries=1
 *     -> 4/5: #1 prob=0.998882 x=460 y=142 w=25 h=27 entries=1
 *     -> 5/5: #0 prob=0.995901 x=43 y=133 w=47 h=40 entries=1
 */


int main(int argc, char * argv[])
{
	try
	{
		Darknet::set_verbose(true);
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		int network_w = 0;
		int network_h = 0;
		int network_c = 0;
		Darknet::network_dimensions(net, network_w, network_h, network_c);
		const cv::Size network_dims(network_w, network_h);

		auto total_reading_from_disk	= std::chrono::nanoseconds(0);
		auto total_resize_images		= std::chrono::nanoseconds(0);
		auto total_darknet_predictions	= std::chrono::nanoseconds(0);
		auto total_darknet_annotate		= std::chrono::nanoseconds(0);
		auto total_save_output			= std::chrono::nanoseconds(0);
		auto total_all_time				= std::chrono::nanoseconds(0);

		size_t file_counter				= 0;
		size_t error_counter			= 0;

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
				const std::filesystem::path input_filename(parm.string);
				file_counter ++;

				std::cout << "loading " << input_filename << std::endl;
				const auto t1 = std::chrono::high_resolution_clock::now();
				cv::Mat mat = cv::imread(input_filename.string());
				const auto t2 = std::chrono::high_resolution_clock::now();
				if (mat.empty())
				{
					std::cout << "...invalid image?" << std::endl;
					error_counter ++;
					continue;
				}

				// Note that INTER_NEAREST gives us *speed*, not image quality.
				const auto t3 = std::chrono::high_resolution_clock::now();
				cv::Mat resized;
				// normally you'd skip this next line and go directly to predict()
				cv::resize(mat, resized, network_dims, cv::INTER_NEAREST);
				const auto t4 = std::chrono::high_resolution_clock::now();

				// output all of the predictions on the console as plain text
				const auto t5 = std::chrono::high_resolution_clock::now();
				/* This is a bad example.  If this was a "real" application, then we wouldn't resize the mat ourself and pass in
				 * the original image to predict().  That way Darknet would resize it, and would remember the original size.
				 * Otherwise, the results will be scaled to the resized mat, instead of the "full image".  The only reason we
				 * manually resized the image above was to get the timing details.  So we're going to ignore the resized mat and
				 * pass in the original mat...which means Darknet must resize it again.  Oops.  But I'd rather get "correct" images
				 * and slightly wrong timing details.
				 */
//				const auto results = Darknet::predict(net, resized);
				const auto results = Darknet::predict(net, mat);
				const auto t6 = std::chrono::high_resolution_clock::now();

				// save the annotated image to disk
				const auto t7 = std::chrono::high_resolution_clock::now();
				cv::Mat output = Darknet::annotate(net, results, mat);
				const auto t8 = std::chrono::high_resolution_clock::now();
				std::string output_filename = input_filename.stem().string() + "_output";

				const auto t9 = std::chrono::high_resolution_clock::now();
#if 1
				output_filename += ".jpg";
				const bool successful = cv::imwrite(output_filename, output, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 70});
#else
				output_filename += ".png";
				const bool successful = cv::imwrite(output_filename, output, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 5});
#endif
				const auto t10 = std::chrono::high_resolution_clock::now();

				if (not successful)
				{
					error_counter ++;
					std::cout << "failed to save the output to " << output_filename << std::endl;
				}

				const auto duration = t10 - t1;
				const int fps = std::round(1000000000.0f / std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());

				std::cout
					<< "-> reading image from disk ........... " << Darknet::format_duration_string(t2 - t1					, 3, Darknet::EFormatDuration::kPad) << " [" << mat.cols << " x " << mat.rows << " x " << mat.channels() << "] [" << Darknet::size_to_IEC_string(std::filesystem::file_size(input_filename)) << "]" << std::endl
					<< "-> resizing image to network dims .... " << Darknet::format_duration_string(t4 - t3					, 3, Darknet::EFormatDuration::kPad) << " [" << network_w << " x " << network_h << " x " << network_c << "]" << std::endl
					// don't count resizing the image twice (see comment above in regards to "mat" vs "resized")
					<< "-> using Darknet to predict .......... " << Darknet::format_duration_string((t6 - t5) - (t4 - t3)	, 3, Darknet::EFormatDuration::kPad) << " [" << results.size() << " object" << (results.size() == 1 ? "" : "s") << "]" << std::endl
					<< "-> using Darknet to annotate image ... " << Darknet::format_duration_string(t8 - t7					, 3, Darknet::EFormatDuration::kPad) << " [" << output.cols << " x " << output.rows << " x " << output.channels() << "]" << std::endl
					<< "-> save output image to disk ......... " << Darknet::format_duration_string(t10 - t9				, 3, Darknet::EFormatDuration::kPad) << " [" << Darknet::size_to_IEC_string(std::filesystem::file_size(output_filename)) << "]" << std::endl
					<< "-> total time elapsed ................ " << Darknet::format_duration_string(duration				, 3, Darknet::EFormatDuration::kPad) << " [" << fps << " FPS]" << std::endl
					<< results << std::endl << std::endl;

				total_reading_from_disk		+= (t2 - t1);
				total_resize_images			+= (t4 - t3);
				total_darknet_predictions	+= (t6 - t5);
				total_darknet_predictions	-= (t4 - t3); // remove the "resize twice" problem (see omment above in regards to "mat" vs "resized")
				total_darknet_annotate		+= (t8 - t7);
				total_save_output			+= (t10 - t9);
				total_all_time				+= duration;
			}
		}

		if (error_counter + file_counter > 1)
		{
			const float fps = static_cast<float>(file_counter) / std::chrono::duration_cast<std::chrono::nanoseconds>(total_all_time).count() * 1000000000.0f;

			std::cout
				<< "TOTALS:"													<< std::endl
				<< "-> number of images with errors ...... " << error_counter	<< std::endl
				<< "-> number of images processed ........ " << file_counter	<< std::endl
				<< "-> total time reading from disk ...... " << Darknet::format_duration_string(total_reading_from_disk		, 3, Darknet::EFormatDuration::kPad) << " [" << Darknet::format_duration_string(total_reading_from_disk		/ file_counter, 3) << " per image]" << std::endl
				<< "-> total time resizing images ........ " << Darknet::format_duration_string(total_resize_images			, 3, Darknet::EFormatDuration::kPad) << " [" << Darknet::format_duration_string(total_resize_images			/ file_counter, 3) << " per image]" << std::endl
				<< "-> total time predicting ............. " << Darknet::format_duration_string(total_darknet_predictions	, 3, Darknet::EFormatDuration::kPad) << " [" << Darknet::format_duration_string(total_darknet_predictions	/ file_counter, 3) << " per image]" << std::endl
				<< "-> total time annotating ............. " << Darknet::format_duration_string(total_darknet_annotate		, 3, Darknet::EFormatDuration::kPad) << " [" << Darknet::format_duration_string(total_darknet_annotate		/ file_counter, 3) << " per image]" << std::endl
				<< "-> total time saving to disk ......... " << Darknet::format_duration_string(total_save_output			, 3, Darknet::EFormatDuration::kPad) << " [" << Darknet::format_duration_string(total_save_output			/ file_counter, 3) << " per image]" << std::endl
				<< "-> total time processing images ...... " << Darknet::format_duration_string(total_all_time				, 3, Darknet::EFormatDuration::kPad) << " [" << Darknet::format_duration_string(total_all_time				/ file_counter, 3) << " per image, or " << std::setprecision(1) << fps << " FPS]" << std::endl
				<< std::endl;
		}

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
