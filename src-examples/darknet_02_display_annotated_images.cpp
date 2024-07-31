#include "darknet.hpp"


int main(int argc, char * argv[])
{
	try
	{
		std::cout << "Darknet v" << DARKNET_VERSION_SHORT << std::endl;

//		Darknet::set_annotation_font(cv::LineTypes::LINE_AA, cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1, 1.00);
		Darknet::set_annotation_bb_line_colour(cv::Scalar(0, 0, 255));
		Darknet::set_annotation_draw_rounded_bb(true, 0.5f);
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
				std::cout << "processing " << parm.string << std::endl;

				cv::Mat mat = cv::imread(parm.string);
				cv::imshow("original", mat);

				Darknet::predict_and_annotate(net, mat);
				cv::imshow("annotated", mat);

				const char c = cv::waitKey(-1);
				if (c == 27) // ESC
				{
					break;
				}
			}
		}

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
