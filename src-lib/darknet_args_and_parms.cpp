#include "darknet_internal.hpp"


Darknet::ArgsAndParms::~ArgsAndParms()
{
	TAT(TATPARMS);

	return;
}


Darknet::ArgsAndParms::ArgsAndParms() :
	ArgsAndParms("", "")
{
	TAT(TATPARMS);

	return;
}


Darknet::ArgsAndParms::ArgsAndParms(const std::string & n1, const std::string & n2, const std::string & txt) :
	name			(n1					),
	name_alternate	(n2					),
	description		(txt				),
	type			(EType::kParameter	),
	expect_parm		(false				),
	arg_index		(-1					),
	value			(0.0f				)
{
	TAT(TATPARMS);

	return;
}


Darknet::ArgsAndParms::ArgsAndParms(const std::string & n1, const EType t, const std::string & txt) :
	ArgsAndParms(n1, "", txt)
{
	TAT(TATPARMS);

	type = t;

	return;
}


Darknet::ArgsAndParms::ArgsAndParms(const std::string & n1, const std::string & n2, const EType t, const std::string & txt) :
	ArgsAndParms(n1, n2, txt)
{
	TAT(TATPARMS);

	type = t;

	return;
}


Darknet::ArgsAndParms::ArgsAndParms(const std::string & n1, const std::string & n2, const int i, const std::string & txt) :
	ArgsAndParms(n1, n2, txt)
{
	TAT(TATPARMS);

	expect_parm	= true;
	value		= i;
}


Darknet::ArgsAndParms::ArgsAndParms(const std::string & n1, const std::string & n2, const float f, const std::string & txt) :
	ArgsAndParms(n1, n2, txt)
{
	TAT(TATPARMS);

	expect_parm	= true;
	value		= f;

	return;
}


Darknet::ArgsAndParms::ArgsAndParms(const std::string & n1, const std::string & n2, const std::string & str, const std::string & txt) :
	ArgsAndParms(n1, n2, txt)
{
	TAT(TATPARMS);

	expect_parm	= true;
	this->str = str;

	return;
}


const Darknet::SArgsAndParms & Darknet::get_all_possible_arguments()
{
	TAT(TATPARMS);

	static const SArgsAndParms all =
	{
		ArgsAndParms("3d"			, ArgsAndParms::EType::kCommand	, "Pass in 2 images as input."),
		ArgsAndParms("average"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("calcanchors"	, ArgsAndParms::EType::kFunction, "Recalculate YOLO anchors."),
		ArgsAndParms("cfglayers"	, ArgsAndParms::EType::kCommand, "Display some information on all config files and layers used."),
		ArgsAndParms("denormalize"	, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("detect"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("detector"		, ArgsAndParms::EType::kCommand	, "Train or check neural networks."),
		ArgsAndParms("help"			, ArgsAndParms::EType::kCommand	, "Display usage information."),
		ArgsAndParms("imtest"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("map"			, ArgsAndParms::EType::kFunction, "Calculate mean average precision for a given dataset."),
		ArgsAndParms("nightmare"	, ArgsAndParms::EType::kCommand	, "Run a neural network in reverse to generate strange images."),
		ArgsAndParms("normalize"	, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("oneoff"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("ops"			, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("partial"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("recall"		, ArgsAndParms::EType::kFunction, ""),
		ArgsAndParms("rescale"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("reset"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("rgbgr"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("speed"		, ArgsAndParms::EType::kCommand	, "Perform a quick test to see how fast the specified neural network runs."),
		ArgsAndParms("statistics"	, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("test"			, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("test"			, ArgsAndParms::EType::kFunction, ""),
		ArgsAndParms("train"		, ArgsAndParms::EType::kFunction, "Train a new neural network, or continue training an existing neural network."),
		ArgsAndParms("valid"		, ArgsAndParms::EType::kFunction, ""),
		ArgsAndParms("version"		, ArgsAndParms::EType::kCommand	, "Display version information."),
		ArgsAndParms("visualize"	, ArgsAndParms::EType::kCommand	, "Display the weights from diferent layers in a neural network."),

		// global options
		ArgsAndParms("colour"		, "color"							, "Enable colour output in the console.  This is the default."),
		ArgsAndParms("nocolour"		, "nocolor"							, "Disable colour output in the console."),

		// I originally didn't know about "show_details" when I implemented "verbose".
		ArgsAndParms("verbose"		, "show_details"					, "Logs more verbose messages."),
		ArgsAndParms("trace"		, ArgsAndParms::EType::kParameter	, "Intended for debug purposes.  This allows Darknet to log trace messages for some commands."),

		// other options

		ArgsAndParms("dontshow"		, "noshow"							, "Do not open a GUI window.  Especially useful when used on a headless server.  This will cause the output image to be saved to disk."),
		ArgsAndParms("clear"		, ArgsAndParms::EType::kParameter	, "Used during training to reset the \"image count\" to zero, necessary when pre-existing weights are used."),
		ArgsAndParms("map"			, ArgsAndParms::EType::kParameter	, "Regularly calculate mAP% score while training."),

		ArgsAndParms("camera"	, "c"			, 0		, "The camera (webcam) index, where numbering is typically sequential and begins with zero."),
		ArgsAndParms("thresh"	, "threshold"	, 0.24f	),

		ArgsAndParms("avgframes"			), //-- takes an int  3
		ArgsAndParms("benchmark"			),
		ArgsAndParms("benchmarklayers"		),
		ArgsAndParms("letterbox"			),
		ArgsAndParms("points"				), //-- takes an int?  0
		ArgsAndParms("random"				, ArgsAndParms::EType::kParameter, "Randomize the list of images.  Default is to sort alphabetically."),
		ArgsAndParms("show"					, ArgsAndParms::EType::kParameter, "Visually display the anchors."),
		ArgsAndParms("heatmaps", "heatmap"	, ArgsAndParms::EType::kParameter, "Display the heatmaps for each class."),
		ArgsAndParms("showimgs"				),
		ArgsAndParms("httpposthost"			),
		ArgsAndParms("timelimitsec"			),
		ArgsAndParms("outfilename"			),
		ArgsAndParms("out"					),
		ArgsAndParms("prefix"				),
		ArgsAndParms("iouthresh"			),
		ArgsAndParms("hier"					),
		ArgsAndParms("s"					),
		ArgsAndParms("numofclusters"		, "", 6		, "The number of YOLO anchors in the configuration. --num_of_clusters 6"	),
		ArgsAndParms("width"				, "", 416	, "The width of the network.  --width 416"									),
		ArgsAndParms("height"				, "", 416	, "The height of the network.  --width 416"									),
		ArgsAndParms("skipclasses"			, "", " "	, "Class indexes which Darknet should skip when returning results or annotating images.  --skip-classes=2,5-8"),
		ArgsAndParms("extoutput"			),
		ArgsAndParms("savelabels"			),
		ArgsAndParms("chart"				),
	};

	return all;
}


void Darknet::display_usage()
{
	TAT(TATPARMS);

	const auto & all = Darknet::get_all_possible_arguments();

	std::cout
		<< std::endl
		<< "Darknet/YOLO CLI usage:" << std::endl
		<< std::endl
		<< "\t\tdarknet <command> [<options>] [<function>] [<more options and filenames>]" << std::endl
		<< std::endl
		<< "Commands:" << std::endl
		<< Darknet::in_colour(Darknet::EColour::kBrightCyan)
		;

	// show all commands (short form)
	int col = 0;
	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kCommand)
		{
			col += 2 + item.name.length();
			if (col >= 80)
			{
				std::cout << std::endl;
				col = 0;
			}
			std::cout << "  " << item.name;
		}
	}

	std::cout << Darknet::in_colour(Darknet::EColour::kNormal) << std::endl << std::endl;

	// show the details for the commands where we have descriptions, one per line
	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kCommand and not item.description.empty())
		{
			std::cout << "  " << Darknet::format_in_colour(item.name, Darknet::EColour::kBrightWhite, -10) << ":  " << item.description << std::endl;
		}
	}

	// show all the functions (short form)
	std::cout << std::endl << "Functions:" << std::endl << Darknet::in_colour(Darknet::EColour::kBrightCyan);
	col = 0;
	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kFunction)
		{
			col += 2 + item.name.length();
			if (col >= 80)
			{
				std::cout << std::endl;
				col = 0;
			}
			std::cout << "  " << item.name;
		}
	}

	std::cout << Darknet::in_colour(Darknet::EColour::kNormal) << std::endl << std::endl;

	// show the details for the functions where we have descriptions, one per line
	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kFunction and not item.description.empty())
		{
			std::cout << "  " << Darknet::format_in_colour(item.name, Darknet::EColour::kBrightWhite, -10) << ":  " << item.description << std::endl;
		}
	}

	std::cout << std::endl << "Options:" << std::endl << Darknet::in_colour(Darknet::EColour::kBrightCyan);
	col = 0;
	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kParameter)
		{
			col += 2 + item.name.length();
			if (col >= 80)
			{
				std::cout << std::endl;
				col = 0;
			}
			std::cout << "  " << item.name;
		}
	}
	std::cout << Darknet::in_colour(Darknet::EColour::kNormal) << std::endl << std::endl;

	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kParameter and not item.description.empty())
		{
			if (item.expect_parm)
			{
				std::cout << "  " << Darknet::format_in_colour(item.name, Darknet::EColour::kBrightWhite, -15) << " <...>  " << item.description << std::endl;
			}
			else
			{
				std::cout << "  " << Darknet::format_in_colour(item.name, Darknet::EColour::kBrightWhite, -15) << "        " << item.description << std::endl;
			}
		}
	}

	std::cout << std::endl << "Several options have aliases for convenience:" << std::endl;

	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kParameter and not item.name_alternate.empty())
		{
			std::cout
				<< "  "		<< Darknet::format_in_colour(item.name			, Darknet::EColour::kBrightWhite, -10)
				<< " -> "	<< Darknet::format_in_colour(item.name_alternate, Darknet::EColour::kBrightWhite, 10)
				<< std::endl;
		}
	}

	const auto YELLOW = [](const std::string & msg) -> std::string
	{
		return Darknet::in_colour(Darknet::EColour::kYellow, msg);
	};

	std::cout
		<< ""																						<< std::endl
		<< "For most Darknet commands, dashes and underscores for CLI parameters are"				<< std::endl
		<< "ignored.  Additionally, the order in which parameters are specified should"				<< std::endl
		<< "be insignificant.  For example, these parameters:"										<< std::endl
		<< ""																						<< std::endl
		<< "    " << in_colour(EColour::kBrightWhite, "-verbose -show -num_of_clusters 6")			<< std::endl
		<< ""																						<< std::endl
		<< "...are exactly the same as:"															<< std::endl
		<< ""																						<< std::endl
		<< "    " << in_colour(EColour::kBrightWhite, "numofclusters 6 verbose show")				<< std::endl
		<< ""																						<< std::endl
		<< "Some legacy commands have not yet been fixed or converted to use the new"				<< std::endl
		<< "command-line parsing, but with time all commands will be retrofited to"					<< std::endl
		<< "use this new format."																	<< std::endl
		<< ""																						<< std::endl
		<< "Several example commands to run:"														<< std::endl
		<< ""																						<< std::endl
		<< "  Recalculate YOLO anchors:"															<< std::endl
		<< YELLOW("    darknet detector calcanchors cars.data -show -num_of_clusters 6 -width 320 -height 160") << std::endl
		<< ""																						<< std::endl
		<< "  Train a new network:"																	<< std::endl
		<< YELLOW("    darknet detector train -map -dont_show cars.data cars.cfg")					<< std::endl
		<< ""																						<< std::endl
		<< "  Train a network starting using existing weights:"										<< std::endl
		<< YELLOW("    darknet detector train -map -dont_show -clear cars.data cars.cfg cars_best.weights") << std::endl
		<< ""																						<< std::endl
		<< "  Check the mAP% results:"																<< std::endl
		<< YELLOW("    darknet detector map cars.data cars.cfg cars_best.weights")					<< std::endl
		<< ""																						<< std::endl
		<< "  Apply the neural network to an image and save the results to disk:"					<< std::endl
		<< YELLOW("    darknet detector test -dont_show cars.data cars.cfg cars_best.weights image1.jpg") << std::endl
		<< "  The equivalent V3 simplified command:"												<< std::endl
		<< YELLOW("    darknet_01_inference_images cars.cfg image1.jpg")							<< std::endl
		<< ""																						<< std::endl
		<< "  Apply the neural network to an image and show the results:"							<< std::endl
		<< YELLOW("    darknet detector test cars.data cars.cfg cars_best.weights image1.jpg")		<< std::endl
		<< "  The equivalent V3 simplified command:"												<< std::endl
		<< YELLOW("    darknet_02_display_annotated_images cars.cfg image1.jpg")					<< std::endl
		<< ""																						<< std::endl
		<< "  Apply the neural network to a video:"													<< std::endl
		<< YELLOW("    darknet detector demo cars.data cars.cfg cars_best.weights -ext_output video1.mp4") << std::endl
		<< "  The equivalent V3 simplified command to display the video in \"realtime\":"			<< std::endl
		<< YELLOW("    darknet_03_display_videos cars.cfg video1.mp4")								<< std::endl
		<< "  The equivalent V3 simplified command to process the video using a single thread:"		<< std::endl
		<< YELLOW("    darknet_04_process_videos cars.cfg video1.mp4")								<< std::endl
		<< "  The equivalent V3 simplified command to process the video using multiple threads:"	<< std::endl
		<< YELLOW("    darknet_05_process_videos_multithreaded cars.cfg video1.mp4")				<< std::endl
		<< "  The previous V3 commands will generate an output video.  The equivalent V2-style command is:" << std::endl
		<< YELLOW("    darknet detector demo cars.data cars.cfg cars_best.weights video1.mp4 -out_filename output.mp4") << std::endl
		<< ""																						<< std::endl
		<< "  Display the weights from different layers in a neural network:"						<< std::endl
		<< YELLOW("    darknet visualize --verbose cars.cfg cars_best.weights")						<< std::endl
		<< ""																						<< std::endl
		<< "  Display the YOLO heatmaps alongside images and videos:"								<< std::endl
		<< YELLOW("    darknet_02_display_annotated_images --heatmaps cars images/*.jpg")			<< std::endl
		<< YELLOW("    darknet_03_display_videos --heatmaps cars videos/*.m4v")						<< std::endl
		<< ""																						<< std::endl
		<< "  Randomize (versus alphabetically sorting) the set of images or videos:"				<< std::endl
		<< YELLOW("    darknet_02_display_annotated_images --random cars images/*.jpg")				<< std::endl
		<< YELLOW("    darknet_03_display_videos --random cars videos/*.m4v")						<< std::endl
		<< ""																						<< std::endl;

	return;
}

