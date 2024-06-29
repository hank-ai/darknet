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


const Darknet::SArgsAndParms & Darknet::get_all_possible_arguments()
{
	TAT(TATPARMS);

	static const SArgsAndParms all =
	{
		ArgsAndParms("3d"			, ArgsAndParms::EType::kCommand	, "Pass in 2 images as input."),
		ArgsAndParms("average"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("calcanchors"	, ArgsAndParms::EType::kFunction, ""),
		ArgsAndParms("cfglayers"	, ArgsAndParms::EType::kCommand, "Obtain some information on all config files and layers used."),
		ArgsAndParms("classify"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("classifier"	, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("demo"			, ArgsAndParms::EType::kFunction, "Process a video using the given neural network."),
		ArgsAndParms("denormalize"	, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("detect"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("detector"		, ArgsAndParms::EType::kCommand	, "Train or check neural networks."),
		ArgsAndParms("draw"			, ArgsAndParms::EType::kFunction, ""),
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
		ArgsAndParms("rnn"			, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("speed"		, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("statistics"	, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("test"			, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("test"			, ArgsAndParms::EType::kFunction, ""),
		ArgsAndParms("train"		, ArgsAndParms::EType::kFunction, "Train a new neural network, or continue training an existing neural network."),
		ArgsAndParms("valid"		, ArgsAndParms::EType::kFunction, ""),
		ArgsAndParms("version"		, ArgsAndParms::EType::kCommand, "Display version information."),
		ArgsAndParms("vid"			, ArgsAndParms::EType::kCommand	, ""),
		ArgsAndParms("visualize"	, ArgsAndParms::EType::kCommand	, ""),

		// global options
		ArgsAndParms("colour"	, "color"	),
		ArgsAndParms("nocolour"	, "nocolor"	),
		ArgsAndParms("verbose"	, "show_details"),	// I originally didn't know about "show_details" when I implemented "verbose"
		ArgsAndParms("trace"				),		// even more verbose

		// other options

		ArgsAndParms("camera"	, "c"			, 0		),
		ArgsAndParms("dontshow"	, "noshow"				),
		ArgsAndParms("thresh"	, "threshold"	, 0.24f	),

		ArgsAndParms("avgframes"			), //-- takes an int  3
		ArgsAndParms("benchmark"			),
		ArgsAndParms("benchmarklayers"		),
		ArgsAndParms("checkmistakes"		),
		ArgsAndParms("clear"				),
		ArgsAndParms("dontdrawbbox"			),
		ArgsAndParms("jsonport"				),
		ArgsAndParms("letterbox"			),
		ArgsAndParms("mjpegport"			), //-- takes an int?
		ArgsAndParms("points"				), //-- takes an int?  0
		ArgsAndParms("show"					),
		ArgsAndParms("showimgs"				),
		ArgsAndParms("httpposthost"			),
		ArgsAndParms("timelimitsec"			),
		ArgsAndParms("outfilename"			),
		ArgsAndParms("out"					),
		ArgsAndParms("prefix"				),
		ArgsAndParms("iouthresh"			),
		ArgsAndParms("hier"					),
//		ArgsAndParms("c"					),
		ArgsAndParms("s"					),
		ArgsAndParms("numofclusters"		),
		ArgsAndParms("width"				),
		ArgsAndParms("height"				),
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
		<< "\t\t" << CfgAndState::get().argv[0] << " <command> [<options>] [<function>] [<more options and filenames>]" << std::endl
		<< std::endl
		<< "Commands:" << std::endl;

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

	std::cout << std::endl << std::endl;

	// show the details for the commands where we have descriptions, one per line
	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kCommand and not item.description.empty())
		{
			std::cout << "  " << std::setw(10) << item.name << ":  " << item.description << std::endl;
		}
	}

	// show all the functions (short form)
	std::cout  << std::endl << "Functions:" << std::endl;
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

	std::cout << std::endl << std::endl;

	// show the details for the functions where we have descriptions, one per line
	for (const auto & item : all)
	{
		if (item.type == ArgsAndParms::EType::kFunction and not item.description.empty())
		{
			std::cout << "  " << std::setw(10) << item.name << ":  " << item.description << std::endl;
		}
	}

	std::cout << std::endl << "Options:" << std::endl;
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
	std::cout <<  std::endl;

	return;
}
