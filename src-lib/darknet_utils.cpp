#include "Chart.hpp"
#include "darknet_internal.hpp"


std::vector<std::string> Darknet::class_names;
std::vector<cv::Scalar> Darknet::class_colours;


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


void Darknet::remember_class_names(char ** names, const int count)
{
	TAT(TATPARMS);

	if (static_cast<size_t>(count) == class_names.size())
	{
		// assume this is a redundant call and we already know all of the class names and colours
		return;
	}

	printf("\nRemembering %d class%s:\n", count, (count == 1 ? "" : "es"));

	class_names.clear();
	class_names.reserve(count);

	class_colours.clear();
	class_colours.reserve(count);

	for (int idx = 0; idx < count; idx ++)
	{
		const std::string name = names[idx];
		if (name.find_first_not_of(" \t\r\n") == std::string::npos)
		{
			display_error_msg("The .names file appears to contain a blank line.\n");
		}

		class_names.push_back(name);

		const int offset = idx * 123457 % count;
		const int r = std::min(255.0f, std::round(256.0f * get_color(2, offset, count)));
		const int g = std::min(255.0f, std::round(256.0f * get_color(1, offset, count)));
		const int b = std::min(255.0f, std::round(256.0f * get_color(0, offset, count)));

		class_colours.push_back(CV_RGB(r, g, b));

		printf("-> class #%d (%s) will use colour #%02X%02X%02X\n", idx, names[idx], r, g, b);
	}

	printf("\n");

	return;
}


std::string Darknet::format_time(const double & seconds_remaing)
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << std::fixed << std::setprecision(3);

	if (seconds_remaing >= 0.5)
	{
		ss << seconds_remaing << " seconds";
	}
	else if (seconds_remaing >= 0.005)
	{
		ss << (seconds_remaing * 1000.0) << " milliseconds";
	}
	else
	{
		ss << (seconds_remaing * 1000000.0) << " microseconds";
	}

	return ss.str();
}


std::string Darknet::format_time_remaining(const float & seconds_remaining)
{
	TAT(TATPARMS);

	const float seconds	= 1.0;
	const float minutes	= 60.0 * seconds;
	const float hours	= 60.0 * minutes;
	const float days	= 24.0 * hours;
	const float weeks	= 7.0 * days;

	std::stringstream ss;
	ss << std::fixed << std::setprecision(0);

	if (seconds_remaining < 0.0f or seconds_remaining >= 4.0f * weeks)
	{
		ss << "unknown";
	}
	else if (seconds_remaining >= 2.0f * weeks)
	{
		ss << (seconds_remaining / weeks) << " weeks";
	}
	else if (seconds_remaining >= 2.0f * days)
	{
		ss << (seconds_remaining / days) << " days";
	}
	else if (seconds_remaining >= 2.0f * hours)
	{
		ss << (seconds_remaining / hours) << " hours";
	}
	else if (seconds_remaining >= 2.0f * minutes)
	{
		ss << (seconds_remaining / minutes) << " minutes";
	}
	else
	{
		const int secs = std::round(seconds_remaining);
		ss << secs << " second";
		if (secs != 1)
		{
			ss << "s";
		}
	}

	return ss.str();
}


std::string Darknet::format_loss(const double & loss)
{
	TAT(TATPARMS);

	EColour colour = EColour::kNormal;

	if (loss < 0.0		||
		loss >= 1000.0	||
		std::isfinite(loss) == false)
	{
		colour = EColour::kBrightRed;
	}
	else if (loss >= 20)
	{
		colour = EColour::kRed;
	}
	else if (loss >= 4)
	{
		colour = EColour::kCyan;
	}
	else
	{
		// else loss is somewhere between 0 and 4
		colour = EColour::kBrightCyan;
	}

	std::stringstream ss;
	ss << std::fixed << std::setprecision(3) << loss;

	return in_colour(colour, ss.str());
}


std::string Darknet::format_map_accuracy(const float & accuracy)
{
	TAT(TATPARMS);

	EColour colour = EColour::kNormal;

	if (accuracy < 0.5f || std::isfinite(accuracy) == false)
	{
		colour = EColour::kBrightRed;
	}
	else if (accuracy < 0.6f)
	{
		colour = EColour::kRed;
	}
	else if (accuracy < 0.7f)
	{
		colour = EColour::kBlue;
	}
	else
	{
		// else accuracy is >= 70%
		colour = EColour::kBrightBlue;
	}

	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << (100.0f * accuracy) << "%";

	return in_colour(colour, ss.str());
}


void Darknet::display_error_msg(const std::string & msg)
{
	TAT(TATPARMS);

	if (not msg.empty())
	{
		std::cout << in_colour(EColour::kBrightRed, msg);
	}

	return;
}


void Darknet::display_warning_msg(const std::string & msg)
{
	TAT(TATPARMS);

	if (not msg.empty())
	{
		std::cout << in_colour(EColour::kYellow, msg);
	}

	return;
}


std::string Darknet::text_to_simple_label(std::string txt)
{
	TAT(TATPARMS);

	// first we convert unknown characters to whitespace
	size_t pos = 0;
	while (true)
	{
		pos = txt.find_first_not_of(
			" "
			"0123456789"
			"abcdefghijklmnopqrstuvwxyz"
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ", pos);
		if (pos == std::string::npos)
		{
			break;
		}
		txt[pos] = ' ';
	}

	// then we merge consecutive spaces together
	pos = 0;
	while (true)
	{
		pos = txt.find("  ", pos);
		if (pos == std::string::npos)
		{
			break;
		}
		txt.erase(pos, 1);
	}

	// finally we convert any remaining whitespace to underscore so it can be used to create filenames without spaces
	pos = 0;
	while (true)
	{
		pos = txt.find(' ', pos);
		if (pos == std::string::npos)
		{
			break;
		}
		txt[pos] = '_';
	}

	return txt;
}


void Darknet::initialize_new_charts(const int max_batches, const float max_img_loss)
{
	TAT(TATPARMS);

	training_chart = Chart("", max_batches, max_img_loss);

	more_charts.clear();

	for (size_t idx = 0; idx < class_names.size(); idx ++)
	{
		Chart chart(class_names[idx], max_batches, max_img_loss);
		chart.map_colour = class_colours[idx];

		more_charts.push_back(chart);
	}

	return;
}


void Darknet::update_loss_in_new_charts(const int current_iteration, const float loss, const float seconds_remaining, const bool dont_show)
{
	TAT(TATPARMS);

	if (training_chart.empty() == false)
	{
		training_chart.update_save_and_display(current_iteration, loss, seconds_remaining, dont_show);

		for (auto & chart : more_charts)
		{
			chart.update_save_and_display(current_iteration, loss, seconds_remaining, true);
		}
	}

	return;
}


void Darknet::update_accuracy_in_new_charts(const int class_index, const float accuracy)
{
	TAT(TATPARMS);

	if (training_chart.empty() == false)
	{
		if (class_index < 0)
		{
			training_chart.update_accuracy(accuracy);
		}
		else if (static_cast<size_t>(class_index) < more_charts.size())
		{
			more_charts[class_index].update_accuracy(accuracy);
		}
	}

	return;
}


std::string Darknet::get_command_output(const std::string & cmd)
{
	std::string output;

	// loosely based on https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
	std::unique_ptr<FILE, void(*)(FILE*)> pipe(popen(cmd.c_str(), "r"),
		[](FILE * f) -> void
		{
			// wrapper to ignore the return value from pclose() is needed with newer versions of gnu g++
			std::ignore = pclose(f);
		});

	if (pipe == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to open a pipe to run %s", cmd.c_str());
	}

	while (true)
	{
		char buffer[200];
		const auto number_of_bytes = std::fread(buffer, 1, sizeof(buffer), pipe.get());
		if (number_of_bytes < 1)
		{
			break;
		}
		output += std::string(buffer, number_of_bytes);
	}

	return output;
}


void Darknet::cfg_layers()
{
	TAT(TATPARMS);

	std::map<LAYER_TYPE, std::string> layer_type_to_name;
	std::map<std::string, LAYER_TYPE> layer_name_to_type;
	for (int layer_type_index = 0; layer_type_index <= IMPLICIT; layer_type_index ++)
	{
		const LAYER_TYPE type = static_cast<LAYER_TYPE>(layer_type_index);
		std::string name = get_layer_string(type);

		layer_type_to_name[type] = name;
		layer_name_to_type[name] = type;
		std::cout << "layer #" << layer_type_index << "=" << name << std::endl;
	}

	if (layer_type_to_name.size() != layer_name_to_type.size())
	{
		darknet_fatal_error(DARKNET_LOC, "layer types and layer names do not match (%ld vs %ld)", layer_type_to_name.size(), layer_name_to_type.size());
	}

	const std::string home = []()
	{
		const char * tmp = getenv("HOME");
		if (tmp == nullptr)
		{
			return "";
		}
		return tmp;
	}();

	VStr places_to_look;

	if (not home.empty())
	{
		auto tmp = std::filesystem::path(home) / "src" / "darknet" / "cfg" / "yolov4.cfg";
		places_to_look.push_back(tmp.string());

		tmp = std::filesystem::path(home) / "Desktop" / "src" / "darknet" / "cfg" / "yolov4.cfg";
		places_to_look.push_back(tmp.string());
	}
	places_to_look.push_back("/src/darknet/cfg/yolov4.cfg");
	places_to_look.push_back("./cfg/yolov4.cfg");
	places_to_look.push_back("../cfg/yolov4.cfg");
	places_to_look.push_back("./yolov4.cfg");
	places_to_look.push_back("/opt/darknet/cfg/yolov4.cfg");
	places_to_look.push_back("C:/Program Files/darknet/cfg/yolov4.cfg");

	// first thing we need to do is locate the .cfg files
	std::filesystem::path config_dir;
	for (const auto & fn : places_to_look)
	{
		std::cout << "Looking for \"" << fn << "\"" << std::endl;
		if (std::filesystem::exists(fn))
		{
			// we found something we can use!
			config_dir = std::filesystem::canonical(std::filesystem::path(fn).parent_path());
			break;
		}
	}
	if (config_dir.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "cannot proceed since we did not find the .cfg files in any of the usual locations");
	}

	std::cout << "Using configuration subdirectory " << config_dir.string() << std::endl;

	// iterate over all the .cfg files in this directory
	VStr filenames;
	for (auto entry : std::filesystem::recursive_directory_iterator(config_dir))
	{
		const auto fn = entry.path();
		const auto ext = fn.extension();
		if (ext == ".cfg")
		{
			filenames.push_back(fn.string());
		}
	}

	std::sort(filenames.begin(), filenames.end());

	const std::regex rx(
		"("			// group #1
		"\\d{4}"	// YYYY
		"-"
		"\\d{2}"	// MM
		"-"
		"\\d{2}"	// DD
		")"
		" "			// "space"
		"("			// group #2
		".+"		// everything that remains (the username)
		")"
		);

	std::cout << "Saving results for " << filenames.size() << " config files to cfg_layers_output.html" << std::endl;

	std::ofstream ofs("cfg_layers_output.html");
	ofs << "<html>" << std::endl
		<< "<head>" << std::endl
		<< "<title>Layers Used In " << filenames.size() << " Configuration Files</title>" << std::endl
		<< "<style>" << std::endl
		<< "table, tr, th, td { white-space: nowrap; border-collapse: collapse; border: 1px solid black; }" << std::endl
		<< "th { padding: 5px; }" << std::endl
		<< "</style>" << std::endl
		<< "</head>" << std::endl
		<< "<body>" << std::endl
		<< "<p>Darknet " << DARKNET_VERSION_STRING << "<br>" << std::endl
		<< "Parsing " << filenames.size() << " config files in " << config_dir.string() << "</p>" << std::endl
		<< "<table>" << std::endl;

	std::stringstream table_header_row;
	table_header_row
		<< "\t<tr>" << std::endl
		<< "\t\t<th>#</th>" << std::endl
		<< "\t\t<th>filename</th>" << std::endl
		<< "\t\t<th>last author</th>" << std::endl
		<< "\t\t<th>last commit</th>" << std::endl;

	std::set<LAYER_TYPE> completely_unused_layer_types;
	std::map<LAYER_TYPE, size_t> number_of_times_layer_was_referenced;
	std::map<LAYER_TYPE, size_t> number_of_times_layer_was_referenced_by_modern_config;

	// loop through all the layers to add the headers at the top of the table
	for (int layer_index = 0; layer_index <= IMPLICIT; layer_index ++)
	{
		const auto type = static_cast<LAYER_TYPE>(layer_index);
		const auto name = layer_type_to_name[type];

		completely_unused_layer_types.insert(type);
		number_of_times_layer_was_referenced[type] = 0;
		number_of_times_layer_was_referenced_by_modern_config[type] = 0;

		table_header_row << "\t\t<th>" << name << "</th>" << std::endl;
	}
	table_header_row << "\t</tr>" << std::endl;
	ofs << table_header_row.str();

	std::filesystem::current_path(config_dir);

	for (size_t file_counter = 0; file_counter < filenames.size(); file_counter ++)
	{
		const auto & fn = filenames[file_counter];

		// key=layer name (net, convolutional, yolo, ...)
		// val=count the number of times that layer appears in the configuration
		std::map<std::string, size_t> m;

		// get the filename relative to the config_dir subdirectory
		const std::string short_fn = fn.substr(config_dir.string().size() + 1);

		// output should look similar to this:  2016-11-17 Joseph Redmon
		const std::string output = get_command_output("git log -1 --pretty=format:\"%as %an\" \"" + short_fn + "\" 2>&1");

		std::string date = "unknown";
		std::string name = "unknown";

		std::smatch matches;
		if (std::regex_match(output, matches, rx))
		{
			date = matches[1];
			name = matches[2];
		}

		std::ifstream ifs(fn);
		std::string line;
		while (std::getline(ifs, line))
		{
			size_t sz = line.size();
			if (sz > 2 and
				line[0] == '[' and
				line[sz - 1] == ']')
			{
				const auto name = line.substr(1, sz - 2);

				if (layer_name_to_type.count(name) == 0)
				{
					darknet_fatal_error(DARKNET_LOC, "layer \"%s\" in %s does not seem to be a valid name", name.c_str(), fn.c_str());
				}

				m[name] ++;
			}
		}

		// output what we know about this file

		ofs	<< "\t<tr>" << std::endl
			<< "\t\t<td>" << (file_counter + 1) << "</td>" << std::endl
			<< "\t\t<td>" << short_fn << "</td>" << std::endl
			<< "\t\t<td>" << name << "</td>" << std::endl
			<< "\t\t<td>" << date << "</td>" << std::endl;

		// loop through all the layers
		for (int layer_index = 0; layer_index <= IMPLICIT; layer_index ++)
		{
			const auto type = static_cast<LAYER_TYPE>(layer_index);
			const auto name = layer_type_to_name[type];

			if (m.count(name) == 0)
			{
				ofs << "\t\t<td style=\"background-color: yellow;\">&nbsp;</td>" << std::endl;
			}
			else
			{
				ofs << "\t\t<td style=\"text-align: center;\">" << m[name] << "</td>" << std::endl;
				completely_unused_layer_types.erase(type);
				number_of_times_layer_was_referenced[type] ++;

				// YOLOv3 has a date of 2018-05-06, so let's use the start of that month as the cutoff for what we consider "modern"
				if (date >= "2018-05-01")
				{
					number_of_times_layer_was_referenced_by_modern_config[type] ++;
				}
			}
		}
		ofs << "\t</tr>" << std::endl;
	}
	ofs << table_header_row.str();
	ofs << "</table>" << std::endl;

	ofs	<< "<p>Layer types which remain unused after parsing " << filenames.size() << " configuration files:  " << completely_unused_layer_types.size() << std::endl
		<< "<ul>" << std::endl;
	for (const auto & type : completely_unused_layer_types)
	{
		ofs << "\t<li>" << layer_type_to_name[type] << "</li>" << std::endl;
	}
	ofs	<< "</ul>" << std::endl
		<< "</p>" << std::endl;

	ofs	<< "<p>Least-used layers:" << std::endl
		<< "<ul>" << std::endl;

	// yuk this is bad, but this is a debug tool and doesn't need to perfectly efficient
	for (size_t counter = 0; counter < 10; counter ++)
	{
		for (const auto & [type, count] : number_of_times_layer_was_referenced)
		{
			if (count == counter)
			{
				ofs << "\t<li>" << layer_type_to_name[type] << " was used " << count << " time" << (count == 1 ? "" : "s") << "</li>" << std::endl;
			}
		}
	}
	ofs << "</ul>" << std::endl
		<< "</p>" << std::endl;

	ofs	<< "<p>If we ignore old configurations (pre-YOLOv3) then these layers are no longer (or rarely) used:" << std::endl
		<< "<ul>" << std::endl;
	for (size_t counter = 0; counter < 5; counter ++)
	{
		for (const auto & [type, count] : number_of_times_layer_was_referenced_by_modern_config)
		{
			if (count == counter)
			{
				ofs << "\t<li>" << layer_type_to_name[type] << " was used " << count << " time" << (count == 1 ? "" : "s") << " in a modern config</li>" << std::endl;
			}
		}
	}
	ofs << "</ul>" << std::endl
		<< "</p>" << std::endl;

	ofs	<< "</body>" << std::endl
		<< "</html>" << std::endl;

	std::cout << "Done." << std::endl;

	return;
}
