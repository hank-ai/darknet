#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	class CfgLine final
	{
		public:

			/// Consructor.
			CfgLine();

			/// Consructor.
			CfgLine(const std::string & l, const size_t ln, const std::string & lhs, const std::string & rhs);

			/// Destructor.
			~CfgLine();

			bool empty() const { return line_number == 0 or line.empty() or key.empty() or val.empty(); }

			/// Log what we know about this line.
			std::string debug() const;

			size_t line_number;		///< line number
			std::string line;		///< original line of text
			std::string key;		///< the text that comes on the left side of the "="
			std::string val;		///< the text that comes on the right side of the "="
			std::optional<float> f;	///< if val is a single numeric value, it will be stored here
			bool used;				///< remember if this line was consumed when the configuraiton was parsed
	};

	/** Lines are not stored in the order in which they are read.  Instead, they are stored as a map which allows us to
	 * quickly look up a value based on the key.  There should be no duplicate keys within a section.
	 */
	using CfgLines = std::map<std::string, CfgLine>;


	/** A class that represents a specific section in a configuration file.  The start of each section has a @p [name]
	 * which gives the section a name.  For example, Darknet/YOLO config files start with either @p [net] or
	 * @p [network].
	 */
	class CfgSection final
	{
		public:

			/// Consructor.
			CfgSection();

			/// Consructor.
			CfgSection(const std::string & l, const size_t ln);

			/// Destructor.
			~CfgSection();

			/// Determine if a section is empty.
			bool empty() const { return line_number == 0 or name.empty() or lines.empty(); }

			const CfgSection & find_unused_lines() const;

			int find_int(const std::string & key, const int default_value);
			float find_float(const std::string & key, const float default_value);
			std::string find_str(const std::string & key, const std::string & default_value="");
			VFloat find_float_array(const std::string & key);

			/// Iterate over the section to log every line.
			std::string debug() const;

			ELayerType type;		///< the layer type for this section (e.g., [convolutional] or [yolo])
			std::string name;		///< the name of the section (so we don't have to keep looking up the type)
			size_t line_number;		///< line number where this section starts
			CfgLines lines;			///< all of the lines within a section
	};
	using CfgSections = std::vector<CfgSection>;


	/** A class that represents a Darknet/YOLO configuration file.  Contains various @ref "sections", which in turn has
	 * lines representing all of the options for each given section.
	 */
	class CfgFile final
	{
		public:

			/// Consructor.
			CfgFile();

			/// Consructor.  Immediately calls @ref read().
			CfgFile(const std::filesystem::path & fn);

			/// Destructor.
			~CfgFile();

			/// Read the given configuration file.  Forgets about any configuration file specified in the constructor (if any).
			CfgFile & read(const std::filesystem::path & fn);

			/// Read the configuration file that was specified in the constructor.
			CfgFile & read();

			/// Determine if a .cfg file has been parsed.
			bool empty() const { return sections.empty(); }

			/// Iterate over the content to record some debug information about the configuration.
			std::string debug() const;

			/// Create a Darknet network object from the configuration that was parsed.
			network * create_network();

			network create_network(network & net, const int batch=1, int time_steps=1);

			CfgFile & parse_net_section(network & net);

			convolutional_layer parse_convolutional_section(const size_t section_idx, network & net);

			std::filesystem::path filename;

			/// The [net] or [network] is not a "real" section, nor is it a layer.  We'll store it apart from the rest of the sections.
			CfgSection network_section;

			/// This is were we'll store every section *except* for the [net] one.  @see @ref network_section
			CfgSections sections;

			/// The total number of lines that was parsed from the .cfg file, including comments.
			size_t total_lines;
	};
}
