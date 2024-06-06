#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	class CfgLine final
	{
		public:

			/// Consructor.
			CfgLine(const std::string & l, const size_t ln, const std::string & lhs, const std::string & rhs);

			/// Destructor.
			~CfgLine();

			/// Log what we know about this line.
			std::string debug() const;

			size_t line_number;		///< line number
			std::string line;		///< original line of text
			std::string key;		///< the text that comes on the left side of the "="
			std::string val;		///< the text that comes on the right side of the "="
			std::optional<float> f;	///< if val is a single numeric value, it will be stored here
			bool used;				///< remember if this line was consumed when the configuraiton was parsed
	};
	using CfgLines = std::list<CfgLine>;


	/** A class that represents a specific section in a configuration file.  The start of each section has a @p [name]
	 * which gives the section a name.  For example, Darknet/YOLO config files start with either @p [net] or
	 * @p [network].
	 */
	class CfgSection final
	{
		public:

			/// Consructor.
			CfgSection(const std::string & l, const size_t ln);

			/// Destructor.
			~CfgSection();

			/// Iterate over the section to log every line.
			std::string debug() const;

			ELayerType type;		///< the layer type for this section (e.g., [convolutional] or [yolo])
			size_t line_number;		///< line number where this section starts
			CfgLines lines;			///< all of the lines within a section
	};
	using CfgSections = std::list<CfgSection>;


	/** A class that represents a Darknet/YOLO configuration file.  Contains various @ref "sections", which in turn has
	 * lines representing all of the options for each given section.
	 */
	class CfgFile final
	{
		public:

			/// Consructor.
			CfgFile();

			/// Consructor.  Immediately calls @ref parse().
			CfgFile(const std::filesystem::path & fn);

			/// Destructor.
			~CfgFile();

			/// Parse the given configuration file.  Forgets about any configuration file specified in the constructor (if any).
			CfgFile & parse(const std::filesystem::path & fn);

			/// Parse the configuration file that was specified in the constructor.
			CfgFile & parse();

			/// Determine if a .cfg file has been parsed.
			bool empty() const { return sections.empty(); }

			/// Iterate over the content to record some debug information about the configuration.
			std::string debug() const;

			std::filesystem::path filename;

			CfgSections sections;

			/// The total number of lines that was parsed from the .cfg file, including comments.
			size_t total_lines;
	};
}
