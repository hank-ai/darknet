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

			/// Determine if a line is empty.
			bool empty() const { return line_number == 0 or line.empty() or key.empty() or val.empty(); }

			/// Log what we know about this line.
			std::string debug() const;

			size_t line_number;		///< The line number within the .cfg file.
			std::string line;		///< Original line of text.
			std::string key;		///< The text that comes on the left side of the "=".
			std::string val;		///< The text that comes on the right side of the "=".
			std::optional<float> f;	///< If val is a single numeric value, it will be stored here.
			bool used;				///< Remember if this line was consumed when the configuraiton was parsed.  @see @ref Darknet::CfgSection::find_unused_lines()
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

			/// Verify that there are no "unused" lines in this section.
			const CfgSection & find_unused_lines() const;

			/// Find an @p int config item.  The key @em must exist.
			int find_int(const std::string & key);

			/// Find an @p int config item.  If the key does not exist, then the given default value will be returned.
			int find_int(const std::string & key, const int default_value);

			/// Find a @p float config item.  If the key does not exist, then the given default value will be returned.
			float find_float(const std::string & key, const float default_value);

			/// Find a text config item.  If the key does not exist, then the given default value will be returned.
			std::string find_str(const std::string & key, const std::string & default_value="");

			/// Find an array of @p float values.  If the key does not exist, then an empty @p std::vector will be returned.
			VFloat find_float_array(const std::string & key);

			/// Find an array of @p int values.  If the key does not exist, then an empty @p std::vector will be returned.
			VInt find_int_array(const std::string & key);

			/// Iterate over the section to log every line.
			std::string debug() const;

			ELayerType	type;			///< The layer type for this section (e.g., [convolutional] or [yolo]).
			std::string	name;			///< The name of the section (so we don't have to keep looking up the type).
			size_t		line_number;	///< Line number where this section starts.
			CfgLines	lines;			///< All of the lines within a section.
	};
	using CfgSections = std::vector<CfgSection>;


	/** A class that represents a Darknet/YOLO configuration file.  Contains various @ref "sections", which in turn has
	 * lines representing all of the options for each given section.  Typical use is to call @ref read() followed by
	 * @ref create_network().
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

			/** Read the given configuration file and parses the individual sections and lines.
			 * Forgets about any configuration file specified in the constructor (if any).
			 *
			 * @note Remember to call @ref create_network() after @p read() has finished.
			 */
			CfgFile & read(const std::filesystem::path & fn);

			/** Read the configuration file that was specified in the constructor.
			 *
			 * @note Remember to call @ref create_network() after @p read() has finished.
			 */
			CfgFile & read();

			/// Determine if a .cfg file has been parsed.
			bool empty() const { return sections.empty(); }

			/// Iterate over the content to record some debug information about the configuration.
			std::string debug() const;

			/** Create and populate the Darknet @p network object @ref net from the configuration that was parsed.
			 *
			 * @note Remember to call @ref read() prior to @p create_network().
			 */
			network & create_network(const int batch=1, int time_steps=1);

			/// The configuration file.
			std::filesystem::path filename;

			/** The [net] or [network] is not a "real" section, nor is it a layer.
			 * This is only populated after @ref read() has been called.
			 *
			 * @see @ref sections
			 */
			CfgSection network_section;

			/** This is were we'll store every section *except* for the [net] one.
			 * This is only populated after @ref read() has been called.
			 *
			 * @see @ref network_section
			 */
			CfgSections sections;

			/// The total number of lines that was parsed from the .cfg file, including comments and blank lines.
			size_t total_lines;

			/// This will remain uninitialized until @ref create_network() is called.
			network net;

			/** Temporary fields which are needed while creating the @ref net object.
			 * It is unlikely that this needs to be exposed or modified externally,
			 * but it must be exposed for use in @ref dump() and the old
			 * @p parse_network_cfg_custom() function.
			 */
			struct CommonParms
			{
				int batch;
				int inputs;
				int h;
				int w;
				int c;
				int index;
				int time_steps;
				int train;
				int last_stop_backward;
				int avg_outputs;
				int avg_counter;
				float bflops;
				size_t workspace_size;
				size_t max_inputs;
				size_t max_outputs;
				int receptive_w;
				int receptive_h;
				int receptive_w_scale;
				int receptive_h_scale;
				int show_receptive_field;
			};
			CommonParms parms;

		private:

			/// @{ Methods to parse different types of sections in .cfg files.
			CfgFile &			parse_net_section			();
			convolutional_layer	parse_convolutional_section	(const size_t section_idx);
			route_layer			parse_route_section			(const size_t section_idx);
			maxpool_layer		parse_maxpool_section		(const size_t section_idx);
			layer				parse_yolo_section			(const size_t section_idx);
			layer				parse_upsample_section		(const size_t section_idx);
			layer				parse_shortcut_section		(const size_t section_idx);
			connected_layer		parse_connected_section		(const size_t section_idx);
			layer				parse_crnn_section			(const size_t section_idx);
			layer				parse_rnn_section			(const size_t section_idx);
			maxpool_layer		parse_local_avgpool_section	(const size_t section_idx);
			layer				parse_lstm_section			(const size_t section_idx);
			layer				parse_reorg_section			(const size_t section_idx);
			avgpool_layer		parse_avgpool_section		(const size_t section_idx);
			cost_layer			parse_cost_section			(const size_t section_idx);
			layer				parse_region_section		(const size_t section_idx);
			layer				parse_gaussian_yolo_section	(const size_t section_idx);
			layer				parse_contrastive_section	(const size_t section_idx);
			softmax_layer		parse_softmax_section		(const size_t section_idx);
			layer				parse_scale_channels_section(const size_t section_idx);
			layer				parse_sam_section			(const size_t section_idx);
			dropout_layer		parse_dropout_section		(const size_t section_idx);
			/// @}
	};
}
