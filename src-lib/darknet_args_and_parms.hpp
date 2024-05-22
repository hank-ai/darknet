#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	class ArgsAndParms final
	{
		public:

			enum class EType
			{
				kInvalid,
				kCommand,
				kFunction,
				kParameter
			};

			/// Destructor.
			~ArgsAndParms();

			/// Default constructor is needed for std::map.
			ArgsAndParms();

			/// Constructor.  @p n1 is the argument name, while @p n2 is an alternate name or spelling.
			ArgsAndParms(const std::string & n1, const std::string & n2 = "", const std::string & txt = "");

			ArgsAndParms(const std::string & n1, const EType t, const std::string & txt = "");

			/// Constructor.  Next argument must be an @p int parameter.
			ArgsAndParms(const std::string & n1, const std::string & n2, const int i, const std::string & txt = "");

			/// Constructor.  Next argument must be a @p float parameter.
			ArgsAndParms(const std::string & n1, const std::string & n2, const float f, const std::string & txt = "");

			/// The name of the argument or command.  For example, this could be @p "dontshow" or @p "version".
			std::string name;

			/// If the argument or command has an alternate spelling.  For example, this could be @p "color" (vs @p "colour").
			std::string name_alternate;

			std::string description;

			EType type;

			/// If an additional parameter is expected.  For example, @p "--threshold" should be followed by a number.
			bool expect_parm;

			/// The argument index into argv[].
			int arg_index;

			/// If @p expect_parm is @p true, then this would be the numeric value that comes next.
			float value;

			/// If this parameter is a filename, or the value is a filename, the path is stored here.
			std::filesystem::path filename;

			/// Needed to store these objects in an ordered set.
			bool operator<(const ArgsAndParms & rhs) const
			{
				return name < rhs.name;
			}
	};

	using SArgsAndParms = std::set<ArgsAndParms>;

	/// The key is the argument name, the value is the details for that argument.
	using MArgsAndParms = std::map<std::string, ArgsAndParms>;

	/** Get all the possible arguments used by Darknet.  This is not what the user specified to @p main() but @em all the
	 * possible arguments against which we validate the input.
	 */
	const SArgsAndParms & get_all_possible_arguments();

	void display_usage();
}
