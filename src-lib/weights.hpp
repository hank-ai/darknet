#pragma once

#include "darknet.h"
#include <filesystem>

#ifdef __cplusplus
extern "C" {
#endif
network parse_network_cfg(const char * filename);
network parse_network_cfg_custom(const char * filename, int batch, int time_steps);
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_upto(network net, char *filename, int cutoff, int save_ema);
void save_weights_double(network net, char *filename);
void load_weights(network * net, const char * filename);
void load_weights_upto(network * net, const char * filename, int cutoff);

#ifdef __cplusplus
}
#endif

namespace Darknet
{
	/** Load all of the names from a text @p .names file.  The file must exist, and must have the same number of classes
	 * as the network.
	 *
	 * This will automatically be called if you use @ref Darknet::load_neural_network() from the new
	 * recommended V3 API.
	 *
	 * You may need to manually call it if you use the old @ref load_network() or @ref load_network_custom() from the
	 * original @p C API.  If you don't call it, some default placeholders will be used instead, such as @p "#0", @p "#1",
	 * etc.
	 *
	 * @since 2024-08-06
	 */
	void load_names(network * net, const std::filesystem::path & filename);

	/** Generate the necessary class colours used to draw bounding boxes and labels.  The colours are stored in
	 * @ref Darknet::NetworkDetails at the time the network is loaded, once the total number of classes are known.
	 *
	 * This is called @em automatically when the network is initialized via @ref Darknet::CfgFile.  There is no need to
	 * manually call this function.
	 *
	 * @since 2024-08-06
	 */
	void assign_default_class_colours(network * net);
}
