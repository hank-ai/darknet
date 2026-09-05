#pragma once

/** @file
 * Graph-only Apple training entry points used by Darknet.
 */

#include <string>

#include "apple_backend.hpp"

namespace Darknet
{
	struct Network;

	namespace Apple
	{
		/// Return @p true if the Apple graph-training runtime can be used.
		bool training_graph_runtime_available(std::string *reason = nullptr);

		/// Build and cache the Apple graph-training resources for @p net.
		bool configure_training_graph(Network & net, std::string *reason);

		/// Return @p true if Apple graph training is currently enabled for @p net.
		bool training_graph_enabled(const Network & net);

		/// Train one datum using the Apple graph-training path.
		float train_graph_network_datum(Network & net, float *x, float *y);

		/// Apply the graph-training optimizer update for @p net.
		void update_training_graph(Network & net);

		/// Synchronize graph-owned training state and optionally parameters back to host buffers.
		void sync_training_host_state(Network & net, bool sync_rolling_stats, bool sync_parameters, double *wait_ms);

		/** Prepare a dedicated inference network for validation.
		 *
		 * Pending training work is drained before parameters and batch-normalization
		 * state are copied. The destination layer storage must not alias @p training.
		 */
		bool prepare_validation_network(Network & training, Network & validation, std::string *reason = nullptr);

		/// Invalidate graph state and cached Apple training resources for a network.
		void invalidate_training_graph(Network * net);

		/// Return concise Apple training capability and state information for a network.
		TrainingGraphInfo build_training_graph_info(const Network * net);

	}
}
