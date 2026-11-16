#pragma once

/** @file
 * Apple backend facade used by Darknet training and inference paths.
 */

#include <cstddef>
#include <cstdint>
#include <string>

#include "apple_backend.hpp"
#include "apple_training_data.hpp"

#ifndef DARKNET_USE_MPS
#error "apple_mps.hpp is private to builds with Apple MPS enabled"
#endif

struct DarknetDetection;

namespace Darknet
{
	struct Layer;
	struct Network;
	struct NetworkState;

	namespace Apple
	{
		/// Return @p true if MPS is available and initialized.
		bool mps_is_available();

		/// Return @p true if the MPS runtime is usable.
		bool mps_runtime_available(std::string & reason);

		inline bool mps_runtime_available()
		{
			std::string reason;
			return mps_runtime_available(reason);
		}

		/// Return @p true if the requested training device count is supported.
		bool mps_training_device_count_supported(int ngpus, std::string & reason);

		/// Configure the MPS training graph for @p net.
		bool mps_configure_training_graph(Darknet::Network & net, std::string & reason);

		/// Return @p true if the MPS training graph is enabled for @p net.
		bool mps_training_graph_enabled(const Darknet::Network & net);

		/// Train one datum using the MPS training graph path.
		float mps_train_network_datum(Darknet::Network & net, const FloatBufferView & input, const FloatBufferView & truth);

		/// Apply MPS-owned network updates after a training step.
		void mps_update_network(Darknet::Network & net);

		/// Synchronize MPS-owned training state and optionally parameters back to host buffers.
		void mps_sync_training_host_state(Darknet::Network & net, const bool sync_rolling_stats, const bool sync_parameters);

		/// Materialize graph-owned state before the network enters host-backed inference.
		void mps_prepare_network_for_inference(Darknet::Network & net);

		/// Drain training work and copy parameters into a dedicated validation network.
		bool mps_prepare_validation_network(Darknet::Network & training, Darknet::Network & validation, std::string & reason);

		/// Invalidate cached MPS training graph state for @p net.
		void mps_invalidate_training_graph(Darknet::Network & net);

		/** Try to execute GEMM using MPS.
		 *
		 * @returns @p true if MPS handled the operation.
		 */
		bool mps_gemm(int TA, int TB, int M, int N, int K, float ALPHA,
			float *A, int lda,
			float *B, int ldb,
			float BETA,
			float *C, int ldc);


		/** Try to execute convolution forward using MPS.
		 *
		 * @param [in] l Convolution layer to execute.
		 * @param [in] prev Previous layer, or @p nullptr when no previous layer is supplied.
		 * @param [in] input Host input data when not using a previous layer's deferred GPU output.
		 * @param [out] output Host output storage, populated when readback is not deferred.
		 * @param [in] defer_readback If @p true, output may remain on GPU for a subsequent MPS layer.
		 * @param [out] activation_applied Set to @p true if MPS applied the activation.
		 * @param [out] reason Fallback reason string when returning @p false.
		 */
		bool mps_convolution_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, bool defer_readback, bool *activation_applied, std::string & reason);

		static inline bool mps_convolution_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, std::string & reason)
		{
			return mps_convolution_forward(l, prev, input, output, false, nullptr, reason);
		}

		static inline bool mps_convolution_forward(const Darknet::Layer & l, const float *input, float *output, std::string & reason)
		{
			return mps_convolution_forward(l, nullptr, input, output, false, nullptr, reason);
		}

		static inline bool mps_convolution_forward(const Darknet::Layer & l, const float *input, float *output)
		{
			std::string reason;
			return mps_convolution_forward(l, nullptr, input, output, false, nullptr, reason);
		}

		/// Try to execute maxpool forward using MPS.
		bool mps_maxpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, bool defer_readback, std::string & reason);

		/// Try to execute avgpool forward using MPS.
		bool mps_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, bool defer_readback, std::string & reason);

		/// Try to execute global avgpool forward using MPS.
		bool mps_global_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, bool defer_readback, std::string & reason);

		/// Try to execute shortcut add using MPS.
		bool mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const Darknet::Layer *from,
			const float *input, float *output, bool defer_readback, bool *activation_applied, std::string & reason);

		/// Try to concatenate route inputs using MPS.
		bool mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
			float *output, bool defer_readback, std::string & reason);

		/// Try to upsample using a Metal kernel on GPU.
		bool mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, bool defer_readback, std::string & reason);

		/// Try to execute reorg using a Metal kernel on GPU.
		bool mps_reorg_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, bool defer_readback, std::string & reason);

		/// Try to execute connected layer forward using MPS, including batchnorm and activation.
		bool mps_connected_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, bool defer_readback, bool *activation_applied, std::string & reason);

		/// Try to execute softmax on GPU.
		bool mps_softmax_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, std::string & reason);

		static inline bool mps_maxpool_forward(const Darknet::Layer & l, const float *input, float *output)
		{
			std::string reason;
			return mps_maxpool_forward(l, nullptr, input, output, false, reason);
		}

		static inline bool mps_avgpool_forward(const Darknet::Layer & l, const float *input, float *output)
		{
			std::string reason;
			return mps_avgpool_forward(l, nullptr, input, output, false, reason);
		}

		static inline bool mps_global_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, std::string & reason)
		{
			return mps_global_avgpool_forward(l, prev, input, output, false, reason);
		}

		static inline bool mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *from,
			const float *input, float *output, std::string & reason)
		{
			return mps_shortcut_forward(l, nullptr, from, input, output, false, nullptr, reason);
		}

		static inline bool mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
			float *output, std::string & reason)
		{
			return mps_route_forward(l, net, output, false, reason);
		}

		static inline bool mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, std::string & reason)
		{
			return mps_upsample_forward(l, prev, input, output, false, reason);
		}

		static inline bool mps_reorg_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, std::string & reason)
		{
			return mps_reorg_forward(l, prev, input, output, false, reason);
		}

		static inline bool mps_connected_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
			const float *input, float *output, std::string & reason)
		{
			return mps_connected_forward(l, prev, input, output, false, nullptr, reason);
		}


		/// Return @p true if @p l can run through MPS for the requested mode.
		bool mps_layer_can_run(const Darknet::Layer & l, const bool train);

		/// Return @p true if this layer currently owns a deferred GPU output.
		bool mps_is_output_deferred(const Darknet::Layer *layer);

		/** Materialize a deferred GPU output into host memory.
		 *
		 * This is a no-op when @p layer has no deferred output.  A materialization
		 * failure is fatal because continuing would expose stale host memory as the
		 * current inference result.
		 */
		void mps_materialize_output(const Darknet::Layer *layer, float *output = nullptr);

		/** Materialize a deferred layer output into its host buffer.
		 *
		 * These compatibility names keep the Apple integration points in generic
		 * Darknet code stable.  Both use the same fail-fast implementation as
		 * mps_materialize_output().
		 */
		void mps_flush_deferred_output(const Darknet::Layer *layer);
		void mps_flush_output_if_needed(const Darknet::Layer *layer, float *output);

		/// Return the previous layer visible to MPS for this network state.
		const Darknet::Layer *mps_prev_layer(const Darknet::NetworkState &state);

		/// Return @p true when the current forward pass should defer output readback.
		bool mps_should_defer_readback(const Darknet::NetworkState &state);

		/// Invalidate cached MPS resources for a layer.
		void mps_invalidate_layer_cache(const Darknet::Layer *layer);

		/// Mark layer parameters as updated from host memory.
		void mps_notify_layer_parameters_updated_cpu(const Darknet::Layer *layer);

		/// Mark layer parameters as updated from GPU memory.
		void mps_notify_layer_parameters_updated_gpu(const Darknet::Layer *layer);

		/// Mark layer optimizer state as updated from host memory.
		void mps_notify_layer_optimizer_state_updated_cpu(const Darknet::Layer *layer);

		/// Mark layer optimizer state as updated from GPU memory.
		void mps_notify_layer_optimizer_state_updated_gpu(const Darknet::Layer *layer);

		/// Mark batch-normalization running state as updated from host memory.
		void mps_notify_layer_running_stats_updated_cpu(const Darknet::Layer *layer);

		/// Mark batch-normalization running state as updated from GPU memory.
		void mps_notify_layer_running_stats_updated_gpu(const Darknet::Layer *layer);

		/// Display Apple MPS runtime information.
		void show_mps_info();
	}
}
