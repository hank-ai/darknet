#pragma once

namespace Darknet::Apple
{
	struct TrainingSgdStep
	{
		int count = 0;
		float rate_over_batch = 0.0f;
		float momentum = 0.0f;
		float decay_term = 0.0f;
	};

	inline float training_sgd_rate_over_batch(const float learning_rate, const float learning_rate_scale, const int update_batch)
	{
		return update_batch > 0 ? (learning_rate * learning_rate_scale) / update_batch : 0.0f;
	}

	inline float training_sgd_decay_term(const float decay, const int update_batch, const bool apply_decay)
	{
		return apply_decay ? -decay * static_cast<float>(update_batch) : 0.0f;
	}

	inline bool training_update_gate_open(const int burnin_update,
		const int burn_in,
		const bool train_only_bn,
		const bool has_update,
		const int dont_update,
		const int iteration)
	{
		return not (burnin_update and burnin_update * burn_in > iteration) and
			not train_only_bn and has_update and dont_update < iteration;
	}

}
