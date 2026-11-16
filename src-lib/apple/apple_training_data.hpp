#pragma once

/** @file
 * Borrowed host data used by Apple training. C++17 has no standard span type.
 */

#include <cstddef>
#include <vector>
#include <string>

namespace Darknet
{
	struct Network;

	namespace Apple
	{
		/** A view of caller-owned floats; constructing a view never copies data.
		 * The caller must keep the storage alive and unchanged until training
		 * returns. Apple training finishes reading these buffers before returning.
		 * Mutable access is retained for the existing NetworkState interface.
		 */
		class FloatBufferView
		{
		public:
			FloatBufferView(float *data, const size_t count) : data_(data), count_(count) {}
			explicit FloatBufferView(std::vector<float> & values) : FloatBufferView(values.data(), values.size()) {}

			float *data() const { return data_; }
			size_t size() const { return count_; }

		private:
			float *data_;
			size_t count_;
		};
		/// Check data extents before graph execution or truth-buffer reads.
		bool validate_training_data_views(const Network & net, const FloatBufferView & input,
			const FloatBufferView & truth, std::string & reason);
	}
}
