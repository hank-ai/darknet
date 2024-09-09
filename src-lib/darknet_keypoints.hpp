/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * This file defines @ref Darknet::Keypoints and related functions.
 */


#include "darknet.hpp"


namespace Darknet
{
	using Skeleton = VInt;
	using Skeletons = std::vector<Skeleton>;

	/** The @p Keypoints class works with %Darknet's V3 API.
	 *
	 * The only format currently supported is the MSCOCO-style keypoint classes
	 * with an extra "person" class appended for top-down grouping.  The classes
	 * are defined as follows:
	 *
	 * 0:	nose
	 * 1:	left eye
	 * 2:	right eye
	 * 3:	left ear
	 * 4:	right ear
	 * 5:	left shoulder
	 * 6:	right shoulder
	 * 7:	left elbow
	 * 8:	right elbow
	 * 9:	left wrist
	 * 10:	right wrist
	 * 11:	left hip
	 * 12:	right hip
	 * 13:	left knee
	 * 14:	right knee
	 * 15:	left ankle
	 * 16:	right ankle
	 * 17:	person
	 *
	 * @since 2024-09-03
	 */
	class Keypoints final
	{
		public:

			Keypoints() = delete;

			/** Constructor needs a neural network pointer.  @see @ref Darknet::load_neural_network()
			 *
			 * @since 2024-09-03
			 */
			Keypoints(const Darknet::NetworkPtr ptr);

			/// Destructor.
			~Keypoints();

			/** Return the set of names for the classes in @p Keypoints.
			 *
			 * @since 2024-09-07
			 */
			VStr names();

			/** Looks through the prediction results and attempts to organize each person into @ref Skeleton.  A skeleton will
			 * always contain exactly 18 indexes.  If a skeleton does not have an entry for a body part, then the index will be
			 * set to @p -1.  Otherwise, any other values is interpreted as an index into @p predictions.
			 *
			 * @since 2024-09-03
			 */
			Skeletons create_skeletons(const Predictions & predictions);

			/** Draw the skeletons onto the given image.
			 *
			 * @since 2024-09-03
			 */
			cv::Mat annotate(const Predictions & predictions, const Skeletons & skeletons, cv::Mat mat);

			/** Similar to the other @ref annotate() method, but automatically calls @ref create_skeletons().
			 *
			 * @since 2024-09-03
			 */
			cv::Mat annotate(const Predictions & predictions, cv::Mat mat);

		private:

			const Darknet::NetworkPtr network_ptr;
	};
}
