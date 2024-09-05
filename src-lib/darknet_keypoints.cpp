#include "darknet_internal.hpp"
#include "darknet_keypoints.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	// The 17 keypoints used in Darknet/YOLO are the same as those defined by MSCOCO.

	constexpr int KP_NOSE		= 0;
	constexpr int KP_L_EYE		= 1;
	constexpr int KP_R_EYE		= 2;
	constexpr int KP_L_EAR		= 3;
	constexpr int KP_R_EAR		= 4;
	constexpr int KP_L_SHOULDER	= 5;
	constexpr int KP_R_SHOULDER	= 6;
	constexpr int KP_L_ELBOW	= 7;
	constexpr int KP_R_ELBOW	= 8;
	constexpr int KP_L_WRIST	= 9;
	constexpr int KP_R_WRIST	= 10;
	constexpr int KP_L_HIP		= 11;
	constexpr int KP_R_HIP		= 12;
	constexpr int KP_L_KNEE		= 13;
	constexpr int KP_R_KNEE		= 14;
	constexpr int KP_L_ANKLE	= 15;
	constexpr int KP_R_ANKLE	= 16;
	constexpr int KP_MAX		= 17;

	const Darknet::VStr KPNames =
	{
		"nose",
		"left eye",
		"right eye",
		"left ear",
		"right ear",
		"left shoulder",
		"right shoulder",
		"left elbow",
		"right elbow",
		"left wrist",
		"right wrist",
		"left hip",
		"right hip",
		"left knee",
		"right knee",
		"left ankle",
		"right ankle"
	};

	// from bottom-up, define all the (easy) points that need to be drawn; some special cases are handled manually
	const std::map<int, int> SkeletonPoints =
	{
		// left leg
		{KP_L_ANKLE		, KP_L_KNEE		},
		{KP_L_KNEE		, KP_L_HIP		},
		// right leg
		{KP_R_ANKLE		, KP_R_KNEE		},
		{KP_R_KNEE		, KP_R_HIP		},
		// hips
		{KP_L_HIP		, KP_R_HIP		},
		// shoulders
		{KP_L_SHOULDER	, KP_R_SHOULDER	},
		// left arm
		{KP_L_WRIST		, KP_L_ELBOW	},
		{KP_L_ELBOW		, KP_L_SHOULDER	},
		// right arm
		{KP_R_WRIST		, KP_R_ELBOW	},
		{KP_R_ELBOW		, KP_R_SHOULDER	},
		// left side of head
		{KP_L_EAR		, KP_L_EYE		},
		{KP_L_EYE		, KP_NOSE		},
		// right side of head
		{KP_R_EAR		, KP_R_EYE		},
		{KP_R_EYE		, KP_NOSE		},
	};
}


Darknet::Keypoints::Keypoints(const Darknet::NetworkPtr ptr) :
	network_ptr(ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(network_ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot instantiate Darknet::Keypoints without a network pointer");
	}

	// MSCOCO keypoints format is the only one currently supported
	if (net->details->class_names.size() != KP_MAX)
	{
		throw std::logic_error("expected the neural network to have 17 classes (MSCOCO keypoints) but network has " + std::to_string(net->details->class_names.size()) + " classes");
	}

	return;
}


Darknet::Skeletons Darknet::Keypoints::create_skeletons(const Predictions & predictions)
{
	TAT(TATPARMS);

	Skeletons skeletons;

	for (int idx = 0; idx < predictions.size(); idx ++)
	{
		// find a skeleton where we can record this index, and if needed add a new skeleton to the vector
		const int best_class = predictions[idx].best_class;
		bool need_a_new_skeleton = true;

		// look for duplicate keypoint entries in the skeletons we already have
		for (auto & skeleton : skeletons)
		{
			if (skeleton[best_class] != -1)
			{
				// if get here then we already have a prediction for this keypoint...but should they be merged?
				const auto & lhs = predictions[skeleton[best_class]];
				const auto & rhs = predictions[idx];

				const auto x_diff = std::abs(1.0f - lhs.normalized_point.x / rhs.normalized_point.x);
				const auto y_diff = std::abs(1.0f - lhs.normalized_point.y / rhs.normalized_point.y);

#if 0
				std::cout
					<< "compare these two for class=" << best_class << " (" << KPNames[best_class] << ")" << std::endl
					<< "-> lhs idx #" << skeleton[best_class]	<< ": class " << lhs << std::endl
					<< "-> rhs idx #" << idx					<< ": class " << rhs << std::endl
					<< "-> x_diff=" << x_diff << " y_diff=" << y_diff << std::endl;
#endif

				if (x_diff < 0.1f and y_diff < 0.1f)
				{
					// these are the same keypoint -- keep the one with the highest confidence
//					std::cout << "merging these 2 entries:  old idx=" << skeleton[best_class] << " and new idx=" << idx << std::endl;
					if (lhs.prob.at(best_class) < rhs.prob.at(best_class))
					{
						skeleton[best_class] = idx;
					}

					// the keypoints have been merged
					need_a_new_skeleton = false;
					break;
				}

				// if we get here we need to keep looking for a skeleton that needs this keypoint
				continue;
			}

			// if we get here we found a skeleton that does not yet have this keypoint

			skeleton[best_class] = idx;
			need_a_new_skeleton = false;
			break;
		}

		if (need_a_new_skeleton)
		{
			// if we get here, then we have no place to store this index, so add a new skeleton
			Skeleton skeleton(KP_MAX, -1);
			skeleton[best_class] = idx;
			skeletons.push_back(skeleton);
		}
	}

	// now if we had support for multiple skeletons, we'd go through and swap the indexes around until things made sense

	#if 0
	// Note that multiple skeletons is NOT YET SUPPORTED!  No attempt is made
	// at comparing the parts to determine which one goes with which skeleton.
	#endif

	if (cfg_and_state.is_trace)
	{
		for (const auto & skeleton : skeletons)
		{
			std::cout << "next skeleton:" << std::endl;
			for (size_t idx = 0; idx < skeleton.size(); idx ++)
			{
				const int prediction_idx = skeleton[idx];

				std::cout << "-> " << idx << " (" << KPNames[idx] << ") = prediction idx " << prediction_idx;
				if (prediction_idx >= 0)
				{
					std::cout << " => " << predictions[prediction_idx];
				}
				std::cout << std::endl;
			}
		}
	}

	return skeletons;
}


cv::Mat Darknet::Keypoints::annotate(const Predictions & predictions, const Darknet::Skeletons & skeletons, cv::Mat mat)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(network_ptr);
	const auto line_type = net->details->cv_line_type;

	const float w = mat.cols;
	const float h = mat.rows;

	// we can have zero or many skeletons in this image
	for (size_t idx = 0; idx < skeletons.size(); idx ++)
	{
		// each skeleton gets a new colour based on the class colours generated by Darknet (should be 17 of them)
		const auto & skeleton = skeletons[idx];
		const auto & colour = net->details->class_colours[idx % net->details->class_colours.size()];

		std::vector<cv::Point> points;

		for (const int & prediction_idx : skeleton)
		{
			if (prediction_idx < 0 or prediction_idx >= predictions.size())
			{
				// a negative index is invalid and means we don't have that part of the skeleton
				points.push_back({-1, -1});
				continue;
			}

			const auto & prediction = predictions[prediction_idx];

			const int x = std::round(w * prediction.normalized_point.x);
			const int y = std::round(h * prediction.normalized_point.y);

			const cv::Point p(x, y);
			points.push_back(p);
			cv::circle(mat, p, 10, colour, cv::FILLED, line_type);
		}

		// don't draw the lines between the dots if we have multiple skeletons
		if (skeletons.size() > 1)
		{
			continue;
		}

		for (const auto & [k, v] : SkeletonPoints)
		{
			const auto & src = points[k];
			const auto & dst = points[v];

			if (src.x >= 0 and
				src.y >= 0 and
				dst.x >= 0 and
				dst.y >= 0)
			{
				cv::line(mat, src, dst, colour, 5, line_type);
			}
		}

		// handle the special case:  mid-hip and mid-shoulder, for which we don't have a prediction point
		cv::Point mid_shoulder(-1, -1);
		cv::Point mid_hip(-1, -1);

		if (skeleton[KP_L_SHOULDER] >= 0 and skeleton[KP_R_SHOULDER] >= 0)
		{
			const auto & src = predictions[skeleton[KP_L_SHOULDER]].normalized_point;
			const auto & dst = predictions[skeleton[KP_R_SHOULDER]].normalized_point;
			mid_shoulder.x = std::round(w * (src.x + dst.x) / 2.0f);
			mid_shoulder.y = std::round(h * (src.y + dst.y) / 2.0f);
		}

		if (skeleton[KP_L_HIP] >= 0 and skeleton[KP_R_HIP] >= 0)
		{
			const auto & src = predictions[skeleton[KP_L_HIP]].normalized_point;
			const auto & dst = predictions[skeleton[KP_R_HIP]].normalized_point;
			mid_hip.x = std::round(w * (src.x + dst.x) / 2.0f);
			mid_hip.y = std::round(h * (src.y + dst.y) / 2.0f);
		}

		if (mid_shoulder.x >= 0 and
			mid_shoulder.y >= 0)
		{
			if (mid_hip.x >= 0 and
				mid_hip.y >= 0)
			{
				cv::line(mat, mid_hip, mid_shoulder, colour, 5, line_type);
			}

			if (skeleton[KP_NOSE] >= 0)
			{
				cv::line(mat, mid_shoulder, points[KP_NOSE], colour, 5, line_type);
			}
		}
	}

	return mat;
}


cv::Mat Darknet::Keypoints::annotate(const Predictions & predictions, cv::Mat mat)
{
	TAT(TATPARMS);

	auto skeletons = create_skeletons(predictions);

	return annotate(predictions, skeletons, mat);
}
