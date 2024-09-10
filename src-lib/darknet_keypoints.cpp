#include "darknet_internal.hpp"
#include "darknet_keypoints.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	/** @{ The first 17 keypoints (zero through 16) used in Darknet/YOLO are the same as those defined by MSCOCO
	 * keypoints.  The only difference is the addition of @p "person" at the end to facilitate top-down grouping.
	 */
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
	constexpr int KP_PERSON		= 17; // not a real keypoint
	constexpr int KP_MAX		= 18; // not a real keypoint

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
		"right ankle",
		"person"
	};
	/// @}

	/** Define all the simple skeleton lines that need to be drawn.  Some special cases are handled manually
	 * in @ref Keypoints::annotate().
	 *
	 * @since 2024-09-03
	 */
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
		throw std::logic_error("expected the neural network to have 18 classes (MSCOCO keypoints + person) but network has " + std::to_string(net->details->class_names.size()) + " classes");
	}

	return;
}


Darknet::Keypoints::~Keypoints()
{
	return;
}


Darknet::VStr Darknet::Keypoints::names()
{
	return KPNames;
}


Darknet::Skeletons Darknet::Keypoints::create_skeletons(const Predictions & predictions)
{
	TAT(TATPARMS);

	// remember where all the skeletons are, because we're going to need this often
	SInt people_prediction_indexes;

	// we want to place all the predictions into skeletons in order from most likely to least likely
	std::multimap<float, int> sorted_prediction_indexes;
	for (size_t prediction_idx = 0; prediction_idx < predictions.size(); prediction_idx ++)
	{
		const auto & prediction = predictions.at(prediction_idx);
		const auto best_class = prediction.best_class;
		if (best_class == KP_PERSON)
		{
			people_prediction_indexes.insert(prediction_idx);
		}
		else
		{
			sorted_prediction_indexes.insert({prediction.prob.at(best_class), prediction_idx});
		}
	}

	// first we pre-allocate skeletons for every person detected in the image
	Skeletons skeletons;
	for (const auto prediction_idx : people_prediction_indexes)
	{
		Skeleton skeleton(KP_MAX, -1); // set all keypoints to -1
		skeleton[KP_PERSON] = prediction_idx;
		skeletons.push_back(skeleton);
	}

	if (people_prediction_indexes.empty() and predictions.size() > 0)
	{
		// we failed to detect a person, but we have keypoints, so create a skeleton anyway
		Skeleton skeleton(KP_MAX, -1);
		skeletons.push_back(skeleton);
	}

	for (auto iter = sorted_prediction_indexes.rbegin(); iter != sorted_prediction_indexes.rend(); iter ++)
	{
		const auto prediction_idx = iter->second;

		const auto & pred = predictions.at(prediction_idx);
		const int best_class = pred.best_class;
//		std::cout << "-> looking for skeleton for this keypoint (best_class=" << best_class << ", " << KPNames[best_class] << ") at idx=" << prediction_idx << ": " << pred << std::endl;

		if (skeletons.size() == 1 and skeletons[0][best_class] == -1)
		{
			// simple case, we only have 1 skeleton and this keypoint hasn't been seen yet
			skeletons[0][best_class] = prediction_idx;
			continue;
		}

		// if we get here we either have multiple skeletons, or duplicate predictions

		if (skeletons.size() == 1)
		{
//			std::cout << "dropping prediction since it is a duplicate: " << pred << std::endl;
			continue;
		}

		// find the best skeleton where this prediction fits; key=IoU, val=skeleton index
		std::multimap<float, int> mm;

		for (size_t skeleton_idx = 0; skeleton_idx < skeletons.size(); skeleton_idx ++)
		{
			auto & skeleton = skeletons.at(skeleton_idx);
			const auto & person_rect = predictions.at(skeleton[KP_PERSON]).rect;

			const auto iou = Darknet::iou(person_rect, pred.rect);
			mm.insert({iou, skeleton_idx});
		}

		// now see which is the most likely skeleton that needs one of these keypoints
		for (auto iter = mm.rbegin(); iter != mm.rend(); iter ++)
		{
			if (iter->first <= 0.0f)
			{
				// we need to drop this prediction, we did not find where it is needed
//				std::cout << "dropping prediction, no skeleton found which needed: " << pred << std::endl;
				break;
			}

			const int skeleton_idx = iter->second;
			auto & skeleton = skeletons[skeleton_idx];
			if (skeleton[best_class] == -1)
			{
				// looks like we found a valid skeleton that needs this part!
				skeleton[best_class] = prediction_idx;
				break;
			}
		}
	}

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

	for (size_t skeleton_idx = 0; skeleton_idx < skeletons.size(); skeleton_idx ++)
	{
		// each skeleton gets a new colour based on the class colours generated by Darknet (should be 18 of them)
		const auto & skeleton = skeletons[skeleton_idx];
		const auto & colour = net->details->class_colours[skeleton_idx % net->details->class_colours.size()];

		std::vector<cv::Point> points;

		for (size_t keypoint_idx = 0; keypoint_idx < skeleton.size(); keypoint_idx ++)
		{
			const int prediction_idx = skeleton.at(keypoint_idx);
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

			if (keypoint_idx == KP_PERSON)
			{
				// draw a rectangle around the entire person
//				cv::rectangle(mat, prediction.rect, colour, 1, line_type);
			}
			else
			{
				// normal keypoint joint
				cv::circle(mat, p, 10, colour, cv::FILLED, line_type);
			}
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

#if 0
			// both shoulders, but only 1 hip
			if (skeleton[KP_L_HIP] == -1 and skeleton[KP_R_HIP] >= 0)
			{
				cv::line(mat, mid_shoulder, points[KP_R_HIP], colour, 5, line_type);
			}

			if (skeleton[KP_R_HIP] == -1 and skeleton[KP_L_HIP] >= 0)
			{
				cv::line(mat, mid_shoulder, points[KP_L_HIP], colour, 5, line_type);
			}
#endif
		}

#if 0
		// both hips, but only 1 shoulder
		if (mid_hip.x >= 0 and
			mid_hip.y >= 0)
		{
			if (skeleton[KP_L_SHOULDER] == -1 and skeleton[KP_R_SHOULDER] >= 0)
			{
				cv::line(mat, mid_hip, points[KP_R_SHOULDER], colour, 5, line_type);
			}
			if (skeleton[KP_R_SHOULDER] == -1 and skeleton[KP_L_SHOULDER] >= 0)
			{
				cv::line(mat, mid_hip, points[KP_L_SHOULDER], colour, 5, line_type);
			}
		}
#endif
	}

#if 0
	if (cfg_and_state.is_trace)
	{
		// for DEBUG purpose only -- now that the lines and circles are done, draw the text for each keypoint

		for (const auto & skeleton : skeletons)
		{
			for (const int & prediction_idx : skeleton)
			{
				if (prediction_idx < 0 or prediction_idx >= predictions.size())
				{
					// unknown/unasigned keypoint
					continue;
				}

				const auto & prediction = predictions[prediction_idx];

				if (prediction.best_class == KP_PERSON)
				{
					continue;
				}

				const int x = std::round(w * prediction.normalized_point.x);
				const int y = std::round(h * prediction.normalized_point.y);

				const cv::Point p(x, y);
				cv::putText(mat, KPNames[prediction.best_class], p, net->details->cv_font_face, net->details->cv_font_scale, {255, 255, 255	}, net->details->cv_font_thickness + 2, line_type);
				cv::putText(mat, KPNames[prediction.best_class], p, net->details->cv_font_face, net->details->cv_font_scale, {0, 0, 0		}, net->details->cv_font_thickness + 0, line_type);
			}
		}
	}
#endif

	return mat;
}


cv::Mat Darknet::Keypoints::annotate(const Predictions & predictions, cv::Mat mat)
{
	TAT(TATPARMS);

	auto skeletons = create_skeletons(predictions);

	return annotate(predictions, skeletons, mat);
}
