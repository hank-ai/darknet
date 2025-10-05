/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Some of the common Darknet/YOLO enums, and methods to convert between them and @p std::string.
 */

#include "darknet_internal.hpp"


namespace Darknet
{
	/** This is the new C++ version of what used to be called @p LAYER_TYPE in the old @p C code.
	 * @see @ref Darknet::all_names_and_layers()
	 * @see @ref Darknet::get_layer_type_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class ELayerType
	{
		CONVOLUTIONAL								,	///< this is used everywhere
		CONNECTED									,	///< used in 3 rarely used configurations
		MAXPOOL										,	///< used often
		LOCAL_AVGPOOL								,	///< only used in yolov4-tiny_contrastive.cfg -- obsolete?
		SOFTMAX										,	///< used in several non-YOLO configs
		DROPOUT										,	///< used in 3 non-YOLO configs
		ROUTE										,	///< used often
		COST										,	///< used in several non-YOLO configs
		AVGPOOL										,	///< used in several non-YOLO configs
		SHORTCUT									,	///< used almost everywhere
		SCALE_CHANNELS								,	///< only used in 2 configs
		SAM											,	///< only used in 3 configs
		RNN											,	///< only used in rnn.train.cfg -- obsolete?
		LSTM										,	///< only used in lstm.train.cfg -- obsolete?
		CRNN										,	///< only used in 2 non-YOLO configs
		NETWORK										,	///< used in every config
		REGION										,	///< only used in tiny-yolo_xnor.cfg (which is NOT a YOLO config)
		YOLO										,	///< used often
		YOLO_BDP									,	///< oriented bounding boxes with 6 parameters (x,y,w,h,fx,fy)
		GAUSSIAN_YOLO								,	///< only used in Gaussian_yolov3_BDD.cfg
		REORG			/* aka "3D" */				,	///< only used in yolov4-sam-mish-csp-reorg-bfm.cfg
		UPSAMPLE									,	///< used often, does downsampling instead if l.reverse=1
		EMPTY			/* aka "SILENCE" */			,	///< *UNUSED*
		BLANK										,	///< *UNUSED*
		CONTRASTIVE									,	///< only used in yolov4-tiny_contrastive.cfg
		LAYER_LAST_IDX	= ELayerType::CONTRASTIVE	,	///< point to the last-used idx
	};

	/// @{ Convert between names and Darknet/YOLO layer types.
	using NamesAndLayers = std::map<std::string, ELayerType>;
	const NamesAndLayers & all_names_and_layers();
	ELayerType get_layer_type_from_name(const std::string & name);
	std::string to_string(const ELayerType type);
	/// @}

	/** The plan is to eventually remove @ref ACTIVATION completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_activations()
	 * @see @ref Darknet::get_activation_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class EActivation
	{
		// Please keep the old C and the new C++ enums in sync!
		LOGISTIC					= ACTIVATION::LOGISTIC				,
		RELU						= ACTIVATION::RELU					,
		RELU6						= ACTIVATION::RELU6					,
		RELIE						= ACTIVATION::RELIE					,
		LINEAR						= ACTIVATION::LINEAR				,
		RAMP						= ACTIVATION::RAMP					,
		TANH						= ACTIVATION::TANH					,
		PLSE						= ACTIVATION::PLSE					,
		REVLEAKY					= ACTIVATION::REVLEAKY				,
		LEAKY						= ACTIVATION::LEAKY					,
		ELU							= ACTIVATION::ELU					,
		LOGGY						= ACTIVATION::LOGGY					,
		STAIR						= ACTIVATION::STAIR					,
		HARDTAN						= ACTIVATION::HARDTAN				,
		LHTAN						= ACTIVATION::LHTAN					,
		SELU						= ACTIVATION::SELU					,
		GELU						= ACTIVATION::GELU					,
		SWISH						= ACTIVATION::SWISH					,
		MISH						= ACTIVATION::MISH					,
		HARD_MISH					= ACTIVATION::HARD_MISH				,
		NORM_CHAN					= ACTIVATION::NORM_CHAN				,
		NORM_CHAN_SOFTMAX			= ACTIVATION::NORM_CHAN_SOFTMAX		,
		NORM_CHAN_SOFTMAX_MAXVAL	= ACTIVATION::NORM_CHAN_SOFTMAX_MAXVAL
	};

	/// @{ Convert between names and activation types.
	using NamesAndActivationTypes = std::map<std::string, EActivation>;
	const NamesAndActivationTypes & all_names_and_activations();
	EActivation get_activation_from_name(const std::string & name);
	std::string to_string(const EActivation activation);
	/// @}

	/** The plan is to eventually remove @ref learning_rate_policy completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_learning_rate_policies()
	 * @see @ref Darknet::get_learning_rate_policy_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class ELearningRatePolicy
	{
		// Please keep the old C and the new C++ enums in sync!
		CONSTANT	= learning_rate_policy::CONSTANT,
		STEP		= learning_rate_policy::STEP	,
		EXP			= learning_rate_policy::EXP		,
		POLY		= learning_rate_policy::POLY	,
		STEPS		= learning_rate_policy::STEPS	,
		SIG			= learning_rate_policy::SIG		,
		RANDOM		= learning_rate_policy::RANDOM	,
		SGDR		= learning_rate_policy::SGDR	,
	};

	/// @{ Convert between names and learning rate policies.
	using NamesAndLearningRatePolicies = std::map<std::string, ELearningRatePolicy>;
	const NamesAndLearningRatePolicies & all_names_and_learning_rate_policies();
	ELearningRatePolicy get_learning_rate_policy_from_name(const std::string & name);
	std::string to_string(const ELearningRatePolicy policy);
	/// @}

	/** The plan is to eventually remove @ref IOU_LOSS completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_IoU_loss()
	 * @see @ref Darknet::get_IoU_loss_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class EIoULoss
	{
		// Please keep the old C and the new C++ enums in sync!
		IOU		= IOU_LOSS::IOU,
		GIOU	= IOU_LOSS::GIOU,
		MSE		= IOU_LOSS::MSE,
		DIOU	= IOU_LOSS::DIOU,
		CIOU	= IOU_LOSS::CIOU,
		RIOU	= IOU_LOSS::RIOU,
	};

	/// @{ Convert between names and IoU loss types.
	using NamesAndIoULoss = std::map<std::string, EIoULoss>;
	const NamesAndIoULoss & all_names_and_IoU_loss();
	EIoULoss get_IoU_loss_from_name(const std::string & name);
	std::string to_string(const EIoULoss loss);
	/// @}

	/** The plan is to eventually remove @ref NMS_KIND completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_NMS_kind()
	 * @see @ref Darknet::get_NMS_kind_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class ENMSKind
	{
		DEFAULT_NMS	= NMS_KIND::DEFAULT_NMS	,
		GREEDY_NMS	= NMS_KIND::GREEDY_NMS	,
		DIOU_NMS	= NMS_KIND::DIOU_NMS	,
		CORNERS_NMS	= NMS_KIND::CORNERS_NMS	, // gaussian yolo
	};

	/// @{ Convert between names and NMS kind.
	using NamesAndNMSKind = std::map<std::string, ENMSKind>;
	const NamesAndNMSKind & all_names_and_NMS_kind();
	ENMSKind get_NMS_kind_from_name(const std::string & name);
	std::string to_string(const ENMSKind nms_kind);
	/// @}

	/** The plan is to eventually remove @ref WEIGHTS_TYPE_T completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_weights_types()
	 * @see @ref Darknet::get_weights_type_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class EWeightsType
	{
		NO_WEIGHTS	= WEIGHTS_TYPE_T::NO_WEIGHTS,
		PER_FEATURE	= WEIGHTS_TYPE_T::PER_FEATURE,
		PER_CHANNEL	= WEIGHTS_TYPE_T::PER_CHANNEL,
	};

	/// @{ Convert between names and weights types.
	using NamesAndWeightsType = std::map<std::string, EWeightsType>;
	const NamesAndWeightsType & all_names_and_weights_types();
	EWeightsType get_weights_type_from_name(const std::string & name);
	std::string to_string(const EWeightsType type);
	/// @}

	/** The plan is to eventually remove @ref WEIGHTS_NORMALIZATION_T completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_weights_normalization()
	 * @see @ref Darknet::get_weights_normalization_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class EWeightsNormalization
	{
		NO_NORMALIZATION		= WEIGHTS_NORMALIZATION_T::NO_NORMALIZATION,
		RELU_NORMALIZATION		= WEIGHTS_NORMALIZATION_T::RELU_NORMALIZATION,
		SOFTMAX_NORMALIZATION	= WEIGHTS_NORMALIZATION_T::SOFTMAX_NORMALIZATION,
	};

	/// @{ Convert between names and weights normalization.
	using NamesAndWeightsNormalization = std::map<std::string, EWeightsNormalization>;
	const NamesAndWeightsNormalization & all_names_and_weights_normalization();
	EWeightsNormalization get_weights_normalization_from_name(const std::string & name);
	std::string to_string(const EWeightsNormalization normalization);
	/// @}

	/** The plan is to eventually remove @ref COST_TYPE completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_cost_types()
	 * @see @ref Darknet::get_cost_types_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class ECostType
	{
		SSE			= COST_TYPE::SSE,
		MASKED		= COST_TYPE::MASKED,
//		L1			= COST_TYPE::L1,		UNUSED?
//		SEG			= COST_TYPE::SEG		UNUSED?
		SMOOTH		= COST_TYPE::SMOOTH,
//		WGAN		= COST_TYPE::WGAN,		UNUSED?
	};

	/// @{ Convert between names and cost types.
	using NamesAndCostTypes = std::map<std::string, ECostType>;
	const NamesAndCostTypes & all_names_and_cost_types();
	ECostType get_cost_types_from_name(const std::string & name);
	std::string to_string(const ECostType type);
	/// @}

	/** The plan is to eventually remove @ref YOLO_POINT completely once we fully switch over to C++.
	 * @see @ref Darknet::all_names_and_yolo_point_types()
	 * @see @ref Darknet::get_yolo_point_types_from_name()
	 * @see @ref Darknet::to_string()
	 */
	enum class EYoloPoint
	{
		YOLO_CENTER			= YOLO_POINT::YOLO_CENTER,
		YOLO_LEFT_TOP		= YOLO_POINT::YOLO_LEFT_TOP,
		YOLO_RIGHT_BOTTOM	= YOLO_POINT::YOLO_RIGHT_BOTTOM,
	};

	/// @{ Convert between names and YOLO point types.
	using NamesAndYoloPointTypes = std::map<std::string, EYoloPoint>;
	const NamesAndYoloPointTypes & all_names_and_yolo_point_types();
	EYoloPoint get_yolo_point_types_from_name(const std::string & name);
	std::string to_string(const EYoloPoint type);
	/// @}
};
