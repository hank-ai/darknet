#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	/** The plan is to eventually remove LAYER_TYPE completely once we fully switch over to C++.
	 * @see @ref all_names_and_layers()
	 * @see @ref get_layer_from_name()
	 */
	enum class ELayerType
	{
		// Please keep the old C and the new C++ enums in sync!
		CONVOLUTIONAL	= LAYER_TYPE::CONVOLUTIONAL				,	///< this is used everywhere
		DECONVOLUTIONAL	= LAYER_TYPE::DECONVOLUTIONAL			,	///< *UNUSED*
		CONNECTED		= LAYER_TYPE::CONNECTED					,	///< used in 3 rarely used configurations
		MAXPOOL			= LAYER_TYPE::MAXPOOL					,	///< used often
		LOCAL_AVGPOOL	= LAYER_TYPE::LOCAL_AVGPOOL				,	///< only used in yolov4-tiny_contrastive.cfg -- obsolete?
		SOFTMAX			= LAYER_TYPE::SOFTMAX					,	///< used in several non-YOLO configs
		DETECTION		= LAYER_TYPE::DETECTION					,	///< *UNUSED*
		DROPOUT			= LAYER_TYPE::DROPOUT					,	///< used in 3 non-YOLO configs
		CROP			= LAYER_TYPE::CROP						,	///< *UNUSED*
		ROUTE			= LAYER_TYPE::ROUTE						,	///< used often
		COST			= LAYER_TYPE::COST						,	///< used in several non-YOLO configs
		NORMALIZATION	= LAYER_TYPE::NORMALIZATION				,	///< *UNUSED*
		AVGPOOL			= LAYER_TYPE::AVGPOOL					,	///< used in several non-YOLO configs
		LOCAL			= LAYER_TYPE::LOCAL						,	///< *UNUSED*
		SHORTCUT		= LAYER_TYPE::SHORTCUT					,	///< used almost everywhere
		SCALE_CHANNELS	= LAYER_TYPE::SCALE_CHANNELS			,	///< only used in 2 configs
		SAM				= LAYER_TYPE::SAM						,	///< only used in 3 configs
		ACTIVE			= LAYER_TYPE::ACTIVE					,	///< *UNUSED*
		RNN				= LAYER_TYPE::RNN						,	///< only used in rnn.train.cfg -- obsolete?
		GRU				= LAYER_TYPE::GRU						,	///< *UNUSED*
		LSTM			= LAYER_TYPE::LSTM						,	///< only used in lstm.train.cfg -- obsolete?
		CONV_LSTM		= LAYER_TYPE::CONV_LSTM					,	///< *UNUSED*
		HISTORY			= LAYER_TYPE::HISTORY					,	///< *UNUSED*
		CRNN			= LAYER_TYPE::CRNN						,	///< only used in 2 non-YOLO configs
		BATCHNORM		= LAYER_TYPE::BATCHNORM					,	///< *UNUSED*
		NETWORK			= LAYER_TYPE::NETWORK					,	///< used in every config
		XNOR			= LAYER_TYPE::XNOR						,	///< *UNUSED*
		REGION			= LAYER_TYPE::REGION					,	///< only used in tiny-yolo_xnor.cfg (which is NOT a YOLO config)
		YOLO			= LAYER_TYPE::YOLO						,	///< used often
		GAUSSIAN_YOLO	= LAYER_TYPE::GAUSSIAN_YOLO				,	///< only used in Gaussian_yolov3_BDD.cfg
		ISEG			= LAYER_TYPE::ISEG						,	///< *UNUSED*
		REORG			= LAYER_TYPE::REORG /* aka "3D" */		,	///< only used in yolov4-sam-mish-csp-reorg-bfm.cfg
		REORG_OLD		= LAYER_TYPE::REORG_OLD					,	///< *UNUSED*
		UPSAMPLE		= LAYER_TYPE::UPSAMPLE					,	///< used often, does downsampling instead if l.reverse=1
		LOGXENT			= LAYER_TYPE::LOGXENT					,	///< *UNUSED*
		L2NORM			= LAYER_TYPE::L2NORM					,	///< *UNUSED*
		EMPTY			= LAYER_TYPE::EMPTY /* aka "SILENCE" */	,	///< *UNUSED*
		BLANK			= LAYER_TYPE::BLANK						,	///< *UNUSED*
		CONTRASTIVE		= LAYER_TYPE::CONTRASTIVE				,	///< only used in yolov4-tiny_contrastive.cfg
		IMPLICIT		= LAYER_TYPE::IMPLICIT					,	///< *UNUSED*
	};

	/// @{ Convert between names and layer types.
	using NamesAndLayers = std::map<std::string, ELayerType>;
	const NamesAndLayers & all_names_and_layers();
	ELayerType get_layer_from_name(const std::string & name);
	std::string to_string(const ELayerType type);
	/// @}

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

	enum class EIoULoss
	{
		// Please keep the old C and the new C++ enums in sync!
		IOU		= IOU_LOSS::IOU,
		GIOU	= IOU_LOSS::GIOU,
		MSE		= IOU_LOSS::MSE,
		DIOU	= IOU_LOSS::DIOU,
		CIOU	= IOU_LOSS::CIOU,
	};

	/// @{ Convert between names and IoU loss types.
	using NamesAndIoULoss = std::map<std::string, EIoULoss>;
	const NamesAndIoULoss & all_names_and_IoU_loss();
	EIoULoss get_IoU_loss_from_name(const std::string & name);
	std::string to_string(const EIoULoss loss);
	/// @}

	enum class ENMSKind
	{
		DEFAULT_NMS	= NMS_KIND::DEFAULT_NMS	,
		GREEDY_NMS	= NMS_KIND::GREEDY_NMS	,
		DIOU_NMS	= NMS_KIND::DIOU_NMS	,
		CORNERS_NMS	= NMS_KIND::CORNERS_NMS	, // gaussian yolo
	};

	/// @{ Convert between names and IoU loss types.
	using NamesAndNMSKind = std::map<std::string, ENMSKind>;
	const NamesAndNMSKind & all_names_and_NMS_kind();
	ENMSKind get_NMS_kind_from_name(const std::string & name);
	std::string to_string(const ENMSKind nms_kind);
	/// @}

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
