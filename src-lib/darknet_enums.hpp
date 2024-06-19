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
		CONVOLUTIONAL	= LAYER_TYPE::CONVOLUTIONAL		,
		DECONVOLUTIONAL	= LAYER_TYPE::DECONVOLUTIONAL	,	// unused
		CONNECTED		= LAYER_TYPE::CONNECTED			,
		MAXPOOL			= LAYER_TYPE::MAXPOOL			,
		LOCAL_AVGPOOL	= LAYER_TYPE::LOCAL_AVGPOOL		,	// only used in 1 old config?
		SOFTMAX			= LAYER_TYPE::SOFTMAX			,
		DETECTION		= LAYER_TYPE::DETECTION			,
		DROPOUT			= LAYER_TYPE::DROPOUT			,
		CROP			= LAYER_TYPE::CROP				,
		ROUTE			= LAYER_TYPE::ROUTE				,
		COST			= LAYER_TYPE::COST				,
		NORMALIZATION	= LAYER_TYPE::NORMALIZATION		,	// unused
		AVGPOOL			= LAYER_TYPE::AVGPOOL			,
		LOCAL			= LAYER_TYPE::LOCAL				,
		SHORTCUT		= LAYER_TYPE::SHORTCUT			,
		SCALE_CHANNELS	= LAYER_TYPE::SCALE_CHANNELS	,	// only used in 2 places?
		SAM				= LAYER_TYPE::SAM				,	// only used in 3 places?
		ACTIVE			= LAYER_TYPE::ACTIVE			,	// unused
		RNN				= LAYER_TYPE::RNN				,	// only used in 2 places?
		GRU				= LAYER_TYPE::GRU				,	// only used in 1 old config?
		LSTM			= LAYER_TYPE::LSTM				,	// only used in 1 old config?
		CONV_LSTM		= LAYER_TYPE::CONV_LSTM			,	// unused
		HISTORY			= LAYER_TYPE::HISTORY			,	// unused
		CRNN			= LAYER_TYPE::CRNN				,	// only used in 2 pleaces?
		BATCHNORM		= LAYER_TYPE::BATCHNORM			,	// only used in 1 old config?
		NETWORK			= LAYER_TYPE::NETWORK			,	// this is a section name, but does not correspond to a layer type
		XNOR			= LAYER_TYPE::XNOR				,	// unused
		REGION			= LAYER_TYPE::REGION			,	// only used in 1 place?
		YOLO			= LAYER_TYPE::YOLO				,
		GAUSSIAN_YOLO	= LAYER_TYPE::GAUSSIAN_YOLO		,	// only used in 1 config?
		ISEG			= LAYER_TYPE::ISEG				,	// unused
		REORG			= LAYER_TYPE::REORG				,	// only used in 1 old config?
		REORG_OLD		= LAYER_TYPE::REORG_OLD			,
		UPSAMPLE		= LAYER_TYPE::UPSAMPLE			,
		LOGXENT			= LAYER_TYPE::LOGXENT			,	// unused
		L2NORM			= LAYER_TYPE::L2NORM			,	// unused
		EMPTY			= LAYER_TYPE::EMPTY				,	// unused
		BLANK			= LAYER_TYPE::BLANK				,	// unused
		CONTRASTIVE		= LAYER_TYPE::CONTRASTIVE		,	// only used in 1 old config?
		IMPLICIT		= LAYER_TYPE::IMPLICIT			,	// unused
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
};
