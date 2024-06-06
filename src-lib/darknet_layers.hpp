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
		//
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
		NETWORK			= LAYER_TYPE::NETWORK			,
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

	using NamesAndLayers = std::map<std::string, ELayerType>;

	/** Get all of the names -> layers mapping.  Note that several layers have multiple names, such as @p "net" and
	 * @p "network", or @p "conv" and @p "convolutional".  So there are more names than there are valid layers.
	 */
	const NamesAndLayers & all_names_and_layers();

	/// Get a layer type given a specific name.  For example, this will return @p CONVOLUTIONAL for the name @p "conv".
	ELayerType get_layer_from_name(const std::string & name);

	/// Get a layer name for a specific layer type.
	std::string get_name_from_layer(const ELayerType type);
};


#include "activation_layer.hpp"
#include "avgpool_layer.hpp"
#include "batchnorm_layer.hpp"
#include "connected_layer.hpp"
#include "conv_lstm_layer.hpp"
#include "convolutional_layer.hpp"
#include "cost_layer.hpp"
#include "crnn_layer.hpp"
#include "crop_layer.hpp"
//#include "deconvolutional_layer.h"
#include "detection_layer.hpp"
#include "dropout_layer.hpp"
#include "gaussian_yolo_layer.hpp"
#include "gru_layer.hpp"
#include "layer.hpp"
#include "local_layer.hpp"
#include "lstm_layer.hpp"
#include "maxpool_layer.hpp"
#include "normalization_layer.hpp"
#include "region_layer.hpp"
#include "reorg_layer.hpp"
#include "reorg_old_layer.hpp"
#include "representation_layer.hpp"
#include "rnn_layer.hpp"
#include "route_layer.hpp"
#include "sam_layer.hpp"
#include "scale_channels_layer.hpp"
#include "shortcut_layer.hpp"
#include "softmax_layer.hpp"
#include "upsample_layer.hpp"
#include "yolo_layer.hpp"
