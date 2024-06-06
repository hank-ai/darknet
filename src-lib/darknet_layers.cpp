#include "darknet_internal.hpp"


const Darknet::NamesAndLayers & Darknet::all_names_and_layers()
{
	TAT(TATPARMS);

	const static NamesAndLayers m =
	{
		{"shortcut"			, ELayerType::SHORTCUT			},
		{"scale_channels"	, ELayerType::SCALE_CHANNELS	},
		{"sam"				, ELayerType::SAM				},
		{"crop"				, ELayerType::CROP				},
		{"cost"				, ELayerType::COST				},
		{"detection"		, ELayerType::DETECTION			},
		{"region"			, ELayerType::REGION			},
		{"yolo"				, ELayerType::YOLO				},
		{"Gaussian_yolo"	, ELayerType::GAUSSIAN_YOLO		},
		{"local"			, ELayerType::LOCAL				},
		{"conv"				, ELayerType::CONVOLUTIONAL		},
		{"convolutional"	, ELayerType::CONVOLUTIONAL		},
		{"activation"		, ELayerType::ACTIVE			},
		{"net"				, ELayerType::NETWORK			},
		{"network"			, ELayerType::NETWORK			},
		{"crnn"				, ELayerType::CRNN				},
		{"gru"				, ELayerType::GRU				},
		{"lstm"				, ELayerType::LSTM				},
		{"conv_lstm"		, ELayerType::CONV_LSTM			},
		{"history"			, ELayerType::HISTORY			},
		{"rnn"				, ELayerType::RNN				},
		{"conn"				, ELayerType::CONNECTED			},
		{"connected"		, ELayerType::CONNECTED			},
		{"max"				, ELayerType::MAXPOOL			},
		{"maxpool"			, ELayerType::MAXPOOL			},
		{"local_avg"		, ELayerType::LOCAL_AVGPOOL		},
		{"local_avgpool"	, ELayerType::LOCAL_AVGPOOL		},
		{"reorg3d"			, ELayerType::REORG				},
		{"reorg"			, ELayerType::REORG_OLD			},
		{"avg"				, ELayerType::AVGPOOL			},
		{"avgpool"			, ELayerType::AVGPOOL			},
		{"dropout"			, ELayerType::DROPOUT			},
		{"lrn"				, ELayerType::NORMALIZATION		},
		{"normalization"	, ELayerType::NORMALIZATION		},
		{"batchnorm"		, ELayerType::BATCHNORM			},
		{"soft"				, ELayerType::SOFTMAX			},
		{"softmax"			, ELayerType::SOFTMAX			},
		{"contrastive"		, ELayerType::CONTRASTIVE		},
		{"route"			, ELayerType::ROUTE				},
		{"upsample"			, ELayerType::UPSAMPLE			},
		{"empty"			, ELayerType::EMPTY				},
		{"silence"			, ELayerType::EMPTY				},
		{"implicit"			, ELayerType::IMPLICIT			},
	};

	return m;
}


Darknet::ELayerType Darknet::get_layer_from_name(const std::string & name)
{
	const auto & m = all_names_and_layers();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet layer name
		throw std::invalid_argument("layer name \"" + name + "\" is not supported");
	}

	return m.at(name);
}


std::string Darknet::get_name_from_layer(const ELayerType type)
{
	const auto & m = all_names_and_layers();
	for (const auto & [k, v] : m)
	{
		if (type == v)
		{
			return k;
		}
	}

	throw std::invalid_argument("unknown layer #" + std::to_string(static_cast<int>(type)));
}
