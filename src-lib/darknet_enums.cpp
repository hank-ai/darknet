#include "darknet_internal.hpp"


const Darknet::NamesAndLayers & Darknet::all_names_and_layers()
{
	TAT(TATPARMS);

	// these are the names we expect to find as sections types in .cfg files
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
	TAT(TATPARMS);

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
	TAT(TATPARMS);

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


const Darknet::NamesAndActivationTypes & Darknet::all_names_and_activations()
{
	TAT(TATPARMS);

	// these are the activation names we expect to find in .cfg files
	const static NamesAndActivationTypes m =
	{
		{"elu"								, EActivation::ELU						},
		{"gelu"								, EActivation::GELU						},
		{"hard_mish"						, EActivation::HARD_MISH				},
		{"hardtan"							, EActivation::HARDTAN					},
		{"leaky"							, EActivation::LEAKY					},
		{"lhtan"							, EActivation::LHTAN					},
		{"linear"							, EActivation::LINEAR					},
		{"logistic"							, EActivation::LOGISTIC					},
		{"loggy"							, EActivation::LOGGY					},
		{"mish"								, EActivation::MISH						},
		{"normalize_channels"				, EActivation::NORM_CHAN				},
		{"normalize_channels_softmax"		, EActivation::NORM_CHAN_SOFTMAX		},
		{"normalize_channels_softmax_maxval", EActivation::NORM_CHAN_SOFTMAX_MAXVAL	},
		{"plse"								, EActivation::PLSE						},
		{"ramp"								, EActivation::RAMP						},
		{"relie"							, EActivation::RELIE					},
		{"relu"								, EActivation::RELU						},
		{"relu6"							, EActivation::RELU6					},
		{"revleaky"							, EActivation::REVLEAKY					},
		{"selu"								, EActivation::SELU						},
		{"stair"							, EActivation::STAIR					},
		{"swish"							, EActivation::SWISH					},
		{"tanh"								, EActivation::TANH						},
	};

	return m;
}


Darknet::EActivation Darknet::get_activation_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_activations();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet activation name
		throw std::invalid_argument("activation name \"" + name + "\" is not supported");
	}

	return m.at(name);
}


std::string Darknet::get_name_from_activation(const Darknet::EActivation activation)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_activations();
	for (const auto & [k, v] : m)
	{
		if (activation == v)
		{
			return k;
		}
	}

	throw std::invalid_argument("unknown activation #" + std::to_string(static_cast<int>(activation)));
}


const Darknet::NamesAndLearningRatePolicies & Darknet::all_names_and_learning_rate_policies()
{
	TAT(TATPARMS);

	const static NamesAndLearningRatePolicies m =
	{
		{"random"	, ELearningRatePolicy::RANDOM	},
		{"poly"		, ELearningRatePolicy::POLY		},
		{"constant"	, ELearningRatePolicy::CONSTANT	},
		{"step"		, ELearningRatePolicy::STEP		},
		{"exp"		, ELearningRatePolicy::EXP		},
		{"sigmoid"	, ELearningRatePolicy::SIG		},
		{"steps"	, ELearningRatePolicy::STEPS	},
		{"sgdr"		, ELearningRatePolicy::SGDR		},
	};

	return m;
}


Darknet::ELearningRatePolicy Darknet::get_learning_rate_policy_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_learning_rate_policies();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet learning rate policy
		throw std::invalid_argument("learning rate policy \"" + name + "\" is not supported");
	}

	return m.at(name);
}


std::string Darknet::get_name_from_learning_rate_policy(const Darknet::ELearningRatePolicy policy)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_learning_rate_policies();
	for (const auto & [k, v] : m)
	{
		if (policy == v)
		{
			return k;
		}
	}

	throw std::invalid_argument("unknown learning rate policy #" + std::to_string(static_cast<int>(policy)));
}
