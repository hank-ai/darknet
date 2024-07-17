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
		{"cost"				, ELayerType::COST				},
		{"region"			, ELayerType::REGION			},
		{"yolo"				, ELayerType::YOLO				},
		{"Gaussian_yolo"	, ELayerType::GAUSSIAN_YOLO		},
		{"conv"				, ELayerType::CONVOLUTIONAL		},
		{"convolutional"	, ELayerType::CONVOLUTIONAL		},
		{"net"				, ELayerType::NETWORK			},
		{"network"			, ELayerType::NETWORK			},
		{"crnn"				, ELayerType::CRNN				},
		{"lstm"				, ELayerType::LSTM				},
		{"rnn"				, ELayerType::RNN				},
		{"conn"				, ELayerType::CONNECTED			},
		{"connected"		, ELayerType::CONNECTED			},
		{"max"				, ELayerType::MAXPOOL			},
		{"maxpool"			, ELayerType::MAXPOOL			},
		{"local_avg"		, ELayerType::LOCAL_AVGPOOL		},
		{"local_avgpool"	, ELayerType::LOCAL_AVGPOOL		},
		{"reorg3d"			, ELayerType::REORG				},
		{"avg"				, ELayerType::AVGPOOL			},
		{"avgpool"			, ELayerType::AVGPOOL			},
		{"dropout"			, ELayerType::DROPOUT			},
		{"soft"				, ELayerType::SOFTMAX			},
		{"softmax"			, ELayerType::SOFTMAX			},
		{"contrastive"		, ELayerType::CONTRASTIVE		},
		{"route"			, ELayerType::ROUTE				},
		{"upsample"			, ELayerType::UPSAMPLE			},
		{"empty"			, ELayerType::EMPTY				},
		{"silence"			, ELayerType::EMPTY				},
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
//		throw std::invalid_argument("layer name \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "layer name \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const ELayerType type)
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

//	throw std::invalid_argument("unknown layer #" + std::to_string(static_cast<int>(type)));
	darknet_fatal_error(DARKNET_LOC, "unknown layer #%d", static_cast<int>(type));
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
//		throw std::invalid_argument("activation name \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "activation name \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const Darknet::EActivation activation)
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

//	throw std::invalid_argument("unknown activation #" + std::to_string(static_cast<int>(activation)));
	darknet_fatal_error(DARKNET_LOC, "unknown activation type #%d", static_cast<int>(activation));
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
//		throw std::invalid_argument("learning rate policy \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "learning rate policy \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const Darknet::ELearningRatePolicy policy)
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

//	throw std::invalid_argument("unknown learning rate policy #" + std::to_string(static_cast<int>(policy)));
	darknet_fatal_error(DARKNET_LOC, "unknown learning rate policy #%d", static_cast<int>(policy));
}


const Darknet::NamesAndIoULoss & Darknet::all_names_and_IoU_loss()
{
	TAT(TATPARMS);

	const static NamesAndIoULoss m =
	{
		{"iou"	, EIoULoss::IOU	},
		{"giou"	, EIoULoss::GIOU},
		{"mse"	, EIoULoss::MSE	},
		{"diou"	, EIoULoss::DIOU},
		{"ciou"	, EIoULoss::CIOU},
	};

	return m;
}


Darknet::EIoULoss Darknet::get_IoU_loss_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_IoU_loss();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet IoU loss type
//		throw std::invalid_argument("IoU loss type \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "IoU loss \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const Darknet::EIoULoss loss)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_IoU_loss();
	for (const auto & [k, v] : m)
	{
		if (loss == v)
		{
			return k;
		}
	}

//	throw std::invalid_argument("unknown IoU loss type #" + std::to_string(static_cast<int>(loss)));
	darknet_fatal_error(DARKNET_LOC, "unknown IoU loss #%d", static_cast<int>(loss));
}


const Darknet::NamesAndNMSKind & Darknet::all_names_and_NMS_kind()
{
	TAT(TATPARMS);

	const static NamesAndNMSKind m =
	{
		{"default"		, ENMSKind::DEFAULT_NMS	},
		{"greedynms"	, ENMSKind::GREEDY_NMS	},
		{"diounms"		, ENMSKind::DIOU_NMS	},
		{"cornersnms"	, ENMSKind::CORNERS_NMS	},
	};

	return m;
}


Darknet::ENMSKind Darknet::get_NMS_kind_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_NMS_kind();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet NMS type
//		throw std::invalid_argument("NMS type \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "NMS kind \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const Darknet::ENMSKind nms_kind)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_NMS_kind();
	for (const auto & [k, v] : m)
	{
		if (nms_kind == v)
		{
			return k;
		}
	}

//	throw std::invalid_argument("unknown NMS kind type #" + std::to_string(static_cast<int>(nms_kind)));
	darknet_fatal_error(DARKNET_LOC, "unknown NMS kind #%d", static_cast<int>(nms_kind));
}


const Darknet::NamesAndWeightsType & Darknet::all_names_and_weights_types()
{
	TAT(TATPARMS);

	const static NamesAndWeightsType m =
	{
		{"none"			, EWeightsType::NO_WEIGHTS	},
		{"per_feature"	, EWeightsType::PER_FEATURE	},
		{"per_channel"	, EWeightsType::PER_CHANNEL	},
	};

	return m;
}


Darknet::EWeightsType Darknet::get_weights_type_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_weights_types();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet weights type
//		throw std::invalid_argument("weights type \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "weights type \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const EWeightsType type)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_weights_types();
	for (const auto & [k, v] : m)
	{
		if (type == v)
		{
			return k;
		}
	}

//	throw std::invalid_argument("unknown weights type #" + std::to_string(static_cast<int>(type)));
	darknet_fatal_error(DARKNET_LOC, "unknown weights type #%d", static_cast<int>(type));
}


const Darknet::NamesAndWeightsNormalization & Darknet::all_names_and_weights_normalization()
{
	TAT(TATPARMS);

	const static NamesAndWeightsNormalization m =
	{
		{"relu"		, EWeightsNormalization::RELU_NORMALIZATION		},
		{"avg_relu"	, EWeightsNormalization::RELU_NORMALIZATION		},
		{"softmax"	, EWeightsNormalization::SOFTMAX_NORMALIZATION	},
		{"none"		, EWeightsNormalization::NO_NORMALIZATION		},
	};

	return m;
}


Darknet::EWeightsNormalization Darknet::get_weights_normalization_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_weights_normalization();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet weights type
//		throw std::invalid_argument("weights normalization \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "weights normalization \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const EWeightsNormalization normalization)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_weights_normalization();
	for (const auto & [k, v] : m)
	{
		if (normalization == v)
		{
			return k;
		}
	}

//	throw std::invalid_argument("unknown weights normalization #" + std::to_string(static_cast<int>(normalization)));
	darknet_fatal_error(DARKNET_LOC, "unknown weights normalization #%d", static_cast<int>(normalization));
}


const Darknet::NamesAndCostTypes & Darknet::all_names_and_cost_types()
{
	TAT(TATPARMS);

	const static NamesAndCostTypes m =
	{
		{"sse"		, ECostType::SSE	},
		{"masked"	, ECostType::MASKED	},
		{"smooth"	, ECostType::SMOOTH	},
#if 0
		/// @todo these next 3 didn't exist in the codebase -- should they exist?  what should they be called?
		{"l1"		, ECostType::L1		},
		{"seg"		, ECostType::SEG	},
		{"wgan"		, ECostType::WGAN	},
#endif
	};

	return m;
}


Darknet::ECostType Darknet::get_cost_types_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_cost_types();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet cost type
//		throw std::invalid_argument("cost type \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "cost type \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const ECostType type)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_cost_types();
	for (const auto & [k, v] : m)
	{
		if (type == v)
		{
			return k;
		}
	}

//	throw std::invalid_argument("unknown cost type #" + std::to_string(static_cast<int>(type)));
	darknet_fatal_error(DARKNET_LOC, "unknown cost type #%d", static_cast<int>(type));
}


const Darknet::NamesAndYoloPointTypes & Darknet::all_names_and_yolo_point_types()
{
	TAT(TATPARMS);

	const static NamesAndYoloPointTypes m =
	{
		{"center"		, EYoloPoint::YOLO_CENTER		},
		{"left_top"		, EYoloPoint::YOLO_LEFT_TOP		},
		{"right_bottom"	, EYoloPoint::YOLO_RIGHT_BOTTOM	},
	};

	return m;
}


Darknet::EYoloPoint Darknet::get_yolo_point_types_from_name(const std::string & name)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_yolo_point_types();

	if (m.count(name) == 0)
	{
		/// @throw sd::invalid_argument if the name is not a valid Darknet YOLO point type
//		throw std::invalid_argument("yolo point type \"" + name + "\" is not supported");
		darknet_fatal_error(DARKNET_LOC, "YOLO point type \"%s\" is not supported", name.c_str());
	}

	return m.at(name);
}


std::string Darknet::to_string(const EYoloPoint type)
{
	TAT(TATPARMS);

	const auto & m = all_names_and_yolo_point_types();
	for (const auto & [k, v] : m)
	{
		if (type == v)
		{
			return k;
		}
	}

//	throw std::invalid_argument("unknown YOLO point type #" + std::to_string(static_cast<int>(type)));
	darknet_fatal_error(DARKNET_LOC, "unknown YOLO point type #%d", static_cast<int>(type));
}
