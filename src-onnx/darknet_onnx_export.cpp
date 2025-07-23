#include "darknet_internal.hpp"


void LogHandler(google::protobuf::LogLevel level, const char* filename, int line, const std::string& message)
{
	std::cout << "Protocol buffer error detected:"
		<< " level="	<< level
		<< " fn="		<< filename
		<< " line="		<< line
		<< " msg="		<< message
		<< std::endl;

	if (level == google::protobuf::LOGLEVEL_ERROR or
		level == google::protobuf::LOGLEVEL_FATAL)
	{
		throw std::runtime_error("cannot continue due to unexpected protocol buffer error");
	}

	return;
}


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms	= Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net	= Darknet::load_neural_network(parms);
		const auto cfg_fn		= Darknet::get_config_filename(net);
		const auto weights_fn	= Darknet::get_weights_filename(net);
		const auto onnx_fn		= std::filesystem::path(cfg_fn).replace_extension(".onnx");

		std::cout << std::endl << "Darknet/YOLO ONNX Export Tool"		<< std::endl
			<< "-> configuration ........ " << cfg_fn		.string()	<< std::endl
			<< "-> weights .............. " << weights_fn	.string()	<< std::endl
			<< "-> onnx output .......... " << onnx_fn		.string()	<< std::endl
			;

		GOOGLE_PROTOBUF_VERIFY_VERSION;

		google::protobuf::SetLogHandler(&LogHandler);

		// try to delete the .onnx file if an old version already exists
		if (std::filesystem::exists(onnx_fn))
		{
			bool success = std::filesystem::remove(onnx_fn);
			if (not success)
			{
				throw std::runtime_error("failed to delete existing file " + onnx_fn.string());
			}
		}

		onnx::ModelProto model;

		// IR = Intermediate Representation, related to versioning
		// https://github.com/onnx/onnx/blob/main/docs/IR.md
		// https://github.com/onnx/onnx/blob/main/docs/Versioning.md
		// 2019_9_19 aka "6" is the last version prior to introducing training
		model.set_ir_version(onnx::Version::IR_VERSION_2019_9_19);
//		model.set_ir_version(onnx::Version::IR_VERSION_2025_05_12);

		// The name of the framework or tool used to generate this model.
		model.set_producer_name("Darknet/YOLO ONNX Export Tool");

		// The version of the framework or tool used to generate this model.
		model.set_producer_version(DARKNET_VERSION_SHORT);

		// We use reverse domain names as name space indicators.
		/// @todo We need a command-line parameter for this field.
		model.set_domain("ai.darknetcv");

		// The version of the graph encoded.
		/// @todo We need a command-line parameter for this field.
		model.set_model_version(1);

		// A human-readable documentation for this attribute. Markdown is allowed.
		std::time_t tt = std::time(nullptr);
		char buffer[50];
		std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %z", std::localtime(&tt));
		model.set_doc_string("ONNX output generated from Darknet/YOLO neural network files " + cfg_fn.filename().string() + " and " + weights_fn.filename().string() + " on " + buffer + ".");

		std::cout
			<< "-> type name ............ " << model.GetTypeName()			<< std::endl
			<< "-> opset import size .... " << model.opset_import_size()	<< std::endl
			<< "-> metadata props size .. " << model.metadata_props_size()	<< std::endl
			<< "-> training info size ... " << model.training_info_size()	<< std::endl
			<< "-> functions size ....... " << model.functions_size()		<< std::endl
			<< "-> configuration size ... " << model.configuration_size()	<< std::endl
			<< "-> producer name ........ " << model.producer_name()		<< std::endl
			<< "-> producer version ..... " << model.producer_version()		<< std::endl
			<< "-> doc string ........... " << model.doc_string()			<< std::endl
			<< "-> has graph ............ " << model.has_graph()			<< std::endl
			<< "-> ir version ........... " << model.ir_version()			<< std::endl
			<< "-> domain ............... " << model.domain()				<< std::endl
			<< "-> model version ........ " << model.model_version()		<< std::endl
			;

		std::ofstream ofs(onnx_fn, std::ios::binary);
		const bool success = model.SerializeToOstream(&ofs);
		if (not success)
		{
			throw std::runtime_error("failed to save ONNX output file " + onnx_fn.string());
		}
		ofs.close();
		std::cout << "-> onnx saved to ........ " << onnx_fn.string() << std::endl;

		google::protobuf::ShutdownProtobufLibrary();

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
