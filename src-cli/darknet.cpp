#include <csignal>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "darknet.h"
#include "parser.hpp"
#include "darknet_internal.hpp"


extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void run_voxel(int argc, char **argv);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_writing(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_dice(int argc, char **argv);
extern void run_compare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_vid_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);

void average(int argc, char *argv[])
{
	char *cfgfile = argv[2];
	char *outfile = argv[3];
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	network sum = parse_network_cfg(cfgfile);

	char *weightfile = argv[4];
	load_weights(&sum, weightfile);

	int i, j;
	int n = argc - 5;
	for(i = 0; i < n; ++i){
		weightfile = argv[i+5];
		load_weights(&net, weightfile);
		for(j = 0; j < net.n; ++j){
			layer l = net.layers[j];
			layer out = sum.layers[j];
			if(l.type == CONVOLUTIONAL){
				int num = l.n*l.c*l.size*l.size;
				axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
				if(l.batch_normalize){
					axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
					axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
					axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
				}
			}
			if(l.type == CONNECTED){
				axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
			}
		}
	}
	n = n+1;
	for(j = 0; j < net.n; ++j){
		layer l = sum.layers[j];
		if(l.type == CONVOLUTIONAL){
			int num = l.n*l.c*l.size*l.size;
			scal_cpu(l.n, 1./n, l.biases, 1);
			scal_cpu(num, 1./n, l.weights, 1);
				if(l.batch_normalize){
					scal_cpu(l.n, 1./n, l.scales, 1);
					scal_cpu(l.n, 1./n, l.rolling_mean, 1);
					scal_cpu(l.n, 1./n, l.rolling_variance, 1);
				}
		}
		if(l.type == CONNECTED){
			scal_cpu(l.outputs, 1./n, l.biases, 1);
			scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
		}
	}
	save_weights(sum, outfile);
}

void speed(char *cfgfile, int tics)
{
	if (tics == 0) tics = 1000;
	network net = parse_network_cfg(cfgfile);
	set_batch_network(&net, 1);
	int i;
	time_t start = time(0);
	image im = make_image(net.w, net.h, net.c);
	for(i = 0; i < tics; ++i){
		network_predict(net, im.data);
	}
	double t = difftime(time(0), start);
	printf("\n%d evals, %f Seconds\n", tics, t);
	printf("Speed: %f sec/eval\n", t/tics);
	printf("Speed: %f Hz\n", tics/t);
}

void operations(char *cfgfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	int i;
	long ops = 0;
	for(i = 0; i < net.n; ++i){
		layer l = net.layers[i];
		if(l.type == CONVOLUTIONAL){
			ops += 2l * l.n * l.size*l.size*l.c * l.out_h*l.out_w;
		} else if(l.type == CONNECTED){
			ops += 2l * l.inputs * l.outputs;
		} else if (l.type == RNN){
			ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
			ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
			ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
		} else if (l.type == GRU){
			ops += 2l * l.uz->inputs * l.uz->outputs;
			ops += 2l * l.uh->inputs * l.uh->outputs;
			ops += 2l * l.ur->inputs * l.ur->outputs;
			ops += 2l * l.wz->inputs * l.wz->outputs;
			ops += 2l * l.wh->inputs * l.wh->outputs;
			ops += 2l * l.wr->inputs * l.wr->outputs;
		} else if (l.type == LSTM){
			ops += 2l * l.uf->inputs * l.uf->outputs;
			ops += 2l * l.ui->inputs * l.ui->outputs;
			ops += 2l * l.ug->inputs * l.ug->outputs;
			ops += 2l * l.uo->inputs * l.uo->outputs;
			ops += 2l * l.wf->inputs * l.wf->outputs;
			ops += 2l * l.wi->inputs * l.wi->outputs;
			ops += 2l * l.wg->inputs * l.wg->outputs;
			ops += 2l * l.wo->inputs * l.wo->outputs;
		}
	}
	printf("Floating Point Operations: %ld\n", ops);
	printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	int oldn = net.layers[net.n - 2].n;
	int c = net.layers[net.n - 2].c;
	net.layers[net.n - 2].n = 9372;
	net.layers[net.n - 2].biases += 5;
	net.layers[net.n - 2].weights += 5*c;
	if(weightfile){
		load_weights(&net, weightfile);
	}
	net.layers[net.n - 2].biases -= 5;
	net.layers[net.n - 2].weights -= 5*c;
	net.layers[net.n - 2].n = oldn;
	printf("%d\n", oldn);
	layer l = net.layers[net.n - 2];
	copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
	copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
	copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
	copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
	*net.seen = 0;
	*net.cur_iteration = 0;
	save_weights(net, outfile);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg_custom(cfgfile, 1, 1);
	if(weightfile){
		load_weights_upto(&net, weightfile, max);
	}
	*net.seen = 0;
	*net.cur_iteration = 0;
	save_weights_upto(net, outfile, max, 0);
}

void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	int i;
	for(i = 0; i < net.n; ++i){
		layer l = net.layers[i];
		if(l.type == CONVOLUTIONAL){
			rescale_weights(l, 2, -.5);
			break;
		}
	}
	save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	int i;
	for(i = 0; i < net.n; ++i){
		layer l = net.layers[i];
		if(l.type == CONVOLUTIONAL){
			rgbgr_weights(l);
			break;
		}
	}
	save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL && l.batch_normalize) {
			denormalize_convolutional_layer(l);
		}
		if (l.type == CONNECTED && l.batch_normalize) {
			denormalize_connected_layer(l);
		}
		if (l.type == GRU && l.batch_normalize) {
			denormalize_connected_layer(*l.input_z_layer);
			denormalize_connected_layer(*l.input_r_layer);
			denormalize_connected_layer(*l.input_h_layer);
			denormalize_connected_layer(*l.state_z_layer);
			denormalize_connected_layer(*l.state_r_layer);
			denormalize_connected_layer(*l.state_h_layer);
		}
		if (l.type == LSTM && l.batch_normalize) {
			denormalize_connected_layer(*l.wf);
			denormalize_connected_layer(*l.wi);
			denormalize_connected_layer(*l.wg);
			denormalize_connected_layer(*l.wo);
			denormalize_connected_layer(*l.uf);
			denormalize_connected_layer(*l.ui);
			denormalize_connected_layer(*l.ug);
			denormalize_connected_layer(*l.uo);
		}
	}
	save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
	int j;
	l.batch_normalize=1;
	l.scales = (float*)xcalloc(n, sizeof(float));
	for(j = 0; j < n; ++j){
		l.scales[j] = 1;
	}
	l.rolling_mean = (float*)xcalloc(n, sizeof(float));
	l.rolling_variance = (float*)xcalloc(n, sizeof(float));
	return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	int i;
	for(i = 0; i < net.n; ++i){
		layer l = net.layers[i];
		if(l.type == CONVOLUTIONAL && !l.batch_normalize){
			net.layers[i] = normalize_layer(l, l.n);
		}
		if (l.type == CONNECTED && !l.batch_normalize) {
			net.layers[i] = normalize_layer(l, l.outputs);
		}
		if (l.type == GRU && l.batch_normalize) {
			*l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
			*l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
			*l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
			*l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
			*l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
			*l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
			net.layers[i].batch_normalize=1;
		}
		if (l.type == LSTM && l.batch_normalize) {
			*l.wf = normalize_layer(*l.wf, l.wf->outputs);
			*l.wi = normalize_layer(*l.wi, l.wi->outputs);
			*l.wg = normalize_layer(*l.wg, l.wg->outputs);
			*l.wo = normalize_layer(*l.wo, l.wo->outputs);
			*l.uf = normalize_layer(*l.uf, l.uf->outputs);
			*l.ui = normalize_layer(*l.ui, l.ui->outputs);
			*l.ug = normalize_layer(*l.ug, l.ug->outputs);
			*l.uo = normalize_layer(*l.uo, l.uo->outputs);
			net.layers[i].batch_normalize=1;
		}
	}
	save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONNECTED && l.batch_normalize) {
			printf("Connected Layer %d\n", i);
			statistics_connected_layer(l);
		}
		if (l.type == GRU && l.batch_normalize) {
			printf("GRU Layer %d\n", i);
			printf("Input Z\n");
			statistics_connected_layer(*l.input_z_layer);
			printf("Input R\n");
			statistics_connected_layer(*l.input_r_layer);
			printf("Input H\n");
			statistics_connected_layer(*l.input_h_layer);
			printf("State Z\n");
			statistics_connected_layer(*l.state_z_layer);
			printf("State R\n");
			statistics_connected_layer(*l.state_r_layer);
			printf("State H\n");
			statistics_connected_layer(*l.state_h_layer);
		}
		if (l.type == LSTM && l.batch_normalize) {
			printf("LSTM Layer %d\n", i);
			printf("wf\n");
			statistics_connected_layer(*l.wf);
			printf("wi\n");
			statistics_connected_layer(*l.wi);
			printf("wg\n");
			statistics_connected_layer(*l.wg);
			printf("wo\n");
			statistics_connected_layer(*l.wo);
			printf("uf\n");
			statistics_connected_layer(*l.uf);
			printf("ui\n");
			statistics_connected_layer(*l.ui);
			printf("ug\n");
			statistics_connected_layer(*l.ug);
			printf("uo\n");
			statistics_connected_layer(*l.uo);
		}
		printf("\n");
	}
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	Darknet::CfgAndState::get().gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL && l.batch_normalize) {
			denormalize_convolutional_layer(l);
			net.layers[i].batch_normalize=0;
		}
		if (l.type == CONNECTED && l.batch_normalize) {
			denormalize_connected_layer(l);
			net.layers[i].batch_normalize=0;
		}
		if (l.type == GRU && l.batch_normalize) {
			denormalize_connected_layer(*l.input_z_layer);
			denormalize_connected_layer(*l.input_r_layer);
			denormalize_connected_layer(*l.input_h_layer);
			denormalize_connected_layer(*l.state_z_layer);
			denormalize_connected_layer(*l.state_r_layer);
			denormalize_connected_layer(*l.state_h_layer);
			l.input_z_layer->batch_normalize = 0;
			l.input_r_layer->batch_normalize = 0;
			l.input_h_layer->batch_normalize = 0;
			l.state_z_layer->batch_normalize = 0;
			l.state_r_layer->batch_normalize = 0;
			l.state_h_layer->batch_normalize = 0;
			net.layers[i].batch_normalize=0;
		}
		if (l.type == GRU && l.batch_normalize) {
			denormalize_connected_layer(*l.wf);
			denormalize_connected_layer(*l.wi);
			denormalize_connected_layer(*l.wg);
			denormalize_connected_layer(*l.wo);
			denormalize_connected_layer(*l.uf);
			denormalize_connected_layer(*l.ui);
			denormalize_connected_layer(*l.ug);
			denormalize_connected_layer(*l.uo);
			l.wf->batch_normalize = 0;
			l.wi->batch_normalize = 0;
			l.wg->batch_normalize = 0;
			l.wo->batch_normalize = 0;
			l.uf->batch_normalize = 0;
			l.ui->batch_normalize = 0;
			l.ug->batch_normalize = 0;
			l.uo->batch_normalize = 0;
			net.layers[i].batch_normalize=0;
		}
	}
	save_weights(net, outfile);
}

void visualize(char *cfgfile, char *weightfile)
{
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	visualize_network(net);
	wait_until_press_key_cv();
}


void darknet_signal_handler(int sig)
{
	// prevent recursion if this signal happens again (set the default signal action)
	std::signal(sig, SIG_DFL);

	std::cout << "calling Darknet's fatal error handler due to signal #" << sig << std::endl;

	#ifdef WIN32
	darknet_fatal_error(DARKNET_LOC, "signal handler invoked for signal #%d", sig);
	#else
	darknet_fatal_error(DARKNET_LOC, "signal handler invoked for signal #%d (%s)", sig, strsignal(sig));
	#endif
}


int main(int argc, char **argv)
{
	try
	{
		// disable console IO buffering since we're still mixing old printf() calls with std::cout
		std::setvbuf(stdout, NULL, _IONBF, 0);
		std::setvbuf(stderr, NULL, _IONBF, 0);

		signal(SIGINT   , darknet_signal_handler);  // 2: CTRL+C
		signal(SIGILL   , darknet_signal_handler);  // 4: illegal instruction
		signal(SIGABRT  , darknet_signal_handler);  // 6: abort()
		signal(SIGFPE   , darknet_signal_handler);  // 8: floating point exception
		signal(SIGSEGV  , darknet_signal_handler);  // 11: segfault
		signal(SIGTERM  , darknet_signal_handler);  // 15: terminate
#ifdef WIN32
		signal(SIGBREAK , darknet_signal_handler);  // Break is different than CTRL+C on Windows
#else
		signal(SIGHUP   , darknet_signal_handler);  // 1: hangup
		signal(SIGQUIT  , darknet_signal_handler);  // 3: quit
		signal(SIGUSR1  , darknet_signal_handler);  // 10: user-defined
		signal(SIGUSR2  , darknet_signal_handler);  // 12: user-defined
#endif

		// process the args before printing anything so we can handle "-colour" and "-nocolour" correctly
		auto & cfg_and_state = Darknet::CfgAndState::get();
		cfg_and_state.set_thread_name("main darknet thread");
		cfg_and_state.process_arguments(argc, argv);

		#ifdef _DEBUG
		_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
		Darknet::display_warning_msg("DEBUG is used\n");
		#endif

		#ifdef DEBUG
		Darknet::display_warning_msg("DEBUG=1 is enabled\n");
		#endif

		errno = 0;

		std::cout << "Darknet V2 \"" << DARKET_VERSION_KEYWORD << "\" " << Darknet::in_colour(Darknet::EColour::kBrightWhite, DARKNET_VERSION_STRING) << std::endl;

		cfg_and_state.gpu_index = find_int_arg(argc, argv, "-i", 0);

#ifndef GPU
		cfg_and_state.gpu_index = -1;
		Darknet::display_warning_msg("Darknet is compiled to only use the CPU.");
		std::cout << "  GPU is " << Darknet::in_colour(Darknet::EColour::kBrightRed, "disabled") << "." << std::endl;
		init_cpu();
#else   // GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			cuda_set_device(cfg_and_state.gpu_index);
			CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
		}

		show_cuda_cudnn_info();
		cuda_debug_sync = find_arg(argc, argv, "-cuda_debug_sync");
#endif

		show_opencv_info();

		errno = 0;

		if		(cfg_and_state.command.empty())				{ Darknet::display_usage();			}
		else if (cfg_and_state.command == "3d")				{ composite_3d		(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0); }
		else if (cfg_and_state.command == "art")			{ run_art			(argc, argv);	}
		else if (cfg_and_state.command == "average")		{ average			(argc, argv);	}
		else if (cfg_and_state.command == "captcha")		{ run_captcha		(argc, argv);	}
		else if (cfg_and_state.command == "cifar")			{ run_cifar			(argc, argv);	}
		else if (cfg_and_state.command == "classify")		{ predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5); }
		else if (cfg_and_state.command == "classifier")		{ run_classifier	(argc, argv);	}
		else if (cfg_and_state.command == "coco")			{ run_coco			(argc, argv);	}
		else if (cfg_and_state.command == "compare")		{ run_compare		(argc, argv);	}
		else if (cfg_and_state.command == "denormalize")	{ denormalize_net	(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "detector")		{ run_detector		(argc, argv);	}
		else if (cfg_and_state.command == "dice")			{ run_dice			(argc, argv);	}
		else if (cfg_and_state.command == "imtest")			{ test_resize		(argv[2]);		}
		else if (cfg_and_state.command == "go")				{ run_go			(argc, argv);	}
		else if (cfg_and_state.command == "help")			{ Darknet::display_usage();			}
		else if (cfg_and_state.command == "nightmare")		{ run_nightmare		(argc, argv);	}
		else if (cfg_and_state.command == "normalize")		{ normalize_net		(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "oneoff")			{ oneoff			(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "ops")			{ operations		(argv[2]); }
		else if (cfg_and_state.command == "partial")		{ partial			(argv[2], argv[3], argv[4], atoi(argv[5])); }
		else if (cfg_and_state.command == "rescale")		{ rescale_net		(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "reset")			{ reset_normalize_net(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "rgbgr")			{ rgbgr_net			(argv[2], argv[3], argv[4]); }
		else if (cfg_and_state.command == "rnn")			{ run_char_rnn		(argc, argv);	}
		else if (cfg_and_state.command == "speed")			{ speed				(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0); }
		else if (cfg_and_state.command == "statistics")		{ statistics_net	(argv[2], argv[3]); }
		else if (cfg_and_state.command == "super")			{ run_super			(argc, argv);	}
		else if (cfg_and_state.command == "tag")			{ run_tag			(argc, argv);	}
		else if (cfg_and_state.command == "test")			{ test_resize		(argv[2]);		}
		else if (cfg_and_state.command == "version")		{ /* nothing else to do, we've already displayed the version information */ }
		else if (cfg_and_state.command == "vid")			{ run_vid_rnn		(argc, argv);	}
		else if (cfg_and_state.command == "visualize")		{ visualize			(argv[2], (argc > 3) ? argv[3] : 0); }
		else if (cfg_and_state.command == "voxel")			{ run_voxel			(argc, argv);	}
		else if (cfg_and_state.command == "writing")		{ run_writing		(argc, argv);	}
		else if (cfg_and_state.command == "yolo")			{ run_yolo			(argc, argv);	}
		else if (cfg_and_state.command == "detect")
		{
			float thresh = find_float_arg(argc, argv, "-thresh", .24);
			int ext_output = find_arg(argc, argv, "-ext_output");
			char *filename = (argc > 4) ? argv[4]: 0;
			test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, 0.5, 0, ext_output, 0, NULL, 0, 0);
		}
		else
		{
			throw std::invalid_argument("invalid command (run \"" + cfg_and_state.argv[0] + " help\" for a list of possible commands)");
		}
	}
	catch (const std::exception & e)
	{
		std::cout << std::endl << "Exception: " << Darknet::in_colour(Darknet::EColour::kBrightRed, e.what()) << std::endl;
		darknet_fatal_error(DARKNET_LOC, e.what());
	}

	return 0;
}
