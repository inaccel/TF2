#include "includes.h"

Model::Model(const std::string& model_file, const std::string& q_file) {
	this->model_file = model_file;
	this->q_file = q_file;

	const int filter_raw_size =  NUM_CONVOLUTIONS * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
	filter_raw = (real*)alignedMalloc(sizeof(real) * filter_raw_size);
	if (filter_raw== NULL) ERROR("Cannot allocate enough space for filter_raw\n");
	memset(filter_raw, 64, sizeof(real) * filter_raw_size);

	const int bias_bn_size = NUM_CONVOLUTIONS * MAX_BIAS_SIZE;
	bias_bn = new inaccel::vector<BiasBnParam>(bias_bn_size);

	q = (char *)alignedMalloc(sizeof(char) * NUM_Q_LAYERS * MAX_OUT_CHANNEL);
    Quantization(q, q_file);

	INFO("Loading convolutional layer params...\n");
	LoadModel(model_file, filter_raw, bias_bn->data(), q);

	const int filter_device_size = NUM_CONVOLUTIONS * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
	filter = (real*)alignedMalloc(sizeof(real) * filter_device_size);
	if (filter == NULL) ERROR("Cannot allocate enough space for filter.\n");

	filter_real = new inaccel::vector<real>(filter_device_size);
	memset(filter_real->data(), 64, sizeof(real) * filter_device_size);

	FilterConvert(filter, filter_raw, filter_real->data());

	wait_after_conv_cycles = new inaccel::vector<int>(NUM_CONVOLUTIONS);
	memcpy(wait_after_conv_cycles->data(), &kSequencerIdleCycle[0], sizeof(int) * NUM_CONVOLUTIONS);
}

Model::~Model() {
	// host buffers
	if (q) free(q);
	if (filter_raw) free(filter_raw);
	if (filter) free(filter);

	//host/fpga shared buffers
	if (bias_bn) delete bias_bn;
	if (filter_real) delete filter_real;
	if (wait_after_conv_cycles) delete wait_after_conv_cycles;
}
