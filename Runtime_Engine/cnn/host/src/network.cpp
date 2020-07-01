/* Copyright 2019 Inspur Corporation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "includes.h"

NetWork::NetWork() {

}

bool NetWork::Init(char *model_file, char* q_file, char *image_file, int num_images) {
  this->model_file = model_file;
  this->q_file = q_file;
  this->image_file = image_file;
  this->num_images = num_images;

  if (!(InitNetwork())) {
    return -1;
  }

  return 0;
}

bool NetWork::InitNetwork() {
  //
  // input array
  //

  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];
  int HXW = H * W;

  const unsigned long long int input_raw_size = C * HXW * num_images;

  input_raw = (float*)alignedMalloc(sizeof(float)* input_raw_size);

  if (input_raw == NULL) printf("Cannot allocate enough space for input_raw\n");
  memset(input_raw, 0, sizeof(float) * input_raw_size);

  INFO("Loading input image binary...\n");

  const int input_device_size = CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) * num_images;
  input = (float*)alignedMalloc(sizeof(float) * input_device_size);
  if (input == NULL) ERROR("Cannot allocate enough space for input\n");
  memset(input, 0, sizeof(float) * input_device_size);

  input_real = new inaccel::vector<real>;
  input_real->resize(input_device_size);

  // filter array(incldue bias array)
  const int filter_raw_size =  NUM_CONVOLUTIONS * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
  filter_raw = (real*)alignedMalloc(sizeof(real) * filter_raw_size);
  if (filter_raw== NULL) ERROR("Cannot allocate enough space for filter_raw\n");
  memset(filter_raw, 64, sizeof(real) * filter_raw_size);

  const int bias_bn_size = NUM_CONVOLUTIONS * MAX_BIAS_SIZE;
  bias_bn = new inaccel::vector<BiasBnParam>;
  bias_bn->resize(bias_bn_size);

  // compute Quantization param
  // it can compute only once
  float *input_raw_images = (float*)malloc(sizeof(float) * INPUT_IMAGE_C * INPUT_IMAGE_H * INPUT_IMAGE_W * 2);

  q = (char *)alignedMalloc(sizeof(char) * NUM_Q_LAYERS * MAX_OUT_CHANNEL);
  Quantization(q, input_raw_images, q_file);

  INFO("Loading convolutional layer params...\n");
  //LoadModel(model_file, filter_raw, bias,alpha,beta, q);
  LoadModel(model_file, filter_raw, bias_bn->data(), q);

  const int filter_device_size = NUM_CONVOLUTIONS * MAX_FILTER_SIZE * NEXT_POWER_OF_2(FW_VECTOR * C_VECTOR);
  filter = (real*)alignedMalloc(sizeof(real) * filter_device_size);
  if (filter == NULL) ERROR("Cannot allocate enough space for filter.\n");

  filter_real = new inaccel::vector<real>;
  filter_real->resize(filter_device_size);
  memset(filter_real->data(), 64, sizeof(real) * filter_device_size);

  FilterConvert(filter, filter_raw, filter_real->data());

  wait_after_conv_cycles = new inaccel::vector<int>;
  wait_after_conv_cycles->resize(NUM_CONVOLUTIONS);
  memcpy(wait_after_conv_cycles->data(), &kSequencerIdleCycle[0], sizeof(int) * NUM_CONVOLUTIONS);

  // output array
  //const int feature_ddr_size = DDR_SIZE * NEXT_POWER_OF_2(W_VECTOR) * NEXT_POWER_OF_2(C_VECTOR);
  const int feature_ddr_size = OUTPUT_OFFSET + num_images * OUTPUT_OFFSET;
  output = new inaccel::vector<real>;
  output->resize(feature_ddr_size);
}

void NetWork::CleanUp() {
  // host buffers
  if (input_raw) free(input_raw);
  if (input) free(input);
  if (q) free(q);
  if (filter_raw) free(filter_raw);
  if (filter) free(filter);

  //host/fpga shared buffers
  delete bias_bn;
  delete filter_real;
  delete input_real;
  delete output;
  delete wait_after_conv_cycles;
}
