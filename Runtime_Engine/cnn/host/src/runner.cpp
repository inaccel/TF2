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

#include "runner.h"

Runner::Runner(NetWork &network) {
  this->network = network;
}

void Runner::Init() {
  this->image_file = network.image_file;
  this->num_images = network.num_images;
}

void Runner::Run() {
  //
  // set args
  //

  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];
  int HXW = H * W;

  const int input_device_size = CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) * num_images;

  float *input_raw_images = (float*)malloc(sizeof(float) * INPUT_IMAGE_C * INPUT_IMAGE_H * INPUT_IMAGE_W * 2);

  for (int i = 0; i < num_images; i++) {
    LoadInputImage(image_file, network.input_raw + i * C * HXW, input_raw_images, 0);
  }

  InputConvert(network.input_raw, network.input, num_images);

  float trans = 1.0f / ( 1 << network.q[0]);
  for (int i = 0; i < input_device_size; i++) {
    float tmp = network.input[i] * trans;
    int tmp_int = (int)(tmp > 0 ? tmp + 0.5 : tmp - 0.5);
    (network.input_real->data())[i] = tmp_int > REALMAX ? REALMAX : tmp_int < REALMIN ? REALMIN : tmp_int;
  }

  inaccel::request resnet50("com.inspur.tf2.resnet50");

  resnet50.arg(*(network.input_real))
    .arg(*(network.filter_real))
    .arg(*(network.bias_bn))
    .arg(*(network.wait_after_conv_cycles))
    .arg(*(network.output))
    .arg(num_images);

  inaccel::wait(inaccel::submit(resnet50));
}
