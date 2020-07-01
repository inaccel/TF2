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

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <inaccel/coral>
#include "includes.h"

// Image loading thread args
struct ImageLoadArgs {
  char *path;
  int size;
  int raw_size;
  int num_images;
};

class NetWork {
public:
  NetWork();
  bool Init(char *model_file, char* q_file, char *image_file, int num_images);
  bool InitNetwork();
  void CleanUp();

  char *model_file;
  char *q_file;
  char *image_file;
  int num_images;
  float* input_raw = NULL;
  float* input = NULL;
  char* q = NULL;
  int top_labels[5];

  // Host/FPGA shared buffers
  inaccel::vector<real> *input_real = NULL;
  inaccel::vector<int> *wait_after_conv_cycles = NULL;
  inaccel::vector<real> *filter_real = NULL;
  inaccel::vector<BiasBnParam> *bias_bn = NULL;
  inaccel::vector<real> *output = NULL;

private:
  real* filter_raw = NULL;
  real* filter = NULL;
};

#endif
