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

#ifndef __NETWORK_HELPER_H__
#define __NETWORK_HELPER_H__

#include "includes.h"

typedef struct {
  int label;
  float feature;
} StatItem;

void Verify(int n, char *file_name, char *q, real *output);
void Evaluation(int n, char *q, const std::string &image_file, const std::vector<imagenet_content> &imagecontents, const real* output);
void LoadLabel(int Num,int *labels);
void LoadLabel_imagenet(std::vector<imagenet_content> &imagecontents);

#endif
