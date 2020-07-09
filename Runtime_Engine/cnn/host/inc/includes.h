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

#ifndef __INCLUDES_H__
#define __INCLUDES_H__

// common includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <vector>
#include <assert.h>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "debug.h"

typedef struct{
  int index;
  std::string label_name;
} imagenet_content;

// cnn includes
#include "cnn.h"

#include "model_loader.h"
#include "input_loader.h"
#include "quantization.h"

#include "network_helper.h"
#include "model.h"
#include "runner.h"

void *alignedMalloc(size_t size);
void read_directory(const std::string& name, std::vector<std::string>& v);

#endif
