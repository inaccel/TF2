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

#ifndef __MODEL_H__
#define __MODEL_H__

#include <inaccel/coral>
#include "includes.h"

class Model {
public:
    Model(const std::string &model_file, const std::string &q_file);
    ~Model();

    char* q = NULL;

    inaccel::vector<BiasBnParam> *bias_bn = NULL;
    inaccel::vector<real> *filter_real = NULL;
    inaccel::vector<int> *wait_after_conv_cycles = NULL;

    // int top_labels[5];

private:
    std::string model_file;
    std::string q_file;
    real* filter_raw = NULL;
    real* filter = NULL;
};

#endif
