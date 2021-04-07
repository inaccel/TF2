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

void load_images(const char *q, std::vector<std::string>::iterator image_files, inaccel::vector<real> **input_real, int num_images) {
    int cnt = 0;
    const int C = kInputChannels[0];
    const int H = kInputHeight[0];
    const int W = kInputWidth[0];
    const int HXW = H * W;
    const unsigned long long int input_raw_size = C * HXW * num_images;
    const int input_device_size = CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) * num_images;

    // Allocate arrays/vectors
    float *input_raw = (float *) alignedMalloc(sizeof(float) * input_raw_size);
    if (input_raw == NULL) printf("Cannot allocate enough space for input_raw\n");

    float *input = (float *) alignedMalloc(sizeof(float) * input_device_size);
    if (input == NULL) ERROR("Cannot allocate enough space for input\n");

    *input_real = new inaccel::vector<real>(input_device_size);

    memset(input_raw, 0, sizeof(float) * input_raw_size);
    memset(input, 0, sizeof(float) * input_device_size);

    INFO("Loading input image(s)...\n");
    for (auto imageIt = image_files; cnt < num_images; imageIt++, cnt++)
        LoadInputJpeg(*imageIt, input_raw + cnt * C * HXW);

    InputConvert(input_raw, input, num_images);

    float trans = 1.0f / ( 1 << q[0]);
    for (int i = 0; i < input_device_size; i++) {
        float tmp = input[i] * trans;
        int tmp_int = (int)(tmp > 0 ? tmp + 0.5 : tmp - 0.5);
        (**input_real)[i] = tmp_int > REALMAX ? REALMAX : tmp_int < REALMIN ? REALMIN : tmp_int;
    }

    free(input_raw);
    free(input);
}

void run_on_fpga(Model &model, inaccel::vector<real> *input, inaccel::vector<real> *output, int num_images) {
    inaccel::request resnet50("com.inspur.tf2.resnet50");

    resnet50.arg(*(input))
    .arg(*(model.filter_real))
    .arg(*(model.bias_bn))
    .arg(*(model.wait_after_conv_cycles))
    .arg(*(output))
    .arg(num_images);

    inaccel::submit(resnet50).get();
}

void predict(char *q, std::vector<std::string>::iterator image_files, std::vector<imagenet_content> &imagecontents, const real *results, int num_images) {
    int cnt = 0;
    for (auto it = image_files; cnt < num_images; it++, cnt++) {
        Evaluation(cnt, q, *(it), imagecontents, results);
    }
}

void run(Model &model, std::vector<std::string> image_files, std::vector<imagenet_content> &imagecontents, int idx, int batch_size) {
    auto startImage = image_files.begin() + idx * batch_size;
    int remaining_images = std::distance(startImage, image_files.end());
    int num_images = remaining_images > batch_size ? batch_size : remaining_images;

    const int feature_ddr_size = OUTPUT_OFFSET + num_images * OUTPUT_OFFSET;

    inaccel::vector<real> *input = nullptr;
    inaccel::vector<real> *output = new inaccel::vector<real>(feature_ddr_size);

    load_images(model.q, startImage, &input, num_images);

    run_on_fpga(model, input, output, num_images);

    predict(model.q, startImage, imagecontents, output->data(), num_images);

    delete input;
    delete output;
}
