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

int main(int argc, char **argv) {
    if (argc != 5) {
        INFO("USAGE:\n%s <model_file> <quantization_file> <images_dir> <batch_size>\n", argv[0]);
        return 1;
    }

    std::string model_file = std::string(argv[1]);
    std::string q_file = std::string(argv[2]);
    std::string images_dir = std::string(argv[3]);
    int batch_size = atoi(argv[4]);

    INFO("model_file = %s\n", model_file.c_str());
    INFO("q_file = %s\n", q_file.c_str());
    INFO("images_dir  = %s\n", images_dir.c_str());
    INFO("batch_size = %d\n", batch_size);

    auto start = std::chrono::high_resolution_clock::now();

    // Prepare filter_real, bias_bn, wait_after_conv_cycles for all requests
    Model model(model_file, q_file);

    // Read all the (image) file names under given directory
    std::vector<std::string> image_files;
    read_directory(images_dir, image_files);

    int num_batches = CEIL(image_files.size(), batch_size);

    // Read the image labels from 'imagenet1000_clsid_to_human.txt'
    std::vector<imagenet_content> image_labels;
    LoadLabel_imagenet(image_labels);

    // Run on the FPGA issuing run function on batches in parallel
    #pragma omp parallel for num_threads(16)
    for (int idx = 0; idx < num_batches; idx++) {
        run(model, image_files, image_labels, idx, batch_size);
    }

    auto end = std::chrono::high_resolution_clock::now();

    float seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;

    std::cout << "\n -- Throughput Summary:\n";
    std::cout << "       Total duration:   " << seconds << "s\n";
    std::cout << std::setprecision(4);
    std::cout << "       Avg throughput:   " << image_files.size() / seconds << " fps\n\n";

    return 0;
}
