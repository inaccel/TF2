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

#include "network_helper.h"

void Verify(int n, char *file_name, char *q, real *output) {
#ifdef CONCAT_LAYER_DEBUG
  int output_channel = kNEnd[NUM_LAYER - 1];
#else
  int output_channel = kOutputChannels[NUM_LAYER - 1];
#endif

  int width = kPoolOutputWidth[NUM_LAYER - 1];
  int height = kPoolOutputHeight[NUM_LAYER - 1];

  int size = output_channel * width * height;
  float *expect = (float*)alignedMalloc(sizeof(float) * size);
  if (!expect) {
    ERROR("malloc expect error!\n");
  }

  FILE *fp;
  fp = fopen(file_name, "r");
  if (!fp) {
    ERROR("fopen file error!\n");
  }
  size_t bytes = fread(expect, sizeof(float), size, fp);
  if(!bytes) exit(1);

  fclose(fp);

  std::string output_file_name = "Lastconv" + std::to_string(n) + ".dat";

  FILE *fp_output;
  fp_output = fopen(output_file_name.c_str(), "wt");
  int H = height;
  int W = width;
#ifdef CONCAT_LAYER_DEBUG
  int output_offset = 0;
  int concat_offset = 0;
#else
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;
  int concat_offset = kNStart[NUM_LAYER - 1] / NARROW_N_VECTOR * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
#endif
  int ddr_write_offset = kDDRWriteBase[NUM_LAYER - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;
  for (int n = 0; n < output_channel; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int n_vec = n / NARROW_N_VECTOR;
        int h_vec = h;
        int w_vec = w / W_VECTOR;
        int ww = w - w_vec * W_VECTOR;
        int nn = n - n_vec * NARROW_N_VECTOR;
        int addr_out =
                    output_offset + concat_offset + ddr_write_offset +
                    n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                    ww * NARROW_N_VECTOR +
                    nn;;
        int addr_exp = n * H * W + h * W + w;
        float expect_val = expect[addr_exp];
#ifdef CONCAT_LAYER_DEBUG
       int current_q = q[(NUM_CONVOLUTIONS + 1 + kConcatLayer[NUM_LAYER - 1]) * MAX_OUT_CHANNEL + n];
#else
       int current_q = q[NUM_LAYER * MAX_OUT_CHANNEL + n];
#endif
        float trans = 1 << (-current_q); //take care of shortcut
        float check_error=fabs(expect_val*trans-output[addr_out]);
        total_error += check_error;
        total_expect += fabs(expect_val*trans);
        {
          fprintf(fp_output,"error=%.6f expect1=%f q=%d expect_trans=%.6f output=%.6f addr=%d n=%d h=%d w=%d\n", check_error, expect_val, current_q, expect_val*trans, 1.0*output[addr_out], addr_out, n, h, w);
        }
      }
    }
  }
  fclose(fp_output);

  INFO("Convolution %d compare finished, error=%f\n", NUM_LAYER, total_error / total_expect);
  if (expect) free(expect);
}

void Evaluation(int n, char* q, const std::string &image_file, const std::vector<imagenet_content> &imagecontents, const real* output) {
  int output_channel = kOutputChannels[NUM_LAYER - 1];
  int width = 1;
  int height = 1;

  int size = output_channel * width * height;

  FILE *fp;
  fp = fopen("Lastconv.dat","wt");
  int H = height;
  int W = width;
  int ddr_write_offset = kDDRWriteBase[NUM_LAYER - 1] * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR);
  int output_offset = OUTPUT_OFFSET + n * OUTPUT_OFFSET;

  int pad = 0;

  float total_error = 0.;
  float total_expect = 0.;

  std::vector<StatItem> stat_array;

  float sum_exp = 0;

  for (int n = 0; n < output_channel; n++) {
    int n_vec = n / NARROW_N_VECTOR;
    int h_vec = 0;
    int w = 0;
    int w_vec = 0;
    int ww = w - w_vec * W_VECTOR;
    int nn = n - n_vec * NARROW_N_VECTOR;
    int addr_out =
                ddr_write_offset + output_offset +
                n_vec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                h_vec * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                w_vec * NEXT_POWER_OF_2(W_VECTOR * NARROW_N_VECTOR) +
                ww * NARROW_N_VECTOR +
                nn;;
    int current_q = q[NUM_LAYER * MAX_OUT_CHANNEL + n];
    float trans = 1 << (-current_q); //take care of shortcut

    StatItem temp;
    temp.label = n;
    temp.feature = output[addr_out] / trans;

    sum_exp += exp(temp.feature);

    stat_array.push_back(temp);
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < output_channel - i - 1; j++) {
      if (stat_array[j].feature > stat_array[j + 1].feature) {
        std::swap(stat_array[j], stat_array[j + 1]);
      }
    }
  }

  INFO("IMAGE: [%s]\n", image_file.c_str());
  for (int i = 0; i < 5; i++) {
    INFO("\t[Rank %d] [Label: %3d] [Prediction: %s] [Probability: %f]\n", i, stat_array[output_channel - i - 1].label, imagecontents[stat_array[output_channel - i - 1].label].label_name.c_str(), exp(stat_array[output_channel - i - 1].feature) / sum_exp);
  }

  fclose(fp);
}

void LoadLabel(int Num,int *labels) {
  FILE *fp;
  if ((fp = fopen("label.dat", "r")) == NULL) {
    ERROR("Error in search label.dat\n");
    exit(0);
  }

  for (int i = 0; i < Num; i++) {
    int items = fscanf(fp,"%d",&labels[i]);
    if (!items) exit(1);
  }

  fclose(fp);
}

void LoadLabel_imagenet(std::vector<imagenet_content> &imagecontents) {
	std::ifstream fin("host/model/imagenet1000_clsid_to_human.txt");
	std::string line;
	int position, pos_s_m_start, pos_d_m_start, pos_m_end, pos_q_end;

	if(fin) {
		while(getline(fin,line)) {
			imagenet_content image_item;
			position = line.find(": ");

			std::string imgnet_mid_arr_index = line.substr(0,position);

			std::string imgnet_mid_arr_content_mid = line.substr(position+2,line.size()-1);

			std::string imgnet_mid_arr_content_q, imgnet_mid_arr_content;

			pos_d_m_start = imgnet_mid_arr_content_mid.find_first_of("\"");

			if(pos_d_m_start != -1) {
				pos_m_end = imgnet_mid_arr_content_mid.find_last_of("\"");
				imgnet_mid_arr_content_q = imgnet_mid_arr_content_mid.substr(pos_d_m_start+1,pos_m_end-1);
			} else {
				pos_s_m_start = imgnet_mid_arr_content_mid.find_first_of("\'");
				pos_m_end = imgnet_mid_arr_content_mid.find_last_of("\'");
				imgnet_mid_arr_content_q = imgnet_mid_arr_content_mid.substr(pos_s_m_start+1,pos_m_end-1);
			}

			pos_q_end = imgnet_mid_arr_content_q.find_first_of(",");

			if( pos_q_end != -1) {
				imgnet_mid_arr_content = imgnet_mid_arr_content_q.substr(0,pos_q_end);
			} else {
				imgnet_mid_arr_content = imgnet_mid_arr_content_q;
			}

			image_item.index = atoi(imgnet_mid_arr_index.c_str());
			image_item.label_name = imgnet_mid_arr_content.c_str();
			imagecontents.push_back(image_item);
		}
	} else {
		/* code */
		std::cout << "fail to open file imagenet1000.txt" << std::endl;
		exit(1);
	}

	fin.close();
}
