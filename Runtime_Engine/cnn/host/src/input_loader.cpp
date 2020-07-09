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

#include "input_loader.h"

#define DIM 224
#define PAD_H_0 3
#define PAD_W_0 3
#define NDIM_H (DIM+2*PAD_H_0)
#define NDIM_W (DIM+2*PAD_W_0)
#define OUTPUT_WIDTH (NDIM_W/2)
#define OUTPUT_HEIGHT (NDIM_H/2)
#define OUTPUTDIMS 114  // 114-3+1=112;

// Implement OpenCV custom handler to suppress console output in errors.
// This is just a quick workardound to support both LoadInputJpeg and LoadInputImage
// without having the user specify if the input images are in JPEG format or not.
int handleError( int status, const char* func_name, const char* err_msg, const char* file_name, int line, void* userdata ) {
    //Do nothing
    return 0;
}

void feature_trans(float *Feas,float *finals_feas)
{
  float Trans_Pad[NDIM_H][NDIM_W] = {{0.0}};
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      Trans_Pad[i + PAD_H_0][j + PAD_W_0] = Feas[i * DIM + j];
    }
  }

  float Fea_media[3][NDIM_H][NDIM_W] = {{{0.0}}};
  for (int i = 0; i < NDIM_H; i++) {
    for (int j = 0; j < NDIM_W; j++) {
      Fea_media[j % 2][i][j / 2] = Trans_Pad[i][j];
    }
  }

  for (int i = 0; i< NDIM_H; i++) {
    for (int j = 0; j < NDIM_W - 1; j++) {
      Fea_media[2][i][j] = Fea_media[0][i][j + 1];
    }
  }

  float height_trans[6][NDIM_H / 2][NDIM_W / 2] = {{{0}}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < NDIM_H; j++) {
      for (int k = 0; k < NDIM_W / 2; k++) {
        height_trans[i * 2 + j % 2][j / 2][k] = Fea_media[i][j][k];
      }
    }
  }

  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < NDIM_H / 2; j++) {
      for (int k = 0; k < NDIM_H / 2; k++) {
        finals_feas[i * OUTPUT_HEIGHT * OUTPUT_WIDTH + j * OUTPUT_WIDTH + k]=height_trans[i][j][k];
      }
    }
 }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < NDIM_H / 2; j++) {
      for (int k = 0; k < NDIM_W / 2; k++) {
        finals_feas[(i + 6) * OUTPUT_HEIGHT * OUTPUT_WIDTH + j * OUTPUT_WIDTH + k] = height_trans[i * 2][j + 1][k];
      }
    }
  }
}

void LoadInputJpeg(const std::string &image_name, float *input_raw) {
  INFO("LoadInputJpeg image_name=%s\n", image_name.c_str());

  std::vector<cv::Mat> channels;
  cv::Mat src, dst1, dst2;

  unsigned char C = 3;
  unsigned char H = 224;
  unsigned char W = 224;

  int  H_middle = 256;
  int  W_middle = 256;

  float googlenet_mean[3] = {104, 117, 123};

  FILE *fp;
  fp = fopen( "host/model/mean.bin", "rb" );
  if (!fp) exit(1);

  // 256 * 256 preprocess
  try {
    cv::redirectError(handleError);
    src = cv::imread( image_name );
    resize( src, dst2, cvSize( H_middle, W_middle ) );
    cv::split( dst2, channels );
  } catch (...) {
     fclose(fp);
     INFO("Falling back to LoadInputImage for image: %s\n", image_name.c_str());
     LoadInputImage(image_name, input_raw);
     return;
  }

  float *middle_images = (float*) malloc( sizeof(float) * C * H_middle * W_middle );

  float *raw_images = (float*) malloc(sizeof(float) * INPUT_IMAGE_C * INPUT_IMAGE_H * INPUT_IMAGE_W * 2);
  if (raw_images == NULL) ERROR("Cannot allocate enough space for raw_images\n");

  for(int c = 0; c < C; c++) {
    for(int h = 0; h < H_middle; h++) {
      for(int w = 0; w < W_middle; w++) {
        int addr = c * H_middle * W_middle + h * W_middle + w;
        float mean;
#if (defined RESNET50) || (defined RESNET50_PRUNED)
        size_t bytes = fread( &mean, sizeof(float), 1, fp );
        if (!bytes) exit(1);
#else
        mean = googlenet_mean[c];
#endif
        middle_images[addr] = channels[c].at<uchar>(h,w) - mean;
        //printf( "c=%d h=%d w=%d middle_images[%d]=%f mean=%f\n", c, h, w, addr, middle_images[addr], mean );
      }
    }
  }

  // crop operation
  int H_edge = ( H_middle - H ) / 2;
  int W_edge = ( W_middle - W ) / 2;

  for(int c = 0; c < C; c++) {
    for(int h = 0; h < H; h++) {
      for(int w = 0; w < W; w++) {
        int addr1 = c * H_middle * W_middle + ( h + H_edge ) * W_middle + ( w + W_edge);
        int addr2 = c * H * W + h * W + w;
        raw_images[addr2] = middle_images[addr1];
        //printf( "c=%d h=%d w=%d addr1=%d raw_images[%d]=%f\n", c, h, w, addr1, addr2, raw_images[addr2] );
      }
    }
  }

  unsigned char CC = 27;
  unsigned char HH = 115;
  unsigned char WW = 115;
  float *new_input=(float*)malloc(sizeof(float) * CC * HH * WW + 1000);
  for(int i = 0; i < 3; i++) {
    feature_trans(raw_images + i * H * W, new_input + 9 * i * OUTPUT_HEIGHT * OUTPUT_WIDTH);
  }

  for(int i = 0; i < 27; i++) {
    for(int j = 0; j < 114; j++) {
      for(int k = 0; k < 114; k++) {
        input_raw[i * 114 * 114 + j * 114 + k]=new_input[i * WW * HH + j * WW + k];
      }
    }
  }

  fclose(fp);
  free(middle_images);
  free(new_input);
  free(raw_images);
}

// load input raw image data:[C][H][W]
void LoadInputImage(const std::string &image_name, float *input_raw) {
  INFO("LoadInputImage image_name=%s\n", image_name.c_str());
  unsigned char C = INPUT_IMAGE_C;
  unsigned char H = INPUT_IMAGE_H;
  unsigned char W = INPUT_IMAGE_W;

  float *raw_images = (float*) malloc(sizeof(float) * INPUT_IMAGE_C * INPUT_IMAGE_H * INPUT_IMAGE_W * 2);
  if (raw_images == NULL) ERROR("Cannot allocate enough space for raw_images\n");

  FILE *fp;
  if ((fp = fopen(image_name.c_str(), "rb"))==NULL){
    printf("load input image : %s Error\n",image_name.c_str());
    exit(1);
  }

  for (int c = 0; c < C; c++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int addr = c * H * W + h * W + w;
        size_t bytes = fread( &raw_images[addr], sizeof(float), 1, fp );
        if(!bytes) exit(1);
      }
    }
  }
  fclose(fp);

  // The below code is just for Resnet50 and GoogLeNet and will remove later.
  unsigned char CC = 27;
  unsigned char HH = 115;
  unsigned char WW = 115;
  float *new_input=(float*)malloc(sizeof(float) * CC * HH * WW + 1000);
  for (int i = 0; i < 3; i++) {
    feature_trans(raw_images + i * H * W, new_input + 9 * i * OUTPUT_HEIGHT * OUTPUT_WIDTH);
  }

  for (int i = 0; i < 27; i++) {
    for (int j = 0; j < 114; j++) {
      for (int k = 0; k < 114; k++) {
        input_raw[i * 114 * 114 + j * 114 + k]=new_input[i * WW * HH + j * WW + k];
      }
    }
  }

  free(new_input);
  free(raw_images);
}

// input_raw[C][H][W]
// to
// input[CEIL(C, C_VECTOR)][H][CEIL(W, W_VECTOR)][W_VECTOR][C_VECTOR]
void InputConvert(float *input_raw, float *input, int num_images)
{
  int C = kInputChannels[0];
  int H = kInputHeight[0];
  int W = kInputWidth[0];

  for ( int n = 0; n < num_images; n++ ) {
    for (int cvec = 0; cvec < CEIL(C, C_VECTOR); cvec++) {
      for (int h = 0; h < H; h++) {
        for (int ww = 0; ww < CEIL(W, W_VECTOR); ww++) {
          for (int wvec = 0; wvec < W_VECTOR; wvec++) {
            for (int c = 0; c < C_VECTOR; c++) {
              unsigned long long int addr = (unsigned long long int)
                                            n * CEIL(C, C_VECTOR) * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            cvec * H * CEIL(W, W_VECTOR) * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            h * CEIL(W, W_VECTOR) *NEXT_POWER_OF_2 (W_VECTOR * C_VECTOR) +
                                            ww * NEXT_POWER_OF_2(W_VECTOR * C_VECTOR) +
                                            wvec * C_VECTOR +
                                            c;

             int linear_c = cvec * C_VECTOR + c;
             int linear_w = ww * W_VECTOR + wvec;

             bool not_out_of_bounds = (linear_c < C && linear_w < W);
             unsigned long long int input_raw_addr = (unsigned long long int) linear_c * H * W + h * W + linear_w;

             input[addr] = not_out_of_bounds ? input_raw[input_raw_addr] : 0.0;
            }
          }
        }
      }
    }
  }
}
