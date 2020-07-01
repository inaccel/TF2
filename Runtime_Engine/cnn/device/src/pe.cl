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


#ifndef OPENCL
#define OPENCL
#endif

#include "../../host/inc/cnn.h"

// Functions:
// 1. Performs convolution operations in a shifting manner.
// 2. Sends data in daisy-chains through convolution kernels

inline Mreal MUL(real feature, real filter) {
  if (BIT_IS_SET(filter, 6)) {
    return 0;
  }

  if (BIT_IS_SET(filter, 7)) {
    feature = -feature;
  }

  filter = 0x1f & filter;
  Mreal data = feature << filter;

  return data;
}

STATIC Mreal DotProduct(DotVector feature_values, DotVector filter_values) {
  int dot_accum = 0; // change from long int to int
  #pragma unroll
  for (int c_inc = 0; c_inc < C_VECTOR; c_inc++)
    dot_accum += MUL(feature_values.v[c_inc], filter_values.v[c_inc]);

  return dot_accum;
}

// this function is the prototype code for each pe kernels in N_VECTOR pe arrays
void PeFunction(int n_inc) {
  int pe_input_data_channel_first_cnt = 0;

  // filter buffer
  DotVector filter_cache[FILTER_CACHE_DEPTH][NEXT_POWER_OF_2(FW_VECTOR)];
  BiasBnParam bias_bn_cache[DOUBLE_BUFFER_DIM];

  INIT_COUNTER(cycle);

  Mreal result[W_VECTOR]={0};
  int cycle_end = FILTER_PRELOAD_CYCLE + CONV_TOTAL_CYCLE;

#ifdef PRINT_PE_INPUT
  int debug_cycle = FILTER_PRELOAD_CYCLE + find_conv_layer_cycles(NUM_LAYER - 1);
  int debug_range = 100000;
#endif

  #pragma ivdep
  do {
    SET_COUNTER(cycle, cycle_end, 0, cycle_end, 1);
    bool conv_done;

    //printf("pe cycle=%d/%d\n", cycle, cycle_end);

    PeInputData pe_in;
    PeInputFilter pe_filter;
    PeControlSignal cont;

    if (n_inc == 0) {
      cont      = read_channel_intel(pe_control_channel_first);
      pe_filter = read_channel_intel(pe_input_filter_channel_first);
      pe_in     = read_channel_intel(pe_input_data_channel_first);
    } else {
      cont      = read_channel_intel(pe_control_channel[n_inc-1]);
      pe_filter = read_channel_intel(pe_input_filter_channel[n_inc-1]);
      pe_in     = read_channel_intel(pe_input_data_channel[n_inc-1]);
    }

    write_channel_intel(pe_control_channel[n_inc],      cont);
    write_channel_intel(pe_input_filter_channel[n_inc], pe_filter);
    write_channel_intel(pe_input_data_channel[n_inc],   pe_in);

    // input feature map data
    DotFeatureVector input_data = pe_in.input_data;
    bool input_data_valid = pe_in.input_data_valid;

    // filter data
    DotFilterVector filter_data = pe_filter.filter_data;
    BiasBnParam bias_bn_data = pe_filter.bias_bn_data;
    bool filter_data_valid = pe_filter.data_valid;
    bool filter_bias_read_page = cont.filter_bias_read_page;
    bool filter_bias_write_page = !filter_bias_read_page;
    int  filter_read_addr = cont.filter_read_addr;
    char filter_read_fw_vec = cont.filter_read_fw_vec;
    int  filter_write_addr = cont.filter_write_addr;
    int  filter_n = pe_filter.n_inc;

    // make sure unnecessary bits are masked off
    filter_n &= BIT_MASK(CLOG2(N_VECTOR));
    filter_read_addr &= BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
    filter_read_fw_vec &= BIT_MASK(CLOG2(FW_VECTOR));
    filter_write_addr &= BIT_MASK(CLOG2(FILTER_CACHE_DEPTH));
    bool conv_start = cont.conv_start;

    conv_done = cont.conv_done[0];

    // save filter and bias data for next n_vec
    if (filter_data_valid && filter_n == n_inc) {
      #pragma unroll
      for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
        filter_cache[filter_write_addr][fw_inc] = filter_data.v[fw_inc];
      }

      // saves bias data and bn parameters to the specific buffer
      if (filter_write_addr == 0 || filter_write_addr == FILTER_CACHE_PAGE_DEPTH) {
        bias_bn_cache[filter_bias_write_page] = bias_bn_data;
      }
    }

    //
    // read filter and bias data for the current input data
    //
    DotFilterVector filter;

    #pragma unroll
    for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
      filter.v[fw_inc] = filter_cache[filter_read_addr][fw_inc];
    }

    BiasBnParam bias_bn = bias_bn_cache[filter_bias_read_page];

    //
    // compute dot product by shifting operation
    //
    Mreal  dot_sum_fw_vec[W_VECTOR] = {0};

    if (cont.is_QVECTOR){
      #pragma unroll
      for (int ow_inc = 0; ow_inc < OW_VECTOR; ow_inc++) {
        #pragma unroll
        for (int fw_inc = 0; fw_inc < FW_VECTOR; fw_inc++) {
          dot_sum_fw_vec[ow_inc] += DotProduct( input_data.v[ow_inc+fw_inc], filter.v[fw_inc]);
#ifdef PRINT_PE_INPUT
          if (n_inc == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) {
            for (int c_inc = 0; c_inc < C_VECTOR; c_inc++ )
              printf ("PE ow_vec=%d fw_vec=%d c_inc=%d input_data=%d filter=%d cycle=%d\frame_index", ow_inc, fw_inc, c_inc, input_data.v[ow_inc+fw_inc].v[c_inc], filter.v[fw_inc].v[c_inc], cycle);
          }
#endif
        }
      }
    } else {
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        dot_sum_fw_vec[w_inc] = DotProduct(input_data.v[w_inc], filter.v[filter_read_fw_vec]);
#ifdef PRINT_PE_INPUT
        if (n == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range) {
          for (int c_inc = 0; c_inc < C_VECTOR; c_inc++)
            printf("PE w_inc=%d c_inc=%d fsvec=%d input_data=%d filter=%d cycle=%d\frame_index", w_inc, c_inc, filter_read_fw_vec, input_data.v[w_inc].v[c_inc], filter.v[filter_read_fw_vec].v[c_inc], cycle);
        }
#endif
      }
    }

    //
    // add the dot product to the current accumulated value
    //
    #pragma unroll
    for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
      Mreal sum = conv_start ? bias_bn.bias : result[w_inc];
      result[w_inc] = sum + dot_sum_fw_vec[w_inc];
    }

    //
    // send out the result
    //
    if (input_data_valid && conv_done) {
      PeOutput pe_output;
      #pragma unroll
      for (int w_inc = 0; w_inc < W_VECTOR; w_inc++) {
        //float bn_data = (result[w_inc] * alpha + beta) * TRANS_INFLAT;
        //int bn_alpha = (result[w_inc] * alpha) >> ALPHA_INFLAT;
        long int bn_alpha_inflat = (long int)result[w_inc] * (long int)bias_bn.alpha;
        int bn_alpha = bn_alpha_inflat >> ALPHA_INFLAT;
        int bn_data = (((bn_alpha + bias_bn.beta) >> (INFLAT - 1)) + 1) >> 1;
        pe_output.data.v[w_inc] = bn_data > REALMAX ? REALMAX : bn_data < REALMIN ? REALMIN : bn_data;
        pe_output.pe_output_relu = cont.pe_output_relu;
#ifdef PRINT_PE_OUTPUT
        if (n_inc == PRINT_N && cycle >= debug_cycle && cycle < debug_cycle + debug_range)
          printf("PE cycle=%d w_inc=%d result=%d bias_bn.alpha=%d bias_bn.beta=%d pe_output.data.v=%d\frame_index", cycle, w_inc, result[w_inc], bias_bn.alpha, bias_bn.beta, pe_output.data.v[w_inc]);
#endif
      }

      write_channel_intel(pe_output_channel[n_inc], pe_output);
    }

    INCREASE_COUNTER(cycle);
    if (COUNTER_DONE(cycle)) { RESET_COUNTER(cycle); }
  } while (1);

}

__attribute__((autorun)) TASK kernel void pe_tail() {

  while (1) {
    bool valid = false;

    PeInputData pe_input_data = read_channel_nb_intel(pe_input_data_channel[N_VECTOR-1], &valid);

    PeInputFilter pe_input_filter = read_channel_nb_intel(pe_input_filter_channel[N_VECTOR-1], &valid);

    PeControlSignal pe_control = read_channel_nb_intel(pe_control_channel[N_VECTOR-1], &valid);
  }
}

void pe_drain(int n_inc) {
  //printf("pe_drain - strat...\n");
  INIT_COUNTER(nn_vec);

  bool have_data_to_forward = false;
  PeOutput output_data = {{{0}}};

  while(1) {
    SET_COUNTER(nn_vec, CEIL(N_VECTOR, NARROW_N_VECTOR), 0, CEIL(N_VECTOR, NARROW_N_VECTOR), 1);

    bool increment_counter = false;

    if (!have_data_to_forward) {
      // once every MYCEIL(N_VECTOR, RELU_N_VECTOR) cycles, we read from our
      // cooresponding PE kernel and foward that data
      if (nn_vec == n_inc / NARROW_N_VECTOR) {
        output_data = read_channel_nb_intel(pe_output_channel[n_inc], &have_data_to_forward);
      // for cycles that happen before our "turn" we need to forward the data
      // from upstream
      } else if (n_inc >= NARROW_N_VECTOR && nn_vec < n_inc / NARROW_N_VECTOR) {
        output_data = read_channel_nb_intel(pe_drain_output_channel[n_inc - NARROW_N_VECTOR], &have_data_to_forward);
      } else {
        increment_counter = true;
      }
    }

    if (have_data_to_forward) {
      bool write_success = write_channel_nb_intel(pe_drain_output_channel[n_inc], output_data);
      if (write_success) {
        have_data_to_forward = false;
        increment_counter = true;
      }
    }

    if (increment_counter) {
      INCREASE_COUNTER(nn_vec);
      if (COUNTER_DONE(nn_vec)) { RESET_COUNTER(nn_vec); }
    }
  }
}

#define PE_KERNEL(X) __attribute__((autorun)) TASK kernel void pe_kernel_##X() { PeFunction(X); }
#define PE_DRAIN(X) __attribute__((autorun)) TASK kernel void pe_drain_##X() { pe_drain(X); }

PE_KERNEL(0);
PE_DRAIN(0);
#if (N_VECTOR > 1)
PE_KERNEL(1);
PE_DRAIN(1);
#if (N_VECTOR > 2)
PE_KERNEL(2);
PE_DRAIN(2);
#if (N_VECTOR > 3)
PE_KERNEL(3);
PE_DRAIN(3);
#if (N_VECTOR > 4)
PE_KERNEL(4);
PE_DRAIN(4);
#if (N_VECTOR > 5)
PE_KERNEL(5);
PE_DRAIN(5);
#if (N_VECTOR > 6)
PE_KERNEL(6);
PE_DRAIN(6);
#if (N_VECTOR > 7)
PE_KERNEL(7);
PE_DRAIN(7);
#if (N_VECTOR > 8)
PE_KERNEL(8);
PE_DRAIN(8);
#if (N_VECTOR > 9)
PE_KERNEL(9);
PE_DRAIN(9);
#if (N_VECTOR > 10)
PE_KERNEL(10);
PE_DRAIN(10);
#if (N_VECTOR > 11)
PE_KERNEL(11);
PE_DRAIN(11);
#if (N_VECTOR > 12)
PE_KERNEL(12);
PE_DRAIN(12);
#if (N_VECTOR > 13)
PE_KERNEL(13);
PE_DRAIN(13);
#if (N_VECTOR > 14)
PE_KERNEL(14);
PE_DRAIN(14);
#if (N_VECTOR > 15)
PE_KERNEL(15);
PE_DRAIN(15);
#if (N_VECTOR > 16)
PE_KERNEL(16);
PE_DRAIN(16);
#if (N_VECTOR > 17)
PE_KERNEL(17);
PE_DRAIN(17);
#if (N_VECTOR > 18)
PE_KERNEL(18);
PE_DRAIN(18);
#if (N_VECTOR > 19)
PE_KERNEL(19);
PE_DRAIN(19);
#if (N_VECTOR > 20)
PE_KERNEL(20);
PE_DRAIN(20);
#if (N_VECTOR > 21)
PE_KERNEL(21);
PE_DRAIN(21);
#if (N_VECTOR > 22)
PE_KERNEL(22);
PE_DRAIN(22);
#if (N_VECTOR > 23)
PE_KERNEL(23);
PE_DRAIN(23);
#if (N_VECTOR > 24)
PE_KERNEL(24);
PE_DRAIN(24);
#if (N_VECTOR > 25)
PE_KERNEL(25);
PE_DRAIN(25);
#if (N_VECTOR > 26)
PE_KERNEL(26);
PE_DRAIN(26);
#if (N_VECTOR > 27)
PE_KERNEL(27);
PE_DRAIN(27);
#if (N_VECTOR > 28)
PE_KERNEL(28);
PE_DRAIN(28);
#if (N_VECTOR > 29)
PE_KERNEL(29);
PE_DRAIN(29);
#if (N_VECTOR > 30)
PE_KERNEL(30);
PE_DRAIN(30);
#if (N_VECTOR > 31)
PE_KERNEL(31);
PE_DRAIN(31);
#if (N_VECTOR > 32)
PE_KERNEL(32);
PE_DRAIN(32);
#if (N_VECTOR > 33)
PE_KERNEL(33);
PE_DRAIN(33);
#if (N_VECTOR > 34)
PE_KERNEL(34);
PE_DRAIN(34);
#if (N_VECTOR > 35)
PE_KERNEL(35);
PE_DRAIN(35);
#if (N_VECTOR > 36)
PE_KERNEL(36);
PE_DRAIN(36);
#if (N_VECTOR > 37)
PE_KERNEL(37);
PE_DRAIN(37);
#if (N_VECTOR > 38)
PE_KERNEL(38);
PE_DRAIN(38);
#if (N_VECTOR > 39)
PE_KERNEL(39);
PE_DRAIN(39);
#if (N_VECTOR > 40)
PE_KERNEL(40);
PE_DRAIN(40);
#if (N_VECTOR > 41)
PE_KERNEL(41);
PE_DRAIN(41);
#if (N_VECTOR > 42)
PE_KERNEL(42);
PE_DRAIN(42);
#if (N_VECTOR > 43)
PE_KERNEL(43);
PE_DRAIN(43);
#if (N_VECTOR > 44)
PE_KERNEL(44);
PE_DRAIN(44);
#if (N_VECTOR > 45)
PE_KERNEL(45);
PE_DRAIN(45);
#if (N_VECTOR > 46)
PE_KERNEL(46);
PE_DRAIN(46);
#if (N_VECTOR > 47)
PE_KERNEL(47);
PE_DRAIN(47);
#if (N_VECTOR > 48)
PE_KERNEL(48);
PE_DRAIN(48);
#if (N_VECTOR > 49)
PE_KERNEL(49);
PE_DRAIN(49);
#if (N_VECTOR > 50)
PE_KERNEL(50);
PE_DRAIN(50);
#if (N_VECTOR > 51)
PE_KERNEL(51);
PE_DRAIN(51);
#if (N_VECTOR > 52)
PE_KERNEL(52);
PE_DRAIN(52);
#if (N_VECTOR > 53)
PE_KERNEL(53);
PE_DRAIN(53);
#if (N_VECTOR > 54)
PE_KERNEL(54);
PE_DRAIN(54);
#if (N_VECTOR > 55)
PE_KERNEL(55);
PE_DRAIN(55);
#if (N_VECTOR > 56)
PE_KERNEL(56);
PE_DRAIN(56);
#if (N_VECTOR > 57)
PE_KERNEL(57);
PE_DRAIN(57);
#if (N_VECTOR > 58)
PE_KERNEL(58);
PE_DRAIN(58);
#if (N_VECTOR > 59)
PE_KERNEL(59);
PE_DRAIN(59);
#if (N_VECTOR > 60)
PE_KERNEL(60);
PE_DRAIN(60);
#if (N_VECTOR > 61)
PE_KERNEL(61);
PE_DRAIN(61);
#if (N_VECTOR > 62)
PE_KERNEL(62);
PE_DRAIN(62);
#if (N_VECTOR > 63)
PE_KERNEL(63);
PE_DRAIN(63);
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
