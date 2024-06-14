/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Convolution based on Kelvin ops
// Data types: input: s8, filter: s8, bias: s32
// Special case for 1x1 filter

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {

void ConvS8K1x1DMod32(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  union {
    vconv_u8_t conv;
    uint32_t raw;
  } cmds;
  cmds.conv.mode = 0;
  cmds.conv.start = 0;
  cmds.conv.stop = 7;
  cmds.conv.sbias1 = input_offset;
  cmds.conv.sdata1 = true;
  cmds.conv.sbias2 = 0;
  cmds.conv.sdata2 = true;

  const size_t swizzled_filter_data_size =
      8 * filter_input_depth;
  std::unique_ptr<int8_t> swizzled_filter_data(reinterpret_cast<int8_t*>(
      ::aligned_alloc(32, swizzled_filter_data_size)));
  int8_t* p_swizzled_filter_data = swizzled_filter_data.get();
  int32_t swizzled_bias_data[32];
  int32_t swizzled_mult_data[32];
  int32_t swizzled_shift_data[32];

  const int n_elems = (output_width * batches * output_height);
  int out_channel = 0;
  do {
    int out_channels_this_iter = std::min(8, output_depth - out_channel);
    Filter_N_H_W_M(filter_data + (out_channel * filter_input_depth),
                   p_swizzled_filter_data, out_channels_this_iter, 1, 1,
                   filter_input_depth);
    if (bias_data) {
      Swizzle(bias_data + out_channel, swizzled_bias_data, out_channels_this_iter);
      vld_w_x_m(v16, swizzled_bias_data);
    } else {
      vdup_w_x_m(v16, 0);
    }
    Swizzle(output_multiplier + out_channel, swizzled_mult_data, out_channels_this_iter);
    Swizzle(output_shift + out_channel, swizzled_shift_data, out_channels_this_iter);

    vld_w_x_m(v20, swizzled_mult_data);
    vld_w_x_m(v24, swizzled_shift_data);
    vrsub_w_vx_m(v24, v24, 0);

    int out = 0;
    for (; out < n_elems; out += 8) {
      int out_this_iter = std::min(8, n_elems - out);

      const int8_t* p_in = input_data + (out * input_depth);
      int8_t* p_out = output_data + (out * output_depth) + out_channel;

      // 8x accumulators
      vmv_v_m(v48, v16);
      vmv_v_m(v52, v16);
      acset_v(v48, v48);
      int in_channel = 0;
      for (; in_channel < filter_input_depth; in_channel += 32) {
        const int8_t* p_input = p_in + in_channel;
        if (out_this_iter < 8) {
          switch (out_this_iter) {
            case 7:
              vld_b_x(v6, p_input + (6 * input_depth));
            case 6:
              vld_b_x(v5, p_input + (5 * input_depth));
            case 5:
              vld_b_x(v4, p_input + (4 * input_depth));
            case 4:
              vld_b_x(v3, p_input + (3 * input_depth));
            case 3:
              vld_b_x(v2, p_input + (2 * input_depth));
            case 2:
              vld_b_x(v1, p_input + input_depth);
            case 1:
              vld_b_x(v0, p_input);
          }
        } else {
          // Inputs
          vld_b_s_xx_m(v0, p_input, input_depth);
          vld_b_s_xx_m(v4, p_input + (4 * input_depth), input_depth);
        }

        int8_t* p_local_filter = p_swizzled_filter_data + (in_channel * 8);
        vld_b_x_m(v8, p_local_filter);
        vld_b_x_m(v12, p_local_filter + (4 * 32));
        aconv_vxv(v48, v0, cmds, v8);
      }

      vcget(v48);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          v48, v52, v20, v24, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(v48, v48, 0);
      vsraqs_b_vx(v52, v52, 0);

      int i = 0;
      for (; i < std::min(4, out_this_iter); i++) {
        vst_b_l_xx(v48, p_out + (i * output_depth), out_channels_this_iter);
        // p_out += output_depth;
        vsliden_h_4_vv(v48, v48, v48);
      }
      for (; i < out_this_iter; i++) {
        vst_b_l_xx(v52, p_out + (i * output_depth), out_channels_this_iter);
        // p_out += output_depth;
        vsliden_h_4_vv(v52, v52, v52);
      }
    }

    out_channel += out_channels_this_iter;
  } while (out_channel < output_depth);
}

void ConvS8K1x1D32(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  assert(input_depth == 32);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int filter_input_depth = filter_shape.Dims(3);
      assert(filter_input_depth == 32);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  union {
    vconv_u8_t conv;
    uint32_t raw;
  } cmds;
  cmds.conv.mode = 0;
  cmds.conv.start = 0;
  cmds.conv.stop = 7;
  cmds.conv.sbias1 = input_offset;
  cmds.conv.sdata1 = true;
  cmds.conv.sbias2 = 0;
  cmds.conv.sdata2 = true;
  const size_t swizzled_filter_data_size =
      8 * filter_input_depth;
  std::unique_ptr<int8_t> swizzled_filter_data(reinterpret_cast<int8_t*>(
      ::aligned_alloc(32, swizzled_filter_data_size)));
  int8_t* p_swizzled_filter_data = swizzled_filter_data.get();
  int32_t swizzled_bias_data[32];
  int32_t swizzled_mult_data[32];
  int32_t swizzled_shift_data[32];

// v0-v7 INPUT0
// v8-v15 FLT
// v16-v23 INPUT1
// v24-v27 ACCB
// v32-v43 unused
// v44-47 BIAS
// v48-v55 aconv ACC
// v56-v59 MULT
// v60-v63 BIAS
#define INPUT0_0 v0
#define INPUT0_1 v4
#define FLT0_0 v8
#define FLT0_1 v12
#define INPUT1_0 v16
#define INPUT1_1 v20
#define ACCB0 v24
#define ACCB1 v25
#define ACCB2 v26
#define ACCB3 v27
#define ACCB4 v28
#define ACCB5 v29
#define ACCB6 v30
#define ACCB7 v31
#define BIAS0 v44
#define ACC0 v48
#define ACC1 v52
#define MULT0 v56
#define SHFT0 v60

  const int n_elems = (output_width * batches * output_height);
  int out_channel = 0;
  do { // out_channel
    int out_channels_this_iter = std::min(8, output_depth - out_channel);
    assert(out_channels_this_iter == 8);
    Filter_N_H_W_M(filter_data + (out_channel * filter_input_depth),
                   p_swizzled_filter_data, out_channels_this_iter, 1, 1,
                   filter_input_depth);
    if (bias_data) {
      Swizzle(bias_data + out_channel, swizzled_bias_data, out_channels_this_iter);
      vld_w_x_m(BIAS0, swizzled_bias_data);
    } else {
      vdup_w_x_m(BIAS0, 0);
    }
    Swizzle(output_multiplier + out_channel, swizzled_mult_data, out_channels_this_iter);
    Swizzle(output_shift + out_channel, swizzled_shift_data, out_channels_this_iter);

    vld_w_x_m(MULT0, swizzled_mult_data);
    vld_w_x_m(SHFT0, swizzled_shift_data);
    vrsub_w_vx_m(SHFT0, SHFT0, 0);

    int out = 0;
    do {
      const int8_t* p_in = input_data + (out * input_depth);
      int8_t* p_out = output_data + (out * output_depth) + out_channel;

      // 8x accumulators
      vmv_v_m(ACC0, BIAS0);
      vmv_v_m(ACC1, BIAS0);

      acset_v(ACC0, ACC0);
      const int8_t* p_input = p_in;
       // Inputs
      vld_b_s_xx_m(INPUT0_0, p_input, input_depth);
      vld_b_s_xx_m(INPUT0_1, p_input + (4 * input_depth), input_depth);

      int8_t* p_local_filter = p_swizzled_filter_data;
      vld_b_x_m(FLT0_0, p_local_filter);
      vld_b_x_m(FLT0_1, p_local_filter + (4 * 32));

      aconv_vxv(ACC0, INPUT0_0, cmds, FLT0_0);
      vld_b_s_xx_m(INPUT1_0, p_input + (8 * input_depth), input_depth);
      vld_b_s_xx_m(INPUT1_1, p_input + ( 12 * input_depth), input_depth);

      vcget(ACC0);
      vmv_v_m(INPUT0_0, ACC0);
      vmv_v_m(INPUT0_1, ACC1);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          INPUT0_0, INPUT0_1, MULT0, SHFT0, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(ACCB0, INPUT0_0, 0);
      vsraqs_b_vx(ACCB1, INPUT0_1, 0);

      vmv_v_m(ACC0, BIAS0);
      vstq_b_s_xx(ACCB0, p_out, output_depth);
      vstq_b_s_xx(ACCB1, p_out + (4 * output_depth), output_depth);
      vmv_v_m(ACC1, BIAS0);
      acset_v(ACC0, ACC0);

      aconv_vxv(ACC0, INPUT1_0, cmds, FLT0_0);
      vld_b_s_xx_m(INPUT0_0, p_input + (16 * input_depth), input_depth);
      vld_b_s_xx_m(INPUT0_1, p_input + (20 * input_depth), input_depth);
      vcget(ACC0);
      vmv_v_m(INPUT1_0, ACC0);
      vmv_v_m(INPUT1_1, ACC1);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          INPUT1_0, INPUT1_1, MULT0, SHFT0, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(ACCB2, INPUT1_0, 0);
      vsraqs_b_vx(ACCB3, INPUT1_1, 0);

      vld_b_s_xx_m(INPUT1_0, p_input + (24 * input_depth), input_depth);
      vld_b_s_xx_m(INPUT1_1, p_input + (28 * input_depth), input_depth);

      vmv_v_m(ACC0, BIAS0);
      vstq_b_s_xx(ACCB2, p_out + (8 * output_depth), output_depth);
      vstq_b_s_xx(ACCB3, p_out + (12 * output_depth), output_depth);
      vmv_v_m(ACC1, BIAS0);
      acset_v(ACC0, ACC0);
      aconv_vxv(ACC0, INPUT0_0, cmds, FLT0_0);
      vcget(ACC0);
      vmv_v_m(INPUT0_0, ACC0);
      vmv_v_m(INPUT0_1, ACC1);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          INPUT0_0, INPUT0_1, MULT0, SHFT0, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(ACCB4, INPUT0_0, 0);
      vsraqs_b_vx(ACCB5, INPUT0_1, 0);

      vmv_v_m(ACC0, BIAS0);
      vstq_b_s_xx(ACCB4, p_out + (16 * output_depth), output_depth);
      vstq_b_s_xx(ACCB5, p_out + (20 * output_depth), output_depth);
      vmv_v_m(ACC1, BIAS0);
      acset_v(ACC0, ACC0);
      aconv_vxv(ACC0, INPUT1_0, cmds, FLT0_0);
      vcget(ACC0);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          ACC0, ACC1, MULT0, SHFT0, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(ACCB6, ACC0, 0);
      vsraqs_b_vx(ACCB7, ACC1, 0);

      vstq_b_s_xx(ACCB6, p_out + (24 * output_depth), output_depth);
      vstq_b_s_xx(ACCB7, p_out + (28 * output_depth), output_depth);

      out += 32;
    } while ((n_elems - out) >= 32);
    do {// remainder loop
      int out_this_iter = std::min(8, n_elems - out);
      const int8_t* p_in = input_data + (out * input_depth);
      int8_t* p_out = output_data + (out * output_depth) + out_channel;

      // 8x accumulators
      vmv_v_m(ACC0, BIAS0);
      vmv_v_m(ACC1, BIAS0);
      acset_v(ACC0, ACC0);
      const int8_t* p_input = p_in;
      if (out_this_iter < 8) {
        switch (out_this_iter) {
          case 7:
            vld_b_x(v6, p_input + (6 * input_depth));
          case 6:
            vld_b_x(v5, p_input + (5 * input_depth));
          case 5:
            vld_b_x(v4, p_input + (4 * input_depth));
          case 4:
            vld_b_x(v3, p_input + (3 * input_depth));
          case 3:
            vld_b_x(v2, p_input + (2 * input_depth));
          case 2:
            vld_b_x(v1, p_input + input_depth);
          case 1:
            vld_b_x(INPUT0_0, p_input);
        }
      } else {
        // Inputs
        vld_b_s_xx_m(INPUT0_0, p_input, input_depth);
        vld_b_s_xx_m(INPUT0_1, p_input + (4 * input_depth), input_depth);
      }

      int8_t* p_local_filter = p_swizzled_filter_data;
      vld_b_x_m(FLT0_0, p_local_filter);
      vld_b_x_m(FLT0_1, p_local_filter + (4 * 32));
      aconv_vxv(ACC0, INPUT0_0, cmds, FLT0_0);

      vcget(v48);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          ACC0, ACC1, MULT0, SHFT0, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(ACC0, ACC0, 0);
      vsraqs_b_vx(ACC1, ACC1, 0);

      int i = 0;
      for (; i < std::min(4, out_this_iter); i++) {
        vst_b_l_xx(ACC0, p_out + (i * output_depth), out_channels_this_iter);
        vsliden_h_4_vv(ACC0, ACC0, ACC0);
      }
      for (; i < out_this_iter; i++) {
        vst_b_l_xx(ACC1, p_out + (i * output_depth), out_channels_this_iter);
        vsliden_h_4_vv(ACC1, ACC1, ACC1);
      }
      out += out_this_iter;
    } while (out < n_elems);
    out_channel += out_channels_this_iter;
  } while (out_channel < output_depth);
}

void ConvS8K1x1D16(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  union {
    vconv_u8_t conv;
    uint32_t raw;
  } cmds;
  cmds.conv.mode = 0;
  cmds.conv.start = 0;
  cmds.conv.stop = 3;
  cmds.conv.sbias1 = input_offset;
  cmds.conv.sdata1 = true;
  cmds.conv.sbias2 = 0;
  cmds.conv.sdata2 = true;

  int8_t swizzled_filter_data[8*16];
  int32_t swizzled_bias_data[32];
  int32_t swizzled_mult_data[32];
  int32_t swizzled_shift_data[32];

  const int effective_output_width = batches * output_width * output_height;

#define INPUT v0 // v0, v1, v2, v3, v4, v5, v6, v7
#define INPUT_SLIDE v4
#define FLT_0 v8 // v8, v9, v10, v11
#define FLT_1 v16 // v16, v17, v18, v19
#define BIAS_0 v20
#define BIAS_1 v24
#define MULT_0 v28
#define MULT_1 v32
#define SHFT_0 v36
#define SHFT_1 v40
#define ACC_0 v48
#define ACC_1 v52
#define RES_0 v60
#define RES_1 v61
#define RES_2 v62
#define RES_3 v63

  int out_channel = 0;
  for (; out_channel < output_depth; out_channel += 16) {
    Filter_N_H_W_M(filter_data + (out_channel * 16),
                   swizzled_filter_data, 8, 1, 1, 16);
    vld_b_x_m(FLT_0, swizzled_filter_data);
    Filter_N_H_W_M(filter_data + ((out_channel + 8) * 16), swizzled_filter_data, 8, 1, 1, 16);
    vld_b_x_m(FLT_1, swizzled_filter_data);

    if (bias_data) {
      Swizzle(bias_data + out_channel, swizzled_bias_data, 8);
      vld_w_x_m(BIAS_0, swizzled_bias_data);
      Swizzle(bias_data + out_channel + 8, swizzled_bias_data, 8);
      vld_w_x_m(BIAS_1, swizzled_bias_data);
    } else {
      vdup_w_x_m(BIAS_0, 0);
      vdup_w_x_m(BIAS_1, 0);
    }

    Swizzle(output_multiplier + out_channel, swizzled_mult_data, 8);
    Swizzle(output_shift + out_channel, swizzled_shift_data, 8);
    vld_w_x_m(MULT_0, swizzled_mult_data);
    vld_w_x_m(SHFT_0, swizzled_shift_data);
    vrsub_w_vx_m(SHFT_0, SHFT_0, 0);

    Swizzle(output_multiplier + out_channel + 8, swizzled_mult_data, 8);
    Swizzle(output_shift + out_channel + 8, swizzled_shift_data, 8);
    vld_w_x_m(MULT_1, swizzled_mult_data);
    vld_w_x_m(SHFT_1, swizzled_shift_data);
    vrsub_w_vx_m(SHFT_1, SHFT_1, 0);

    int8_t* p_output = output_data + out_channel;
    int out = 0;
    for (; out + 8 <= effective_output_width; out += 8) {
      // 8x accumulators
      const int8_t* p_in = input_data + (out * input_depth);

      vld_b_x_m(INPUT, p_in);
      vslidevp_w_4_vv_m(INPUT_SLIDE, INPUT, INPUT);

      vmv_v_m(ACC_0, BIAS_0);
      vmv_v_m(ACC_1, BIAS_0);
      acset_v(ACC_0, ACC_0);
      aconv_vxv(ACC_0, INPUT, cmds, FLT_0);

      vcget(ACC_0);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          ACC_0, ACC_1, MULT_0, SHFT_0, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(RES_0, ACC_0, 0);
      vsraqs_b_vx(RES_1, ACC_1, 0);
      vstq_b_s_xx(RES_0, p_output, 2 * output_depth);
      vstq_b_s_xx(RES_1, p_output + output_depth, 2 * output_depth);

      vmv_v_m(ACC_0, BIAS_1);
      vmv_v_m(ACC_1, BIAS_1);
      acset_v(ACC_0, ACC_0);
      aconv_vxv(ACC_0, INPUT, cmds, FLT_1);
      vcget(ACC_0);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          ACC_0, ACC_1, MULT_1, SHFT_1, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(RES_2, ACC_0, 0);
      vsraqs_b_vx(RES_3, ACC_1, 0);
      vstq_b_s_xx(RES_2, p_output + 8, 2 * output_depth);
      vstq_b_s_xx(RES_3, p_output + output_depth + 8, 2 * output_depth);
      p_output += (8 * output_depth);

    }  // out_x

    // Remainder
    int remainder_x = (effective_output_width - out);
    if (remainder_x != 0) {
      const int8_t* p_in = input_data + (out * input_depth);

      // Load inputs
      switch (8 - remainder_x) { // rest (stripmines?)
        case 0:
          vld_b_l_xx(v7, p_in + (7 * input_depth), 16);
        case 1:
          vld_b_l_xx(v6, p_in + (6 * input_depth), 16);
        case 2:
          vld_b_l_xx(v5, p_in + (5 * input_depth), 16);
        case 3:
          vld_b_l_xx(v4, p_in + (4 * input_depth), 16);
        case 4:
          vld_b_l_xx(v3, p_in + (3 * input_depth), 16);
        case 5:
          vld_b_l_xx(v2, p_in + (2 * input_depth), 16);
        case 6:
          vld_b_l_xx(v1, p_in + (1 * input_depth), 16);
        case 7:
          vld_b_l_xx(v0, p_in, 16);
      }

      vmv_v_m(ACC_0, BIAS_0);
      vmv_v_m(ACC_1, BIAS_0);
      acset_v(ACC_0, ACC_0);
      aconv_vxv(ACC_0, INPUT, cmds, FLT_0);
      vcget(ACC_0);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          ACC_0, ACC_1, MULT_0, SHFT_0, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(RES_0, ACC_0, 0);
      vsraqs_b_vx(RES_1, ACC_1, 0);

      vmv_v_m(ACC_0, BIAS_1);
      vmv_v_m(ACC_1, BIAS_1);
      acset_v(ACC_0, ACC_0);
      aconv_vxv(ACC_0, INPUT, cmds, FLT_1);
      vcget(ACC_0);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
          ACC_0, ACC_1, MULT_1, SHFT_1, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(RES_2, ACC_0, 0);
      vsraqs_b_vx(RES_3, ACC_1, 0);

      int i = 0;
      for (; i < std::min(4, remainder_x); i++) {
        vst_b_l_xx(RES_0, p_output, 8);
        vsliden_w_2_vv(RES_0, RES_0, RES_0);
        vst_b_l_xx(RES_2, p_output + 8, 8);
        vsliden_w_2_vv(RES_2, RES_2, RES_2);
        p_output += output_depth;
      }

      for (; i < remainder_x; i++) {
        vst_b_l_xx(RES_1, p_output, 8);
        vsliden_w_2_vv(RES_1, RES_1, RES_1);
        vst_b_l_xx(RES_3, p_output + 8, 8);
        vsliden_w_2_vv(RES_3, RES_3, RES_3);
        p_output += output_depth;
      }
    }
  }
#undef INPUT
#undef INPUT_SLIDE
#undef FLT_0
#undef FLT_1
#undef BIAS_0
#undef BIAS_1
#undef MULT_0
#undef MULT_1
#undef SHFT_0
#undef SHFT_1
#undef ACC_0
#undef ACC_1
#undef RES_0
#undef RES_1
#undef RES_2
#undef RES_3
}

}  // namespace kelvin::opt
