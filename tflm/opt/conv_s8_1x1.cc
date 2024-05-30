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

void ConvS8K1x1(
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
        vld_b_p_x_m(v8, p_local_filter);
        vld_b_x_m(v12, p_local_filter);

        aconv_vxv(v48, v0, cmds, v8);
      }

      vcget(v48);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
          v48, v20, v24, output_activation_min, output_activation_max,
          output_offset);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
          v52, v20, v24, output_activation_min, output_activation_max,
          output_offset);
      vsraqs_b_vx(v48, v48, 0);
      vsraqs_b_vx(v52, v52, 0);

      int i = 0;
      for (; i < std::min(4, out_this_iter); i++) {
        vst_b_l_xx(v48, p_out, out_channels_this_iter);
        p_out += output_depth;
        vsliden_h_4_vv(v48, v48, v48);
      }
      for (; i < out_this_iter; i++) {
        vst_b_l_xx(v52, p_out, out_channels_this_iter);
        p_out += output_depth;
        vsliden_h_4_vv(v52, v52, v52);
      }
    }

    out_channel += out_channels_this_iter;
  } while (out_channel < output_depth);
}

}  // namespace kelvin::opt
