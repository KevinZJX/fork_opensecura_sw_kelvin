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

#include <algorithm>

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {
namespace {
void JumptableSwizzle(const int32_t* input, int32_t* output, int n) {
  switch (n) {
    case 32:
      output[7] = input[28];
      output[15] = input[30];
      output[23] = input[29];
      output[31] = input[31];
    case 28:
      output[6] = input[24];
      output[14] = input[26];
      output[22] = input[25];
      output[30] = input[27];
    case 24:
      output[5] = input[20];
      output[13] = input[22];
      output[21] = input[21];
      output[29] = input[23];
    case 20:
      output[4] = input[16];
      output[12] = input[18];
      output[20] = input[17];
      output[28] = input[19];
    case 16:
      output[27] = input[15];
      output[19] = input[13];
      output[11] = input[14];
      output[3] = input[12];
    case 12:
      output[2] = input[8];
      output[10] = input[10];
      output[18] = input[9];
      output[26] = input[11];
    case 8:
      output[1] = input[4];
      output[9] = input[6];
      output[17] = input[5];
      output[25] = input[7];
    case 4:
      output[0] = input[0];
      output[8] = input[2];
      output[16] = input[1];
      output[24] = input[3];
  }
}
}  // namespace

void ConvPerChannelD1(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = tflite::MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // Scratch pads to juggle data
  const size_t swizzled_filter_data_size = 32 * filter_height * filter_width;
  std::unique_ptr<int8_t> swizzled_filter_data(
      reinterpret_cast<int8_t*>(
          ::aligned_alloc(32, swizzled_filter_data_size)));
  int32_t swizzled_bias_data[32];
  int32_t swizzled_output_multiplier[32];
  int32_t swizzled_output_shift[32];

  for (int out_channel = 0; out_channel < output_depth; out_channel += 32) {
    int n_channels = std::min(32, output_depth - out_channel);

    // Transpose filter for easy loading
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        for (int i = 0; i < n_channels; i++) {
          int filter_location =
              (filter_y * filter_width * 32) + (filter_x * 32) + i;
          swizzled_filter_data.get()[filter_location] = filter_data[
              tflite::Offset(filter_shape, out_channel + i, filter_y, filter_x,
                             0)];
        }
      }
    }

    if (bias_data) {
      JumptableSwizzle(bias_data + out_channel, swizzled_bias_data, n_channels);
      vld_w_x_m(v52, swizzled_bias_data);
    } else {
      vdup_w_x_m(v52, 0);
    }

    JumptableSwizzle(output_multiplier + out_channel,
                     swizzled_output_multiplier, n_channels);
    vld_w_x_m(v56, swizzled_output_multiplier);

    JumptableSwizzle(output_shift + out_channel, swizzled_output_shift,
                     n_channels);
    vld_w_x_m(v60, swizzled_output_shift);
    vrsub_w_vx_m(v60, v60, 0);

    int8_t* local_output_data = output_data + out_channel;

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;

          // Accumulator loop
          vmv_v_m(v48, v52);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            if ((in_y < 0) || (in_y >= input_height)) {
              continue;
            }

            const int8_t* local_input_data = input_data +
                tflite::Offset(input_shape, batch, in_y, 0, 0);
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              if ((in_x < 0) || (in_x >= input_width)) {
                continue;
              }

              int16_t input_val = local_input_data[in_x];
              int16_t input_val16 = static_cast<int16_t>(
                  input_val + input_offset);
              vdup_h_x(v32, input_val16);

              const int8_t* local_filter_data = swizzled_filter_data.get() +
                  (filter_y * filter_width * 32) + (filter_x * 32);
              vld_b_l_xx(v0, local_filter_data, n_channels);
              vaddw_h_vx(v0, v0, 0);

              // Multiply
              vmulw_w_vv(v4, v0, v32);
              vmulw_w_vv(v6, v1, v32);

              // Accumulate
              vadd_w_vv_m(v48, v48, v4);
            }
          }

          // Output pipeline
          vdmulh_w_rn_vv_m(v48, v48, v56);
          vsha_w_r_vv_m(v48, v48, v60);
          vadd_w_vx_m(v48, v48, output_offset);
          vmin_w_vx_m(v48, v48, output_activation_max);
          vmax_w_vx_m(v48, v48, output_activation_min);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_l_xx(v48, output_data, n_channels);
          output_data += output_depth;
        }
      }
    }
  }
}

}  // namespace kelvin::opt