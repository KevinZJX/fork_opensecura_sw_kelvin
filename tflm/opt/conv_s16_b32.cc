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
// Data types: input: s16, filter: s8, bias s32

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {
namespace {
void ConvS16B32Generic(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto stride_width = params.stride_width;
  const auto stride_height = params.stride_height;
  const auto dilation_width_factor = params.dilation_width_factor;
  const auto dilation_height_factor = params.dilation_height_factor;
  const auto pad_width = params.padding_values.width;
  const auto pad_height = params.padding_values.height;
  const auto input_height = input_shape.Dims(1);
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto filter_height = filter_shape.Dims(1);
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_height = output_shape.Dims(1);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;
  const auto groups = input_depth / filter_depth;
  const auto filters_per_group = output_depth / groups;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = out_y * stride_height - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_width - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int32_t acc32 = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const bool inside = (in_x >= 0) && (in_x < input_width) &&
                                  (in_y >= 0) && (in_y < input_height);
              if (!inside) {
                continue;
              }
              int in_channel = 0;
              do {
                int load_count = std::min(filter_depth - in_channel, 16L);
                int32_t input_swizzled[16];
                const int16_t* p_input = &input_data[tflite::Offset(
                    input_shape, batch, in_y, in_x,
                    in_channel + group * filter_depth)];
                for (int i = 0; i < 16; ++i) {
                  int swizzle_idx = swizzle[i];
                  if (swizzle_idx < load_count)
                    input_swizzled[i] = *(p_input + swizzle_idx) + input_offset;
                  else
                    input_swizzled[i] = 0;
                }
                vld_w_l_xx(v0, input_swizzled, 4);
                vld_w_l_xx(v1, input_swizzled + 4, 4);
                vld_w_l_xx(v2, input_swizzled + 8, 4);
                vld_w_l_xx(v3, input_swizzled + 12, 4);
                vld_b_l_xx(v4,
                           &filter_data[tflite::Offset(filter_shape,
                                                       out_channel, filter_y,
                                                       filter_x, in_channel)],
                           load_count);
                vaddw_h_vx(v4, v4, 0);
                vaddw_w_vx(v6, v5, 0);
                vaddw_w_vx(v4, v4, 0);

                vmul_w_vv_m(vm0, vm0, vm1);
                vadd_w_vv(v0, v0, v1);
                vadd_w_vv(v0, v0, v2);
                vadd_w_vv(v0, v0, v3);
                int32_t acc_spill[4];
                vst_w_l_xx(v0, acc_spill, 4);
                for (int i = 0; i < 4; ++i) {
                  acc32 += acc_spill[i];
                }
                in_channel += 16;
              } while (in_channel + 16 <= filter_depth);
            }
          }
          if (bias_data) {
            acc32 = acc32 + bias_data[out_channel];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
              acc32, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          output_data[tflite::Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] = static_cast<int16_t>(acc);
        }
      }
    }
  }
}
}  // namespace

void ConvS16B32(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  // generic implementation by default
  auto fn = ConvS16B32Generic;

  // can add special cases below

  fn(params, output_multiplier, output_shift, input_shape, input_data,
     filter_shape, filter_data, bias_shape, bias_data, output_shape,
     output_data);
}

}  // namespace kelvin::opt
