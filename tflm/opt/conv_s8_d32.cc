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
// Special case for filter depth = 32n

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {

// Fixed-point per-channel-quantization convolution reference kernel.
void ConvS8D32(const tflite::ConvParams& params,
               const int32_t* output_multiplier, const int32_t* output_shift,
               const tflite::RuntimeShape& input_shape,
               const int8_t* input_data,
               const tflite::RuntimeShape& filter_shape,
               const int8_t* filter_data,
               const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
               const tflite::RuntimeShape& output_shape, int8_t* output_data) {
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
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
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

  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          vdup_w_x_m(v60, 0);
          int32_t acc = 0;
          for (int in_channel = 0; in_channel + 32 <= filter_input_depth;
               in_channel += 32) {
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;

                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);

                if (!is_point_inside_image) {
                  continue;
                }

                vld_b_x(v0, &input_data[tflite::Offset(input_shape, batch, in_y,
                                                       in_x, in_channel)]);
                vaddw_h_vx(v0, v0, 0);
                vadd_h_vx(v0, v0, static_cast<int16_t>(input_offset));
                vadd_h_vx(v1, v1, static_cast<int16_t>(input_offset));
                vld_b_x(v2, &filter_data[tflite::Offset(filter_shape,
                                                        out_channel, filter_y,
                                                        filter_x, in_channel)]);
                vaddw_h_vx(v2, v2, 0);
                vmulw_w_vv(v48, v0, v2);
                vmulw_w_vv(v50, v1, v3);
                vadd_w_vv_m(v60, v60, v48);
              }
            }
          }
          int32_t accumulators[32];
          vst_w_x_m(v60, accumulators);
          for (int i = 0; i < 32; ++i) {
            acc += accumulators[i];
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = tflite::MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[tflite::Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] = static_cast<int8_t>(acc);
        }
      }
    }
  }
}

}  // namespace kelvin::opt
