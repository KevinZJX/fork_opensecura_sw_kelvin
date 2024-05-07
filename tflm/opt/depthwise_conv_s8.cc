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

// Depthwise convolution based on Kelvin ops
// Data types: input: s8, filter: s8, bias s32

#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tflm/opt/conv_util.h"

namespace kelvin::opt {
namespace {
// special case of input depth = 32n
void DepthwiseConvS8D32(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data

) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  int32_t swizzled_bias_data[32 * 4];
  int32_t swizzled_shift_multi[32 * 4];
  int32_t swizzled_output_multi[32 * 4];

  for (int in_channel = 0; in_channel + 32 <= input_depth; in_channel += 32) {
    const int output_channel = in_channel;
    Swizzle(bias_data + output_channel, swizzled_bias_data, 32);
    Swizzle(output_multiplier + output_channel, swizzled_output_multi, 32);
    Swizzle(output_shift + output_channel, swizzled_shift_multi, 32);

    vld_w_x_m(v20, swizzled_bias_data);
    vld_w_x_m(v24, swizzled_output_multi);
    vld_w_x_m(v28, swizzled_shift_multi);
    vrsub_w_vx_m(v28, v28, 0);

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + filter_y;
            if ((in_y < 0) || (in_y >= input_height)) {
              continue;
            }
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              if ((in_x < 0) || (in_x >= input_width)) {
                continue;
              }

              vld_b_x(v0, &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x, in_channel)]);  // xp
              vld_b_x(v4, &filter_data[tflite::Offset(filter_shape, 0, filter_y,
                                                      filter_x, in_channel)]);

              vaddw_h_vx(v0, v0, 0);
              vadd_h_vx(v0, v0, static_cast<int16_t>(input_offset));
              vadd_h_vx(v1, v1,
                        static_cast<int16_t>(input_offset));  // v0 v1 input

              vaddw_h_vx(v4, v4, static_cast<int16_t>(0));
              vmulw_w_vv(v8, v0, v4);
              vmulw_w_vv(v10, v1, v5);

              vadd_w_vv_m(v48, v48, v8);
            }
          }

          vadd_w_vv_m(v48, v48, v20);  // add bias
          vdmulh_w_r_vv_m(v48, v48, v24);
          vsha_w_r_vv_m(v48, v48, v28);
          vadd_w_vx_m(v48, v48, output_offset);
          vmax_w_vx_m(v48, v48, output_activation_min);
          vmin_w_vx_m(v48, v48, output_activation_max);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, &output_data[tflite::Offset(output_shape, batch, out_y,
                                                   out_x, output_channel)]);
        }
      }
    }
  }
}

// generic implementation based on Kelvin ops
void DepthwiseConvS8Generic(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // TBD: Use Kelvin implementation to replace the below
  tflite::reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multiplier, output_shift, input_shape, input_data,
      filter_shape, filter_data, bias_shape, bias_data, output_shape,
      output_data);
  return;
}
}  // namespace

void DepthwiseConvS8(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_depth = input_shape.Dims(3);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  if (depth_multiplier == 1 && pad_height < 2 && pad_width < 2 &&
      dilation_height_factor == 1 && dilation_width_factor == 1 &&
      stride_height == 1 && stride_width == 1) {
    // generic implementation by default
    auto fn = DepthwiseConvS8Generic;

    // special case of output depth = 32n
    if (output_depth % 32 == 0) {
      fn = DepthwiseConvS8D32;
    }

    fn(params, output_multiplier, output_shift, input_shape, input_data,
       filter_shape, filter_data, bias_shape, bias_data, output_shape,
       output_data);
    return;
  }

  // Use reference implementation
  tflite::reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multiplier, output_shift, input_shape, input_data,
      filter_shape, filter_data, bias_shape, bias_data, output_shape,
      output_data);
}

}  // namespace kelvin::opt
