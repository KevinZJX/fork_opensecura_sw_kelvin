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
namespace {
void ConvS8D32Pw1Ow8Id8(
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
      8 * filter_height * filter_width * filter_input_depth;
  std::unique_ptr<int8_t> swizzled_filter_data(reinterpret_cast<int8_t*>(
      ::aligned_alloc(32, swizzled_filter_data_size)));
  int8_t* p_swizzled_filter_data = swizzled_filter_data.get();
  int32_t swizzled_bias_data[32];
  int32_t swizzled_mult_data[32];
  int32_t swizzled_shift_data[32];

  for (int out_channel = 0; out_channel + 8 <= output_depth; out_channel += 8) {
    Filter_N_H_W_M<8>(filter_data + (out_channel * filter_height *
                                     filter_width * filter_input_depth),
                      p_swizzled_filter_data, filter_height, filter_width,
                      filter_input_depth);
    Swizzle(bias_data + out_channel, swizzled_bias_data, 8);
    Swizzle(output_multiplier + out_channel, swizzled_mult_data, 8);
    Swizzle(output_shift + out_channel, swizzled_shift_data, 8);
    vld_w_x_m(v16, swizzled_bias_data);
    vld_w_x_m(v20, swizzled_mult_data);
    vld_w_x_m(v24, swizzled_shift_data);
    vrsub_w_vx_m(v24, v24, 0);

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x + 8 <= output_width; out_x += 8) {
          // 8x accumulators
          vdup_w_x_m(v48, 0);
          vdup_w_x_m(v52, 0);
          acset_v(v48, v48);
          for (int in_channel = 0; in_channel + 32 <= filter_input_depth;
               in_channel += 32) {
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              const bool is_row_inside_input =
                  (in_y >= 0) && (in_y < input_height);
              if (!is_row_inside_input) {
                continue;
              }

              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                int in_x[8];
                bool left_pad = false;
                bool right_pad = false;
                for (int i = 0; i < 8; ++i) {
                  const int in_x_origin =
                      ((out_x + i) * stride_width) - pad_width;
                  in_x[i] = in_x_origin + dilation_width_factor * filter_x;
                  if (in_x[i] < 0) {
                    left_pad = true;
                  }
                  if (in_x[i] >= input_width) {
                    right_pad = true;
                  }
                }

                if (left_pad) {
                  vdup_b_x(v0, -input_offset);
                  vld_b_s_xx(
                      v1,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[1], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v2,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[2], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v3,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[3], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx_m(
                      v4,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[4], in_channel)],
                      input_depth * stride_width);
                } else if (right_pad) {
                  vld_b_s_xx_m(
                      v0,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[0], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v4,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[4], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v5,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[5], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v6,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[6], in_channel)],
                      input_depth * stride_width);
                  vdup_b_x(v7, -input_offset);
                } else if (!left_pad && !right_pad) {
                  // Inputs
                  vld_b_s_xx_m(
                      v0,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[0], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx_m(
                      v4,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[4], in_channel)],
                      input_depth * stride_width);
                } else {
                  vdup_b_x(v0, -input_offset);
                  vdup_b_x(v7, -input_offset);
                  vld_b_s_xx_m(
                      v1,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[1], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v5,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[5], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v6,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[6], in_channel)],
                      input_depth * stride_width);
                }
                size_t local_filter_offset =
                    (filter_y * filter_width * 8 * input_depth) +
                    (filter_x * 8 * input_depth) + (in_channel * 8);
                int8_t* p_local_filter_start =
                    p_swizzled_filter_data + local_filter_offset;
                vld_b_p_x_m(v8, p_local_filter_start);
                vld_b_x_m(v12, p_local_filter_start);

                aconv_vxv(v48, v0, cmds, v8);
              }
            }
          }
          vcget(v48);
          vadd_w_vv_m(v48, v48, v16);
          vadd_w_vv_m(v52, v52, v16);
          vdmulh_w_r_vv_m(v48, v48, v20);
          vdmulh_w_r_vv_m(v52, v52, v20);
          vsha_w_r_vv_m(v48, v48, v24);
          vsha_w_r_vv_m(v52, v52, v24);
          vadd_w_vx_m(v48, v48, output_offset);
          vadd_w_vx_m(v52, v52, output_offset);
          vmin_w_vx_m(v48, v48, output_activation_max);
          vmin_w_vx_m(v52, v52, output_activation_max);
          vmax_w_vx_m(v48, v48, output_activation_min);
          vmax_w_vx_m(v52, v52, output_activation_min);
          vsraqs_b_vx(v56, v48, 0);
          vsraqs_b_vx(v57, v52, 0);
          vstq_b_s_xx(v56,
                      &output_data[tflite::Offset(output_shape, batch, out_y,
                                                  out_x, out_channel)],
                      output_depth);
          vstq_b_s_xx(v57,
                      &output_data[tflite::Offset(output_shape, batch, out_y,
                                                  out_x + 4, out_channel)],
                      output_depth);
        }
      }
    }
  }
}

}  // namespace

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

  // filter_depth = 32n && input_channels = 8n && output_width = 8n
  if (output_depth % 8 == 0 && output_width % 8 == 0 && pad_width <= 1) {
    ConvS8D32Pw1Ow8Id8(params, output_multiplier, output_shift, input_shape,
                       input_data, filter_shape, filter_data, bias_shape,
                       bias_data, output_shape, output_data);
    return;
  }

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
