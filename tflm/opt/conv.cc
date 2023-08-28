// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <memory>

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tflm/opt/opt.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {
namespace {
/* clang-format off */
constexpr const int swizzle[16] = {
    0, 4, 8, 12,
    2, 6, 10, 14,
    1, 5, 9, 13,
    3, 7, 11, 15,
};
/* clang-format on */
} // namespace

void conv_per_channel_b32(
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

void conv_per_channel_b64(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
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
          int64_t acc64 = 0;
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
                int32_t acc32[4];
                vst_w_l_xx(v0, acc32, 4);
                for (int i = 0; i < 4; ++i) {
                  acc64 += acc32[i];
                }
                in_channel += 16;
              } while (in_channel + 16 <= filter_depth);
            }
          }
          if (bias_data) {
            acc64 = acc64 + bias_data[out_channel];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
              acc64, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          output_data[tflite::Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] = static_cast<int16_t>(acc);
        }
      }
    }
  }
}

#define INA0 v0
#define FLTA0 v8
#define FLTA1 v9
#define FLTA2 v10
#define FLTA3 v11
#define FLTA4 v12
#define FLTA5 v13
#define FLTA6 v14
#define FLTA7 v15
#define ACC v48
#define ACC0 v48
#define OUT0 v56
void conv_per_channel_b8(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
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

  // Zero out accumulators.
  vdup_b_x(v0, 0);
  acset_v(ACC, v0);
  vdup_b_x_m(ACC0, 0);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; /*out_x += 32*/ ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            const int in_x = in_x_origin + dilation_width_factor * 0;

            // Zero padding by omitting the areas outside the image.
            const bool is_point_inside_image =
                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                (in_y < input_height);
            if (!is_point_inside_image) {
              continue;
            }

            int q = filter_width * filter_depth;
            for (int i = 0; i < q; i += 32) {
              int count = std::min(q - i, 32);
              count = std::min(
                  count, static_cast<int>((input_width - in_x) * filter_depth));
              int input_offset = tflite::Offset(input_shape, batch, in_y, in_x,
                                                group * filter_depth) +
                                 i;
              vdup_w_x_m(vm0, 0);
              vdup_w_x_m(vm1, 0);
              vld_b_l_xx(INA0, &input_data[input_offset], count);
              int filter_offset =
                  tflite::Offset(filter_shape, out_channel, filter_y, 0, 0) + i;
              vdup_w_x_m(FLTA0, 0);
              vdup_w_x_m(FLTA4, 0);
              if (count > 0) {
                vld_b_l_xx(FLTA0, &filter_data[filter_offset],
                           std::min(count, 4));
              }
              if (count > 4) {
                vld_b_l_xx(FLTA1, &filter_data[filter_offset + 4],
                           std::min(count - 4, 4));
              }
              if (count > 8) {
                vld_b_l_xx(FLTA2, &filter_data[filter_offset + 8],
                           std::min(count - 8, 4));
              }
              if (count > 12) {
                vld_b_l_xx(FLTA3, &filter_data[filter_offset + 12],
                           std::min(count - 12, 4));
              }
              if (count > 16) {
                vld_b_l_xx(FLTA4, &filter_data[filter_offset + 16],
                           std::min(count - 16, 4));
              }
              if (count > 20) {
                vld_b_l_xx(FLTA5, &filter_data[filter_offset + 20],
                           std::min(count - 20, 4));
              }
              if (count > 24) {
                vld_b_l_xx(FLTA6, &filter_data[filter_offset + 24],
                           std::min(count - 24, 4));
              }
              if (count > 28) {
                vld_b_l_xx(FLTA7, &filter_data[filter_offset + 28],
                           std::min(count - 28, 4));
              }
              aconv_vxv(ACC, INA0, cmds, FLTA0);
            }
          }
          vcget(ACC);
          vadd_w_vx_m(ACC0, ACC0, bias_data[out_channel]);
          vsll_w_vx_m(ACC0, ACC0, LEFT_SHIFT(output_shift[out_channel]));
          vdmulh_w_r_vx_m(ACC0, ACC0, output_multiplier[out_channel]);
          vsha_w_r_vx_m(ACC0, ACC0, RIGHT_SHIFT(output_shift[out_channel]));
          vadd_w_vx_m(ACC0, ACC0, output_offset);
          vmin_w_vx_m(ACC0, ACC0, output_activation_max);
          vmax_w_vx_m(ACC0, ACC0, output_activation_min);
          vsraqs_b_vx(OUT0, ACC0, 0);
          size_t output_offset =
              tflite::Offset(output_shape, batch, out_y, out_x, out_channel);
          vst_b_l_xx(OUT0, &output_data[output_offset], 1);
        }
      }
    }
  }
}
}  // namespace kelvin::opt
